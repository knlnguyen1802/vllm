# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Optional, override

from filelock import FileLock, Timeout

import safetensors

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Cache FileLock objects per path to avoid reallocating locks repeatedly.
_file_locks: dict[str, FileLock] = {}


def _get_file_lock(path: str) -> FileLock:
    lock_path = path + ".lock"
    lock = _file_locks.get(lock_path)
    if lock is None:
        lock = FileLock(lock_path)
        _file_locks[lock_path] = lock
    return lock


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int
    read_count: int = 0
    write_count: int = 0
    deallocate_cache: bool = False

    @staticmethod
    def make_meta(mm_hash, num_token, deallocate_cache=False) -> "MMMeta":
        return MMMeta(
            mm_hash=mm_hash,
            num_token=num_token,
            read_count=0,
            write_count=0,
            deallocate_cache=deallocate_cache
        )


@dataclass
class ECExampleConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class ECExampleConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index
        self._mm_datas_need_loads: dict[str, int] = {}
        # mm_hash -> num_encoder_token for saves
        self._mm_datas_need_saves: dict[str, int] = {}
        # list of mm_hash to update meta (read or write depending on role)
        self._mm_datas_need_update_meta: list[str] = []
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is not None:
            self._storage_path = transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
            logger.debug(transfer_config)
            logger.debug("Shared storage path is %s", self._storage_path)
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")
        
        # Default deallocate_cache flag
        self._deallocate_cache_enabled = transfer_config.get_from_extra_config(
            "deallocate_cache", False
        ) if transfer_config else False
        logger.info(
            "_deallocate_cache_enabled enable is %s",
            self._deallocate_cache_enabled,
        )

        # Protect in-memory pending-state structures from concurrent
        # access if this connector is used from multiple threads.
        self._mm_state_lock = threading.Lock()

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue

            filename = self._generate_filename_debug(mm_data.mm_hash)
            try:
                ec_cache = safetensors.torch.load_file(filename)["ec_cache"].cuda()
                encoder_cache[mm_data.mm_hash] = ec_cache
                logger.debug("Success load encoder cache for hash %s", mm_data.mm_hash)
            except Exception as e:
                logger.error(
                    "Failed to load encoder cache for %s: %s",
                    mm_data.mm_hash,
                    str(e),
                )
            
            # If deallocation is enabled, load metadata (increments read_count)
            if self._deallocate_cache_enabled:
                self.update_mm_meta_read(mm_data.mm_hash)

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        
        filename = self._generate_filename_debug(mm_hash)
        ec_cache = encoder_cache[mm_hash]
        tensors = {"ec_cache": ec_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
        
        # Save metadata with write_count tracking
        # Extract num_token from kwargs if available, default to 0
        num_token = kwargs.get("num_token", 0)
        # Only create and save meta when deallocation behavior is enabled
        if self._deallocate_cache_enabled:
            mm_meta = MMMeta.make_meta(
                mm_hash=mm_hash,
                num_token=num_token,
                deallocate_cache=self._deallocate_cache_enabled,
            )
            self.update_mm_meta_write(mm_meta)
        
        logger.debug("Save cache successful for mm_hash %s", mm_hash)

    def has_caches(
        self,
        request: "Request",
    ) -> list[bool]:
        """
        Check if cache exist externally for each mm_data of request

        Args:
            request (Request): the request object.

        Returns:
            List of bool indicate that ith mm_data exist in cache or not
        """
        result = []
        for feature in request.mm_features:
            result.append(self._found_match_for_mm_data(feature.identifier))
        return result

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
        local_hit: bool,
        remote_hit: bool,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = request.get_num_encoder_tokens(index)
        if remote_hit and not local_hit:
            self._mm_datas_need_loads[mm_hash] = num_encoder_token
        elif not remote_hit and local_hit:
            self._mm_datas_need_saves[mm_hash] = num_encoder_token
        elif remote_hit and local_hit:
            self._mm_datas_need_update_meta.append(mm_hash)

    def maybe_update_remote_cache_state(
        self,
        encoder_cache,
        **kwargs
    ) -> None:
        # First: process any pending saves. These should be executed
        # by the producer role. We always attempt to save when we are
        # a producer and the encoder cache contains the mm_hash.
        processed_saves: list[str] = []
        # Iterate over a snapshot so we can modify the original safely.
        for mm_hash, num_token in list(self._mm_datas_need_saves.items()):
            # Only try to save if this connector is a producer and cache exists.
            if not self.is_producer:
                continue

            if mm_hash not in encoder_cache:
                # Can't save if we don't have the cache locally; skip for now.
                continue

            try:
                # save_caches may raise on IO errors; treat success as processed
                # so we don't retry infinitely. If you prefer retries, change
                # this to only remove on success.
                self.save_caches(
                    encoder_cache=encoder_cache,
                    mm_hash=mm_hash,
                    num_token=num_token,
                )
                processed_saves.append(mm_hash)
            except Exception as e:
                logger.warning("Failed to save cache for %s: %s", mm_hash, str(e))

        # Second: process any pending meta updates. If we are a producer
        # we should increment the write_count (update_mm_meta_write). If we
        # are a consumer/non-producer we increment the read_count
        # (update_mm_meta_read) so deallocation can be observed.
        processed_updates: list[str] = []
        for mm_hash in list(self._mm_datas_need_update_meta):
            try:
                if self.is_producer:
                    # For producers: increment write_count. Try to preserve
                    # the existing num_token by reading existing meta if present.
                    num_token = 0
                    try:
                        meta_filename = self._generate_meta_filename(mm_hash)
                        if os.path.exists(meta_filename):
                            with open(meta_filename, "r") as f:
                                data = json.load(f)
                                num_token = data.get("num_token", 0)
                        else:
                            num_token = kwargs.get("num_token", 0)
                    except Exception:
                        num_token = kwargs.get("num_token", 0)

                    if self._deallocate_cache_enabled:
                        mm_meta = MMMeta.make_meta(
                            mm_hash=mm_hash,
                            num_token=num_token,
                            deallocate_cache=self._deallocate_cache_enabled,
                        )
                        try:
                            self.update_mm_meta_write(mm_meta)
                            processed_updates.append(mm_hash)
                        except Exception as e:
                            logger.warning(
                                "Failed to update meta write for %s: %s",
                                mm_hash,
                                str(e),
                            )
                else:
                    # For consumers: increment read_count which may trigger
                    # deallocation when read_count == write_count.
                    if self._deallocate_cache_enabled:
                        try:
                            self.update_mm_meta_read(mm_hash)
                            processed_updates.append(mm_hash)
                        except Exception as e:
                            logger.warning(
                                "Failed to update meta read for %s: %s",
                                mm_hash,
                                str(e),
                            )
            except Exception as e:
                logger.exception(
                    "Unexpected error while processing meta update for %s: %s",
                    mm_hash,
                    str(e),
                )

        # Remove processed entries under lock to avoid races with other threads
        # that may append new items to the lists/dicts.
        if processed_saves or processed_updates:
            with self._mm_state_lock:
                for mm_hash in processed_saves:
                    self._mm_datas_need_saves.pop(mm_hash, None)
                for mm_hash in processed_updates:
                    try:
                        self._mm_datas_need_update_meta.remove(mm_hash)
                    except ValueError:
                        pass

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECExampleConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(self, mm_hash) -> bool:
        """Check if the cache is hit for the request."""
        filename = self._generate_filename_debug(mm_hash)
        return os.path.exists(filename)

    def _generate_foldername_debug(
        self,
        mm_hash: str,
        create_folder: bool = True,  # <- now defaults to True
    ) -> str:
        """
        Return the folder in which the cache for this mm_hash lives.
        If `create_folder` is True (default) the directory is created
        recursively the first time it is needed.
        """
        foldername = os.path.join(self._storage_path, mm_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(self, mm_hash: str) -> str:
        """
        Return the full path of the safetensors file for this mm_hash.
        Ensures the parent directory exists because
        `_generate_foldername_debug` is called with its default
        (`create_folder=True`).
        """
        foldername = self._generate_foldername_debug(mm_hash)  # <- folder auto-created
        return os.path.join(foldername, "encoder_cache.safetensors")

    def _generate_meta_filename(self, mm_hash: str) -> str:
        """
        Return the full path of the metadata JSON file for this mm_hash.
        """
        foldername = self._generate_foldername_debug(mm_hash)
        return os.path.join(foldername, "meta.json")

    def _generate_meta_log_filename(self, mm_hash: str) -> str:
        """Return the full path of the persistent log file for this mm_hash."""
        foldername = self._generate_foldername_debug(mm_hash)
        return os.path.join(foldername, "meta.log")

    def update_mm_meta_write(self, mm_meta: MMMeta) -> bool:
        """
        Save or update the metadata file for the given mm_hash.
        If the file exists, increment `write_count`.
        Otherwise create it with `write_count=1`.
        Uses exclusive file locking to prevent race conditions.
        
        Args:
            mm_meta (MMMeta): The metadata object to save.
        """
        # No-op when deallocation metadata behavior is disabled.
        if not self._deallocate_cache_enabled:
            logger.debug(
                "update_mm_meta_write skipped because deallocate_cache is "
                "disabled for %s",
                mm_meta.mm_hash,
            )
            return

        meta_filename = self._generate_meta_filename(mm_meta.mm_hash)

        lock = _get_file_lock(meta_filename)
        # Acquire per-file lock before reading/writing metadata
        with lock:
            if os.path.exists(meta_filename):
                # Update existing meta
                with open(meta_filename, "r+") as f:
                    data = json.load(f)
                    data["write_count"] = data.get("write_count", 0) + 1
                    data["mm_hash"] = mm_meta.mm_hash
                    data["num_token"] = mm_meta.num_token
                    data["deallocate_cache"] = mm_meta.deallocate_cache

                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                    logger.debug(
                        "Updated meta for %s, write_count=%d",
                        mm_meta.mm_hash,
                        data["write_count"],
                    )
            else:
                # Create new meta atomic under lock
                with open(meta_filename, "w") as f:
                    data = asdict(mm_meta)
                    data["write_count"] = 1
                    data["read_count"] = 0
                    json.dump(data, f, indent=2)
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                    logger.debug("Created meta for %s, write_count=1", mm_meta.mm_hash)

                    # Create a persistent log file alongside the meta to mark creation.
                    try:
                        log_filename = self._generate_meta_log_filename(mm_meta.mm_hash)
                        with open(log_filename, "w") as lf:
                            lf.write(json.dumps({
                                "mm_hash": mm_meta.mm_hash,
                                "created_at": int(time.time()),
                                "num_token": mm_meta.num_token,
                            }))
                            lf.flush()
                            try:
                                os.fsync(lf.fileno())
                            except Exception:
                                pass
                        logger.debug("Created log file for %s", mm_meta.mm_hash)
                    except Exception as e:
                        logger.warning(
                            "Failed to create log file for %s: %s",
                            mm_meta.mm_hash,
                            str(e),
                        )

    def update_mm_meta_read(self, mm_hash: str) -> bool:
        """
        Load the metadata file for the given mm_hash and increment read_count.
        Uses exclusive file locking to prevent race conditions.
        Handles the case where the file might be deleted by a concurrent reader.
        
        Args:
            mm_hash (str): The hash identifier for the multimodal data.
            
        Returns:
            Optional[MMMeta]: The metadata object if loaded successfully,
                None otherwise.
        """
        # No-op when deallocation metadata behavior is disabled.
        if not self._deallocate_cache_enabled:
            logger.debug(
                "update_mm_meta_read skipped because deallocate_cache is "
                "disabled for %s",
                mm_hash,
            )
            return

        meta_filename = self._generate_meta_filename(mm_hash)

        lock = _get_file_lock(meta_filename)
        with lock:
            if not os.path.exists(meta_filename):
                logger.warning(
                    "Meta file not found for %s (may have been deleted)",
                    mm_hash,
                )
                return

            try:
                with open(meta_filename, "r+") as f:
                    data = json.load(f)
                    data["read_count"] = data.get("read_count", 0) + 1

                    # Write back with incremented read_count
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                    read_count = data["read_count"]
                    write_count = data["write_count"]

                    logger.debug(
                        "Loaded meta for %s, read_count=%d, write_count=%d",
                        mm_hash,
                        read_count,
                        write_count,
                    )
                    meta = MMMeta(
                        mm_hash=data["mm_hash"],
                        num_token=data["num_token"],
                        read_count=data["read_count"],
                        write_count=data.get("write_count", 0),
                        deallocate_cache=data.get("deallocate_cache", False),
                    )

                    if read_count == write_count:
                        self.maybe_deallocate_cache(meta)
            except json.JSONDecodeError as e:
                logger.error("Failed to decode meta file for %s: %s", mm_hash, str(e))


    def maybe_deallocate_cache(self, mm_meta: MMMeta) -> None:
        """
        Lazily deallocate cache files when read_count equals write_count.
        This is only called when deallocate_cache flag is set.
        Deletes both metadata and value files.
        
        Args:
            mm_meta (MMMeta): The metadata object to check for deallocation.
        """
        # Respect global flag as well as per-meta flag
        if not self._deallocate_cache_enabled or not mm_meta.deallocate_cache:
            return
            
        meta_filename = self._generate_meta_filename(mm_meta.mm_hash)
        cache_filename = self._generate_filename_debug(mm_meta.mm_hash)
        
        lock = _get_file_lock(meta_filename)
        try:
            with lock:
                if not os.path.exists(meta_filename):
                    logger.debug(
                        "Meta file already deleted for %s during deallocation check",
                        mm_meta.mm_hash,
                    )
                    return

                # Read latest counts under lock
                with open(meta_filename, "r") as f:
                    data = json.load(f)
                    read_count = data.get("read_count", 0)
                    write_count = data.get("write_count", 0)

                # Check if we can deallocate
                if read_count == write_count and read_count > 0:
                    # Delete cache file first
                    try:
                        if os.path.exists(cache_filename):
                            os.remove(cache_filename)
                            logger.info(
                                "Deleted cache file for %s (read=%d, write=%d)",
                                mm_meta.mm_hash,
                                read_count,
                                write_count,
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to delete cache file for %s: %s",
                            mm_meta.mm_hash,
                            str(e),
                        )

                    # Delete metadata file
                    try:
                        if os.path.exists(meta_filename):
                            os.remove(meta_filename)
                            logger.info("Deleted meta file for %s", mm_meta.mm_hash)
                    except Exception as e:
                        logger.warning(
                            "Failed to delete meta file for %s: %s",
                            mm_meta.mm_hash,
                            str(e),
                        )

                    # Also delete the persistent log file if present
                    try:
                        log_filename = self._generate_meta_log_filename(mm_meta.mm_hash)
                        if os.path.exists(log_filename):
                            os.remove(log_filename)
                            logger.info("Deleted log file for %s", mm_meta.mm_hash)
                    except Exception as e:
                        logger.warning(
                            "Failed to delete log file for %s: %s",
                            mm_meta.mm_hash,
                            str(e),
                        )

                    # Try to remove the directory if empty
                    folder = self._generate_foldername_debug(
                        mm_meta.mm_hash, create_folder=False
                    )
                    logger.info("Folder name is %s", folder)
                    try:
                        os.rmdir(folder)
                        logger.info("Removed empty folder for %s", mm_meta.mm_hash)
                    except OSError:
                        # Directory not empty or doesn't exist, ignore
                        pass
        except Exception as e:
            logger.error(
                "Error during deallocation for %s: %s",
                mm_meta.mm_hash,
                str(e),
            )
