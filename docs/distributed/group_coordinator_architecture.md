# DAG Parallelism with GroupCoordinator in vLLM

## Overview

This document details how to implement **Directed Acyclic Graph (DAG) parallelism** using vLLM's `GroupCoordinator` class. In DAG parallelism, each node in the computation graph is itself a distributed parallel group that can use tensor parallelism (TP), pipeline parallelism (PP), or other strategies internally.

## Problem Statement

Traditional parallelism strategies (TP, PP, DP) are designed for single-model execution. However, complex scenarios require:

1. **Multiple model stages** where each stage runs on a dedicated set of GPUs
2. **Internal parallelism** within each stage (TP/PP within the stage)
3. **Inter-stage communication** to pass results between stages
4. **DAG topology** where stages can have multiple inputs/outputs (not just linear pipeline)

**Example Use Case:**
- Stage 1 (DAG Node 1): Draft model with 4 GPUs (TP=2, PP=2)
- Stage 2 (DAG Node 2): Verification model with 4 GPUs (TP=2, PP=2)
- Stage 3 (DAG Node 3): Final output stage with 4 GPUs (TP=2, PP=2)
- Each stage processes independently, then sends results to the next stage

## Architecture: Hierarchical Group Structure

### Conceptual Model

```
┌──────────────────────────────────────────────────────────────────┐
│                    WORLD GROUP (12 ranks)                        │
│                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ All subgroups are children
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ DAG Node 1    │     │ DAG Node 2    │     │ DAG Node 3    │
│ [0, 1, 2, 3]  │     │ [4, 5, 6, 7]  │     │ [8, 9, 10, 11]│
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        │                     │                     │
   TP=2, PP=2            TP=2, PP=2            TP=2, PP=2
        │                     │                     │
  ┌─────┴─────┐         ┌─────┴─────┐         ┌─────┴─────┐
  │ TP Groups │         │ TP Groups │         │ TP Groups │
  │ [0,1]     │         │ [4,5]     │         │ [8,9]     │
  │ [2,3]     │         │ [6,7]     │         │ [10,11]   │
  └───────────┘         └───────────┘         └───────────┘
  ┌─────────┐           ┌─────────┐           ┌─────────┐
  │ PP Grps │           │ PP Grps │           │ PP Grps │
  │ [0,2]   │           │ [4,6]   │           │ [8,10]  │
  │ [1,3]   │           │ [5,7]   │           │ [9,11]  │
  └─────────┘           └─────────┘           └─────────┘
```

### Data Flow

```
Input → DAG Node 1 (ranks 0-3)
         │  Internal: all_reduce across [0,1,2,3]
         │  Output: single tensor from rank 0
         ▼
        DAG Node 2 (ranks 4-7)
         │  Receives tensor at all ranks [4,5,6,7]
         │  Internal: all_reduce across [4,5,6,7]
         │  Output: single tensor from rank 4
         ▼
        DAG Node 3 (ranks 8-11)
         │  Receives tensor at all ranks [8,9,10,11]
         │  Internal: all_reduce across [8,9,10,11]
         │  Final output from rank 8
         ▼
        Output
```

## Implementation Details

### Step 1: Global Initialization

First, initialize the distributed environment for all 12 ranks:

```python
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    get_world_group,
    init_model_parallel_group,
)

# Initialize world group (all 12 ranks)
init_distributed_environment(
    world_size=12,
    rank=current_rank,  # 0-11
    local_rank=local_rank,  # 0-3 on each node
    backend="nccl"
)

# Now _WORLD is initialized with all 12 ranks
world_group = get_world_group()
print(f"Rank {world_group.rank} initialized in world of size {world_group.world_size}")
```

### Step 2: Create DAG Node Groups

Each DAG node is a `GroupCoordinator` containing a subset of ranks:

```python
# Global state
_DAG_NODE_GROUPS = []  # List of GroupCoordinator for each DAG node
_DAG_NODE_ID = -1      # Which DAG node this rank belongs to
_DAG_NODE_TP = None    # TP group within this DAG node
_DAG_NODE_PP = None    # PP group within this DAG node

def initialize_dag_parallelism(
    num_dag_nodes: int = 3,
    ranks_per_node: int = 4,
    tp_per_node: int = 2,
    pp_per_node: int = 2,
    backend: str = "nccl"
):
    """
    Initialize DAG parallelism with hierarchical groups.
    
    Args:
        num_dag_nodes: Number of DAG nodes (stages)
        ranks_per_node: Number of ranks per DAG node
        tp_per_node: Tensor parallel size within each node
        pp_per_node: Pipeline parallel size within each node
        backend: Communication backend
    """
    global _DAG_NODE_GROUPS, _DAG_NODE_ID, _DAG_NODE_TP, _DAG_NODE_PP
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    assert world_size == num_dag_nodes * ranks_per_node, (
        f"World size {world_size} != num_dag_nodes * ranks_per_node "
        f"({num_dag_nodes} * {ranks_per_node})"
    )
    assert ranks_per_node == tp_per_node * pp_per_node, (
        f"ranks_per_node {ranks_per_node} != tp_per_node * pp_per_node "
        f"({tp_per_node} * {pp_per_node})"
    )
    
    # Step 1: Create GroupCoordinator for each DAG node
    # All ranks participate in creating all groups
    for node_id in range(num_dag_nodes):
        node_ranks = list(range(
            node_id * ranks_per_node,
            (node_id + 1) * ranks_per_node
        ))
        
        # Create DAG node group
        # This creates ProcessGroups that all ranks know about
        dag_group = init_model_parallel_group(
            group_ranks=[node_ranks],
            local_rank=get_world_group().local_rank,
            backend=backend,
            group_name=f"dag_node_{node_id}",
            use_device_communicator=True  # Enable custom communicators
        )
        
        _DAG_NODE_GROUPS.append(dag_group)
        
        # Mark which DAG node this rank belongs to
        if rank in node_ranks:
            _DAG_NODE_ID = node_id
    
    print(f"Rank {rank} belongs to DAG node {_DAG_NODE_ID}")
    
    # Step 2: Create TP/PP groups WITHIN this rank's DAG node
    _create_intra_dag_node_groups(
        ranks_per_node=ranks_per_node,
        tp_per_node=tp_per_node,
        pp_per_node=pp_per_node,
        backend=backend
    )

def _create_intra_dag_node_groups(
    ranks_per_node: int,
    tp_per_node: int,
    pp_per_node: int,
    backend: str
):
    """Create TP and PP groups within each DAG node."""
    global _DAG_NODE_TP, _DAG_NODE_PP
    
    rank = torch.distributed.get_rank()
    my_dag_node = _DAG_NODE_GROUPS[_DAG_NODE_ID]
    
    # Base rank of this DAG node
    base_rank = my_dag_node.ranks[0]
    
    # Create TP groups within this DAG node
    # For ranks_per_node=4, tp_per_node=2:
    #   TP groups: [[0,1], [2,3]] or [[4,5], [6,7]] or [[8,9], [10,11]]
    tp_group_ranks = []
    for pp_idx in range(pp_per_node):
        tp_ranks = [
            base_rank + pp_idx * tp_per_node + tp_idx
            for tp_idx in range(tp_per_node)
        ]
        tp_group_ranks.append(tp_ranks)
    
    _DAG_NODE_TP = init_model_parallel_group(
        group_ranks=tp_group_ranks,
        local_rank=get_world_group().local_rank,
        backend=backend,
        group_name=f"dag_node_{_DAG_NODE_ID}_tp",
        use_device_communicator=True
    )
    
    # Create PP groups within this DAG node
    # For ranks_per_node=4, tp_per_node=2:
    #   PP groups: [[0,2], [1,3]] or [[4,6], [5,7]] or [[8,10], [9,11]]
    pp_group_ranks = []
    for tp_idx in range(tp_per_node):
        pp_ranks = [
            base_rank + pp_idx * tp_per_node + tp_idx
            for pp_idx in range(pp_per_node)
        ]
        pp_group_ranks.append(pp_ranks)
    
    _DAG_NODE_PP = init_model_parallel_group(
        group_ranks=pp_group_ranks,
        local_rank=get_world_group().local_rank,
        backend=backend,
        group_name=f"dag_node_{_DAG_NODE_ID}_pp",
        use_device_communicator=False  # PP typically doesn't need custom communicator
    )
    
    print(f"Rank {rank} - TP group: {_DAG_NODE_TP.ranks}, "
          f"PP group: {_DAG_NODE_PP.ranks}")

# Accessor functions
def get_dag_node_group(node_id: int = None):
    """Get DAG node group. If node_id is None, returns current rank's DAG node."""
    if node_id is None:
        node_id = _DAG_NODE_ID
    return _DAG_NODE_GROUPS[node_id]

def get_dag_node_tp_group():
    """Get TP group within current DAG node."""
    return _DAG_NODE_TP

def get_dag_node_pp_group():
    """Get PP group within current DAG node."""
    return _DAG_NODE_PP
```

### Step 3: Communication Patterns

#### Pattern 1: All-Reduce Within DAG Node, Send to Next Node

This is the most common pattern: aggregate results within a stage, then send to the next stage.

```python
def dag_forward_pass(
    input_tensor: torch.Tensor,
    src_node_id: int,
    dst_node_id: int,
    broadcast_to_all_dst_ranks: bool = True
) -> torch.Tensor:
    """
    Forward pass from one DAG node to another.
    
    Flow:
    1. All-reduce within source DAG node
    2. Send result from source node to destination node
    3. Optionally broadcast within destination node
    
    Args:
        input_tensor: Input tensor (can be different on each rank)
        src_node_id: Source DAG node ID
        dst_node_id: Destination DAG node ID
        broadcast_to_all_dst_ranks: If True, send to all dst ranks;
                                     if False, only send to first rank
    
    Returns:
        Output tensor (valid on destination node ranks)
    """
    rank = torch.distributed.get_rank()
    src_group = get_dag_node_group(src_node_id)
    dst_group = get_dag_node_group(dst_node_id)
    
    output_tensor = None
    
    # Phase 1: All-reduce within source DAG node
    if _DAG_NODE_ID == src_node_id:
        # All ranks in source node participate in all-reduce
        reduced_tensor = src_group.all_reduce(input_tensor)
        print(f"Rank {rank}: Reduced tensor in DAG node {src_node_id}")
    
    # Phase 2: Inter-node communication
    if broadcast_to_all_dst_ranks:
        # Send to ALL ranks in destination node
        if _DAG_NODE_ID == src_node_id:
            # Only the first rank in source sends
            if rank == src_group.first_rank:
                for dst_rank in dst_group.ranks:
                    torch.distributed.send(
                        reduced_tensor,
                        dst=dst_rank,
                        tag=src_node_id  # Use tag to identify source
                    )
                    print(f"Rank {rank}: Sent to rank {dst_rank}")
        
        if _DAG_NODE_ID == dst_node_id:
            # All ranks in destination receive
            output_tensor = torch.empty_like(input_tensor)
            torch.distributed.recv(
                output_tensor,
                src=src_group.first_rank,
                tag=src_node_id
            )
            print(f"Rank {rank}: Received from rank {src_group.first_rank}")
    else:
        # Send only to first rank in destination node
        if rank == src_group.first_rank:
            torch.distributed.send(
                reduced_tensor,
                dst=dst_group.first_rank,
                tag=src_node_id
            )
            print(f"Rank {rank}: Sent to rank {dst_group.first_rank}")
        
        if rank == dst_group.first_rank:
            output_tensor = torch.empty_like(input_tensor)
            torch.distributed.recv(
                output_tensor,
                src=src_group.first_rank,
                tag=src_node_id
            )
            print(f"Rank {rank}: Received from rank {src_group.first_rank}")
            
            # Broadcast to other ranks in destination node
            output_tensor = dst_group.broadcast(output_tensor, src=0)
    
    return output_tensor
```

#### Pattern 2: Selective Reception (PP-aware)

When using pipeline parallelism within nodes, only certain ranks need to receive:

```python
def dag_forward_pass_pp_aware(
    input_tensor: torch.Tensor,
    src_node_id: int,
    dst_node_id: int,
    dst_pp_stage: int = 0  # Which PP stage in dst node receives
) -> torch.Tensor:
    """
    Forward pass with pipeline parallelism awareness.
    Only sends to the first PP stage of the destination node.
    
    Args:
        input_tensor: Input tensor
        src_node_id: Source DAG node ID
        dst_node_id: Destination DAG node ID
        dst_pp_stage: Which PP stage in destination should receive (0=first)
    
    Returns:
        Output tensor (valid on destination node's specified PP stage)
    """
    rank = torch.distributed.get_rank()
    src_group = get_dag_node_group(src_node_id)
    dst_group = get_dag_node_group(dst_node_id)
    
    output_tensor = None
    
    # Phase 1: All-reduce within source node (last PP stage only)
    if _DAG_NODE_ID == src_node_id:
        # Get PP group for this DAG node
        src_pp_group = get_dag_node_pp_group()
        
        # Only last PP stage has the final result
        if rank == src_pp_group.last_rank:
            # Gather from all TP ranks in last PP stage first
            src_tp_group = get_dag_node_tp_group()
            reduced_tensor = src_tp_group.all_reduce(input_tensor)
            print(f"Rank {rank}: Last PP stage reduced tensor")
        else:
            reduced_tensor = None
    
    # Phase 2: Determine destination ranks (first PP stage only)
    if _DAG_NODE_ID == dst_node_id:
        dst_pp_group = get_dag_node_pp_group()
        dst_tp_group = get_dag_node_tp_group()
        
        # Only first PP stage receives
        if rank in [dst_group.ranks[i] for i in range(dst_tp_group.world_size)]:
            # This rank is in the first PP stage
            should_receive = True
        else:
            should_receive = False
    else:
        should_receive = False
    
    # Phase 3: Send from source last PP stage to destination first PP stage
    if _DAG_NODE_ID == src_node_id and rank == src_pp_group.last_rank:
        # Find all ranks in destination's first PP stage
        dst_tp_group = get_dag_node_tp_group()  # Would need to get dst's TP group
        # Simplified: send to all TP ranks in first PP stage
        for tp_idx in range(dst_tp_group.world_size):
            dst_rank = dst_group.ranks[tp_idx]  # First PP stage
            torch.distributed.send(
                reduced_tensor,
                dst=dst_rank,
                tag=src_node_id
            )
            print(f"Rank {rank}: Sent to dst rank {dst_rank}")
    
    if should_receive:
        output_tensor = torch.empty_like(input_tensor)
        src_pp_group = get_dag_node_pp_group()
        torch.distributed.recv(
            output_tensor,
            src=src_pp_group.last_rank,  # Receive from src's last PP rank
            tag=src_node_id
        )
        print(f"Rank {rank}: Received from rank {src_pp_group.last_rank}")
    
    return output_tensor
```

#### Pattern 3: DAG with Multiple Inputs

When a DAG node receives from multiple sources:

```python
def dag_multi_input_forward(
    input_tensors: dict[int, torch.Tensor],  # {src_node_id: tensor}
    dst_node_id: int,
    combine_fn: callable = None  # How to combine multiple inputs
) -> torch.Tensor:
    """
    Forward pass where destination receives from multiple source nodes.
    
    Example topology:
        Node 0 ──┐
                 ├──> Node 2
        Node 1 ──┘
    
    Args:
        input_tensors: Dict of {source_node_id: tensor}
        dst_node_id: Destination DAG node ID
        combine_fn: Function to combine multiple tensors (default: sum)
    
    Returns:
        Combined output tensor
    """
    rank = torch.distributed.get_rank()
    dst_group = get_dag_node_group(dst_node_id)
    
    if combine_fn is None:
        combine_fn = lambda tensors: sum(tensors)  # Default: sum
    
    received_tensors = []
    
    # Each source node sends independently
    for src_node_id, input_tensor in input_tensors.items():
        src_group = get_dag_node_group(src_node_id)
        
        # Phase 1: All-reduce within source
        if _DAG_NODE_ID == src_node_id:
            reduced = src_group.all_reduce(input_tensor)
            
            # Send to destination
            if rank == src_group.first_rank:
                for dst_rank in dst_group.ranks:
                    torch.distributed.send(
                        reduced,
                        dst=dst_rank,
                        tag=src_node_id  # Tag identifies source
                    )
        
        # Phase 2: Receive at destination
        if _DAG_NODE_ID == dst_node_id:
            received = torch.empty_like(
                next(iter(input_tensors.values()))  # Use any tensor for shape
            )
            torch.distributed.recv(
                received,
                src=src_group.first_rank,
                tag=src_node_id
            )
            received_tensors.append(received)
            print(f"Rank {rank}: Received from node {src_node_id}")
    
    # Phase 3: Combine inputs at destination
    if _DAG_NODE_ID == dst_node_id:
        output = combine_fn(received_tensors)
        return output
    
    return None
```

### Step 4: Complete Example Usage

```python
# Example: Full forward pass through 3-node DAG
def run_dag_inference(input_data: torch.Tensor):
    """
    Run inference through the DAG:
    Node 0 (Draft) -> Node 1 (Verify) -> Node 2 (Output)
    """
    rank = torch.distributed.get_rank()
    
    # Initialize DAG parallelism
    initialize_dag_parallelism(
        num_dag_nodes=3,
        ranks_per_node=4,
        tp_per_node=2,
        pp_per_node=2,
        backend="nccl"
    )
    
    # Node 0: Process input
    if _DAG_NODE_ID == 0:
        # Each rank processes part of the data (TP-sharded)
        tp_group = get_dag_node_tp_group()
        pp_group = get_dag_node_pp_group()
        
        # Shard input across TP dimension
        input_shard = input_data.chunk(tp_group.world_size, dim=-1)[tp_group.rank_in_group]
        
        # Forward through PP stages
        hidden = model_forward_pp(input_shard, pp_group)
        
        # All-reduce across TP at last PP stage
        if pp_group.is_last_rank:
            output_node_0 = tp_group.all_reduce(hidden)
        else:
            output_node_0 = None
    else:
        output_node_0 = None
    
    # Send Node 0 -> Node 1
    output_node_1 = dag_forward_pass(
        input_tensor=output_node_0 if output_node_0 is not None else torch.zeros(128),
        src_node_id=0,
        dst_node_id=1,
        broadcast_to_all_dst_ranks=True
    )
    
    # Node 1: Process received data
    if _DAG_NODE_ID == 1:
        tp_group = get_dag_node_tp_group()
        pp_group = get_dag_node_pp_group()
        
        # Shard across TP
        input_shard = output_node_1.chunk(tp_group.world_size, dim=-1)[tp_group.rank_in_group]
        
        # Forward through PP stages
        hidden = model_forward_pp(input_shard, pp_group)
        
        # All-reduce at last PP stage
        if pp_group.is_last_rank:
            output_node_1_final = tp_group.all_reduce(hidden)
        else:
            output_node_1_final = None
    else:
        output_node_1_final = None
    
    # Send Node 1 -> Node 2
    output_node_2 = dag_forward_pass(
        input_tensor=output_node_1_final if output_node_1_final is not None else torch.zeros(128),
        src_node_id=1,
        dst_node_id=2,
        broadcast_to_all_dst_ranks=True
    )
    
    # Node 2: Final processing
    if _DAG_NODE_ID == 2:
        tp_group = get_dag_node_tp_group()
        pp_group = get_dag_node_pp_group()
        
        # Shard across TP
        input_shard = output_node_2.chunk(tp_group.world_size, dim=-1)[tp_group.rank_in_group]
        
        # Forward through PP stages
        hidden = model_forward_pp(input_shard, pp_group)
        
        # All-reduce at last PP stage
        if pp_group.is_last_rank:
            final_output = tp_group.all_reduce(hidden)
            
            # Only rank 8 (first rank of node 2) returns final result
            if rank == get_dag_node_group(2).first_rank:
                return final_output
    
    return None

def model_forward_pp(input_tensor, pp_group):
    """Simplified PP forward (placeholder)."""
    # This would do actual PP communication
    # For now, just return input
    return input_tensor
```

## Key Insights

### 1. **All Ranks Create All Groups**

```python
# IMPORTANT: All ranks participate in creating all groups
for node_id in range(3):  # All ranks iterate
    node_ranks = [...]
    dag_group = init_model_parallel_group(
        group_ranks=[node_ranks],  # All ranks call this
        ...
    )
```

Even though rank 0 only belongs to DAG node 0, it still participates in creating the ProcessGroups for nodes 1 and 2. This is required by PyTorch's `torch.distributed.new_group()`.

### 2. **GroupCoordinator Stores Only Relevant Groups**

```python
if self.rank in ranks:
    self.ranks = ranks          # Only store if this rank is in the group
    self.device_group = device_group
    self.cpu_group = cpu_group
```

Inside `GroupCoordinator.__init__`, each rank only saves the group it belongs to.

### 3. **Global vs Local Ranks**

- **Global rank**: `torch.distributed.get_rank()` → 0-11
- **DAG node rank**: Position within the DAG node → 0-3
- **TP rank within node**: Position within TP group → 0-1
- **PP rank within node**: Position within PP group → 0-1

Always use global ranks for point-to-point communication.

### 4. **Communication Cost**

```
Within DAG node:  Fast (local all-reduce via NCCL)
Between DAG nodes: Slower (point-to-point, may cross nodes)
```

Minimize inter-DAG-node communication for best performance.

## Advanced: Broadcast Groups for Efficiency

Instead of multiple point-to-point sends, create broadcast groups:

```python
def create_dag_edge_broadcast_groups():
    """
    Create ProcessGroups for DAG edges to use broadcast instead of
    multiple send/recv operations.
    """
    # Edge: Node 0 -> Node 1
    # Create group with [rank_0_last, all_rank_1]
    src_group_0 = get_dag_node_group(0)
    dst_group_1 = get_dag_node_group(1)
    
    edge_0_to_1_ranks = [src_group_0.first_rank] + dst_group_1.ranks
    
    edge_0_to_1 = init_model_parallel_group(
        group_ranks=[edge_0_to_1_ranks],
        local_rank=get_world_group().local_rank,
        backend="nccl",
        group_name="dag_edge_0_to_1",
        use_device_communicator=False
    )
    
    return edge_0_to_1

# Usage
edge_group = create_dag_edge_broadcast_groups()
if rank in edge_group.ranks:
    # Use broadcast instead of multiple sends
    edge_group.broadcast(tensor, src=0)  # src=0 is the sender
```

This is more efficient when sending to many destination ranks.

## Error Handling and Debugging

### Common Issues

1. **Deadlock**: Ensure all ranks in a group call collective operations
2. **Wrong ranks**: Double-check `_DAG_NODE_ID` vs global `rank`
3. **Tag mismatch**: Use consistent tags for send/recv pairs
4. **Shape mismatch**: Ensure tensor shapes match between sender/receiver

### Debug Utilities

```python
def print_dag_topology():
    """Print the DAG topology for debugging."""
    rank = torch.distributed.get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("DAG Topology:")
        print("="*60)
        for node_id, group in enumerate(_DAG_NODE_GROUPS):
            print(f"DAG Node {node_id}: ranks {group.ranks}")
        print("="*60 + "\n")
    
    # All ranks report their assignments
    torch.distributed.barrier()
    for r in range(torch.distributed.get_world_size()):
        if rank == r:
            print(f"Rank {rank}: DAG node {_DAG_NODE_ID}, "
                  f"TP rank {_DAG_NODE_TP.rank_in_group}, "
                  f"PP rank {_DAG_NODE_PP.rank_in_group}")
        torch.distributed.barrier()
```

## Summary

**DAG Parallelism Architecture:**

1. **World Group**: Contains all ranks (12 ranks)
2. **DAG Node Groups**: Subgroups for each stage ([0-3], [4-7], [8-11])
3. **TP/PP Groups**: Subgroups within each DAG node
4. **Communication**: 
   - Intra-node: Collective operations (all_reduce, broadcast)
   - Inter-node: Point-to-point operations (send/recv) or broadcast groups

**Key Benefits:**

- **Modularity**: Each DAG node is independent
- **Flexibility**: Arbitrary DAG topology (not just linear pipeline)
- **Efficiency**: Local all-reduce + targeted inter-node communication
- **Scalability**: Can scale nodes independently

This architecture enables complex multi-stage model execution with full control over parallelism at each stage.
