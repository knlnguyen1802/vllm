#!/usr/bin/env bash
# ------------------------------------------------------------------
# Run vLLM benchmarks for different NUM_PROMPTS and TP sizes
# ------------------------------------------------------------------
set -euo pipefail

REST_PORT=10001          # where /v1/chat/completions is exposed
SERVER_TIMEOUT=600       # seconds to wait until the server is ready
NUM_PROMPTS_LIST=(100 200 500 1000)

# PID of the currently-running vLLM server (if any)
SERVER_PID=""

cleanup () {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "🧹  Cleaning up server PID ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    timeout 10 tail --pid="${SERVER_PID}" -f /dev/null 2>/dev/null || kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server () {
  local port=$1
  echo -n "⏳  Waiting for server on port ${port} … "
  if timeout "${SERVER_TIMEOUT}" bash -c \
      "until curl -sf localhost:${port}/metrics >/dev/null; do sleep 1; done"
  then
    echo "✅  up."
  else
    echo "❌  timed out."
    return 1
  fi
}

for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do
    pkill -9 python && kill -9 $(ps aux|grep python| awk '{ print $2 }')
    redis-cli FLUSHALL
    export NUM_PROMPTS

    # ---------------------------------------------------------------
    # Start vLLM
    # ---------------------------------------------------------------
    echo "🚀  Starting vLLM …"
    bash ./launch_epd_serve.sh &
    SERVER_PID=$!
    echo "    vLLM PID: ${SERVER_PID}"

    wait_for_server "${REST_PORT}" || { cleanup; continue; }

    # ---------------------------------------------------------------
    # Run benchmark in background, wait for it to *really* finish
    # ---------------------------------------------------------------
    echo "🏃  Running benchmark …"
    bash ./benchmark.sh &
    BENCH_PID=$!
    wait "${BENCH_PID}"
    BENCH_RC=$?

    if [[ ${BENCH_RC} -eq 0 ]]; then
        echo "🎉  Benchmark finished successfully."
    else
        echo "⚠️  Benchmark exited with code ${BENCH_RC}"
    fi

    # ---------------------------------------------------------------
    # Shut down only the vLLM server we started
    # ---------------------------------------------------------------
    echo "🛑  Stopping vLLM PID ${SERVER_PID} …"
    cleanup          # will kill SERVER_PID
    SERVER_PID=""    # so the trap won’t run twice
  done
done

echo
echo "✅  All experiments completed."