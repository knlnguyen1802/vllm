#!/usr/bin/env bash
# run_benchmark.sh
#
# Benchmark VisionArena-Chat with vLLM serving.
# All parameters can be overridden via environment variables.

set -euo pipefail

# ── Configurable variables ────────────────────────────────────────────────────
NUM_PROMPTS="${NUM_PROMPTS:-1000}"          # Number of prompts to run
PORT="${PORT:-10001}"                      # Local port where the server is running
LOG_PATH="${LOG_PATH:-./logs}"             # Directory to store logs & JSON results
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_PATH"

python /workspace/n0090/epd/epd-vllm/vllm/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model /workspace/models/qwen2.5_7B_Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path /run/VisionArena-Chat \
  --hf-split train \
  --num-prompts "$NUM_PROMPTS" \
  --seed 40 \
  --save-result \
  --save-detailed \
  --result-dir "$LOG_PATH" \
  --result-filename "vision_arena_outputs_$(date +%Y%m%d_%H%M%S).json" \
  --port "$PORT" \
  2>&1 | tee "$LOG_PATH/benchmark_VisionArena_${NUM_PROMPTS}reqs.log"