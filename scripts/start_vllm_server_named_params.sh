#!/bin/bash

# Default values for parameters
GPU_IDX=0
GPU_MEM_UTILIZATION=0.9
MAX_NUM_SEQS=256
ENABLE_PREFIX_CACHING=false
DISABLE_SLIDING_WINDOW=false
DISABLE_FRONTEND_MULTIPROCESSING=false
MAX_MODEL_LEN=""

# Parse named parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --swap-space) SWAP_SPACE="$2"; shift ;;
        --gpu-idx) GPU_IDX="$2"; shift ;;
        --gpu-memory-utilization) GPU_MEM_UTILIZATION="$2"; shift ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift ;;
        --enable-prefix-caching) ENABLE_PREFIX_CACHING=true ;;
        --disable-sliding-window) DISABLE_SLIDING_WINDOW=true ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift ;;
        --disable-frontend-multiprocessing) DISABLE_FRONTEND_MULTIPROCESSING=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export VLLM_HF_FOLDER_CACHE_FILE=$HF_HOME/vllm_hf_folder_cache.json

CUDA_VISIBLE_DEVICES=$GPU_IDX python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --seed "$SEED" \
    --swap-space "$SWAP_SPACE" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    $(if [ "$ENABLE_PREFIX_CACHING" = true ]; then echo "--enable-prefix-caching"; fi) \
    $(if [ "$DISABLE_SLIDING_WINDOW" = true ]; then echo "--disable-sliding-window"; fi) \
    $(if [ -n "$MAX_MODEL_LEN" ]; then echo "--max-model-len $MAX_MODEL_LEN"; fi) \
    $(if [ "$DISABLE_FRONTEND_MULTIPROCESSING" = true ]; then echo "--disable-frontend-multiprocessing"; fi)
