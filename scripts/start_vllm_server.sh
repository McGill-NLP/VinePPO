#!/bin/bash

MODEL=$1
PORT=$2
SEED=$3
SWAP_SPACE=$4

# Read GPU IDX to use. Default is 0
GPU_IDX=${5:-0}

export VLLM_HF_FOLDER_CACHE_FILE=$HF_HOME/vllm_hf_folder_cache.json

CUDA_VISIBLE_DEVICES=$GPU_IDX python -m vllm.entrypoints.openai.api_server \
	--model "$MODEL" \
	--host 0.0.0.0 \
	--port "$PORT" \
	--seed "$SEED" \
	--swap-space "$SWAP_SPACE" \
	--dtype bfloat16