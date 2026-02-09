#!/bin/bash
#SBATCH --job-name=cytoatlas-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/vllm_%j.log

module load CUDA/12.8.1 cuDNN

export HF_HOME=/data/parks34/.cache/huggingface

source ~/bin/myconda
conda activate secactpy

python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --tokenizer-mode mistral \
    --enable-prefix-caching
