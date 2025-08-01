#!/bin/bash

declare -A models
models=(
  ["glm4-9b"]=4096
  ["llama3-8b"]=4096
  ["llama2-7b"]=2048
  ["mistral-7b"]=2048
  ["vicuna-7b"]=2048
  ["phi4-mini"]=1024
  ["qwen2.5-3b"]=1024
  ["stablelm-3b"]=1024
  ["llama3-3b"]=1024
  ["gemma2-2b"]=1024
  ["llama3-1b"]=1024
)

for model in "${!models[@]}"; do
  max_length=${models[$model]}
  echo "Running: model=$model, max_length=$max_length"
  # LoopLLM-t
  CUDA_VISIBLE_DEVICES=0 python main.py --model_name "$model" --max_length "$max_length" --c 1 --root_dir 'res/experiment/LoopLLM_t'
  
  # LoopLLM-p
  CUDA_VISIBLE_DEVICES=0 python main.py --model_name "$model" --max_length "$max_length" --c 5 --root_dir 'res/experiment/LoopLLM_p'

  # baselines
  CUDA_VISIBLE_DEVICES=0 python baseline/LLMEffiChecker.py --model_name "$model" --max_length "$max_length"
  CUDA_VISIBLE_DEVICES=0 python baseline/Engorgio.py --model_name "$model" #--max_length "$max_length"
  CUDA_VISIBLE_DEVICES=0 python baseline/normal.py --model_name "$model" --max_length "$max_length"
  echo -e "-----------Finish-----------"
done