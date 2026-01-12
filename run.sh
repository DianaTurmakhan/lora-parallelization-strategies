#!/bin/bash

START_TIME=$(date +%s)


python main.py \
  --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --dataset_name "databricks/databricks-dolly-15k" \
  --max_samples 4000 \
  --num_train_epochs 10 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --fp16 \
  --do_eval \
  --gradient_checkpointing \
  --eval_steps 50 \
  --logging_steps 10 \
  --save_steps 100 \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --output_dir "results/baseline" \
  --wandb_project "main" \
  --wandb_name "baseline_single_gpu-8B" \
  --target_loss 0.5 \
  --metrics_log_interval 10 \
  --wandb_entity "ml710_project"\

END_TIME=$(date +%s)

if command -v nvidia-smi &> /dev/null; then
  echo "GPU Memory Usage:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
fi
