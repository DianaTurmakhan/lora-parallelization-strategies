#!/bin/bash

# This script runs your main.py training code and collects timing metrics.

# Optionally record the start time
START_TIME=$(date +%s)

# Call your Python script with the desired arguments.
# Make sure to point it to the correct file containing your `main()` function.
python main.py \
  --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --dataset_name "databricks/databricks-dolly-15k" \
  --max_samples 500 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
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
  --wandb_project "lora-parallelization" \
  --wandb_name "baseline_single_gpu" \
  --target_loss 0.5 \
  --metrics_log_interval 10

# Optionally record the end time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================"
echo ""

# (Optional) Print GPU memory usage if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
  echo "GPU Memory Usage:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
fi
