#!/bin/bash
#SBATCH --job-name=lora_train_job
#SBATCH --output=logs/%x_%j.out     # %x = job-name, %j = job-id
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00  # 7 days (D-HH:MM:SS)
#SBATCH --partition=ws-ia       # Partition to use
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --mem=40G         # Request 64 GB total memory
#SBATCH --nodelist=ws-l4-009

# Optional: Load your environment
# module load cuda/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
source /home/ainur.khamitova/miniconda3/bin/activate ml710

# Optionally record the start time
START_TIME=$(date +%s)

# Run your Python script
#   --model_id "meta-llama/Llama-3.2-1B-Instruct" \
python main.py \
  --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --dataset_name "databricks/databricks-dolly-15k" \
  --max_samples 4000 \
  --num_train_epochs 10 \
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
  --wandb_project "main" \
  --wandb_name "baseline_single_gpu" \
  --target_loss 0.5 \
  --metrics_log_interval 10 \
  --wandb_entity "ml710_project"\


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

# GPU memory usage
if command -v nvidia-smi &> /dev/null; then
  echo "GPU Memory Usage:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
fi
