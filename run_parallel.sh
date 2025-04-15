#!/bin/bash
set -ex

# ------------ Editable Variables ------------

workdir=/home/ainur.khamitova/Desktop/ML710
master_node=ws-l4-018
worker_node=ws-l4-020

TP=1
PP=2
GLOBAL_BATCH=32
MICRO_BATCH=2
ZERO_STAGE=0

# ------------ Paths ------------
BASE_PATH=$workdir/lora-parallelization-strategies
DS_CONFIG=$BASE_PATH/configs/ds_config_pp.json
HOST_FILE=$BASE_PATH/hostfile
OUTPUT_DIR=$BASE_PATH/output/lora_pipeline_run

export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p $OUTPUT_DIR

# ------------ Host File ------------
cat <<EOT > $HOST_FILE
${master_node} slots=1
${worker_node} slots=1
EOT

# ------------ DeepSpeed Config ------------
cat <<EOT > $DS_CONFIG
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true
  },
  "pipeline": {
    "enabled": true,
    "stages": $PP
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  },
  "wall_clock_breakdown": true
}
EOT


# ------------ DeepSpeed Args ------------
ds_args="--deepspeed --deepspeed_config $DS_CONFIG"

# ------------ Launch Script ------------

COMMON_ARGS=(
  --model_id meta-llama/Llama-3.2-1B-Instruct
  --dataset_name databricks/databricks-dolly-15k
  --max_samples 4000
  --num_train_epochs 10
  --learning_rate 2e-5
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 4
  --fp16
  --do_eval
  --gradient_checkpointing
  --eval_steps 50
  --logging_steps 10
  --save_steps 100
  --lora_r 64
  --lora_alpha 16
  --lora_dropout 0.1
  --output_dir results/baseline
  --wandb_project main
  --wandb_name baseline_single_gpu
  --target_loss 0.5
  --metrics_log_interval 10
  --wandb_entity ml710_project
  --parallelization_strategy pipeline
)

# ------------ Determine Node Rank ------------
if [ "$(hostname)" == "$master_node" ]; then
  NODE_RANK=0
else
  NODE_RANK=1
fi

# ------------ Run DeepSpeed ------------
deepspeed \
  --num_gpus 1 \
  --num_nodes 2 \
  --hostfile $HOST_FILE \
  --no_ssh \
  --master_addr $master_node \
  --master_port 29500 \
  --node_rank $NODE_RANK \
  main.py \
  "${COMMON_ARGS[@]}" \
  $ds_args | tee $OUTPUT_DIR/training_node${NODE_RANK}.log
