import os
import time
import datetime
import torch
import wandb
import json
from trl import SFTTrainer
from src.data_utils import load_and_prepare_dataset
from src.train_utils import (
    get_compute_dtype,
    create_output_dir,
    create_training_config,
    create_lora_config,
    create_metrics_callback,
    generate_summary_metrics
)
from transformers import AutoModelForCausalLM

def verify_tensor_topology(engine):
    topology = engine.mpu.get_tensor_model_parallel_world_size()
    print(f"Tensor parallel size : {topology}")
    print(f"Data parallel size   : {engine.mpu.get_data_parallel_world_size()}")

def train_model_parallel(args):
    torch.manual_seed(args.seed)

    run_name = args.wandb_name or f"tp_lora_r{args.lora_r}_bs{args.per_device_train_batch_size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )

    train_dataset, eval_dataset, tokenizer, max_length = load_and_prepare_dataset(args)

    compute_dtype = get_compute_dtype(args)
    output_dir = create_output_dir(args)

    print(f"Using DeepSpeed config at: {args.deepspeed_config}")

    training_args = create_training_config(args, output_dir)
    peft_config = create_lora_config(args)
    metrics_callback = create_metrics_callback(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=compute_dtype
    )

    print("\nCreating SFTTrainer with DeepSpeed tensor parallelism...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        packing=False,
        peft_config=peft_config,
        max_seq_length=max_length,
        dataset_text_field="messages",
        dataset_kwargs={"add_special_tokens": True, "append_concat_token": False}
    )

    trainer.add_callback(metrics_callback)

    print("\nStarting training using DeepSpeed-based tensor parallelism...")
    train_start = time.time()
    train_result = trainer.train()
    train_end = time.time()

    total_train_time = train_end - train_start
    throughput = len(train_dataset) / total_train_time
    print(f"Training completed in {total_train_time:.2f} sec | Throughput: {throughput:.2f} samples/sec")

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(output_dir)

    summary_metrics = generate_summary_metrics(
        train_result, train_dataset, total_train_time, args, "accelerate+deepspeed_tensor"
    )

    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)

    wandb.finish()
    return summary_metrics