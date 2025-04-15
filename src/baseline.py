# src/trainers/base_trainer.py

import os
import time
import datetime
import torch
import wandb
import json
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from src.metrics import MetricsCallback
from src.data_utils import load_and_prepare_dataset
from transformers import AutoModelForCausalLM


def train_single_gpu(args):
    """Baseline single-GPU training implementation for LoRA fine-tuning."""
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Generate run name for wandb
    run_name = args.wandb_name
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"baseline_lora_r{args.lora_r}_bs{args.per_device_train_batch_size}_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )
    
    # Prepare dataset
    train_dataset, eval_dataset, tokenizer, max_length = load_and_prepare_dataset(args)
    
    # Print GPU info
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\nNo GPU detected, training on CPU")
    
    # Set compute dtype
    compute_dtype = None
    if args.bf16:
        compute_dtype = torch.bfloat16
        print("Using bfloat16 precision")
    elif args.fp16:
        compute_dtype = torch.float16
        print("Using float16 precision")
    else:
        compute_dtype = torch.float32
        print("Using float32 precision")
    
    # Define model kwargs
    model_kwargs = dict(
        torch_dtype=compute_dtype,
        use_cache=False,
        device_map="auto",
    )
    
    # Create output directory
    model_name = args.model_id.split("/")[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{model_name}_lora_r{args.lora_r}_ep{args.num_train_epochs}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}_{timestamp}"
    print(f"Output directory: {output_dir}")
    
    # Define training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        report_to="wandb",
        eval_steps=args.eval_steps if args.do_eval else None,
        dataset_text_field="text",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        do_eval=args.do_eval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_level="info",
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        max_steps=-1,
        seed=args.seed,
        overwrite_output_dir=True,
    )
    
    # Define LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Create metrics callback
    metrics_callback = MetricsCallback(
        log_interval=args.metrics_log_interval,
        target_loss=args.target_loss
    )
    
    # Set baseline throughput if provided
    if args.baseline_throughput is not None:
        metrics_callback.set_baseline_throughput(args.baseline_throughput)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=compute_dtype,
        use_cache=False,
        # device_map="auto",
        # low_cpu_mem_usage=True,  # Add this parameter
    )
    
    print("\nCreating SFT Trainer...")
    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        # model_init_kwargs=model_kwargs,
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
    
    # Add metrics callback
    trainer.add_callback(metrics_callback)
    
    # Train the model
    print("\nStarting training...")
    train_start = time.time()
    train_result = trainer.train()
    train_end = time.time()
    
    # Calculate training time
    total_train_time = train_end - train_start
    print(f"\nTotal training time: {total_train_time:.2f} seconds")

    # Log metrics
    print("\nTraining completed.")
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Calculate throughput
    throughput = len(train_dataset) / total_train_time
    print(f"Training throughput: {throughput:.2f} samples/second")
    
    # Save the model
    print(f"Saving model to {output_dir}")
    trainer.save_state()
    trainer.save_model(output_dir)
    
    # Create a summary of key metrics
    summary_metrics = {
        "wall_time": total_train_time,
        "training_throughput": throughput,
        "final_loss": metrics.get("train_loss", None),
        "samples_processed": len(train_dataset),
        "batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, torch.cuda.device_count()),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "lora_rank": args.lora_r,
        "model": args.model_id,
        "parallelization": "single_gpu"
    }
    
    # Save summary metrics to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)
    
    # Finish wandb run
    wandb.finish()
    
    return summary_metrics