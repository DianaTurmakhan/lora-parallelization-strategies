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

def verify_tensor_topology(engine):
    topology = engine.mpu.get_tensor_model_parallel_world_size()
    print(f"Tensor parallel size : {topology}")
    print(f"Data parallel size   : {engine.mpu.get_data_parallel_world_size()}")

def train_model_parallel(args):
    """Tensor parallel LoRA fine-tuning using Accelerate + DeepSpeed."""

    torch.manual_seed(args.seed)

    run_name = args.wandb_name or f"tp_lora_r{args.lora_r}_bs{args.per_device_train_batch_size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )

    train_dataset, eval_dataset, tokenizer, max_length = load_and_prepare_dataset(args)

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=compute_dtype
    )

    # === Output directory ===
    model_name = args.model_id.split("/")[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{model_name}_dstp_lora_r{args.lora_r}_ep{args.num_train_epochs}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # === DeepSpeed config path ===
    ds_config_path = args.deepspeed_config  # Must point to a JSON file with tensor parallelism setup
    print(f"Using DeepSpeed config at: {ds_config_path}")

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

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    metrics_callback = MetricsCallback(
        log_interval=args.metrics_log_interval,
        target_loss=args.target_loss
    )

    if args.baseline_throughput:
        metrics_callback.set_baseline_throughput(args.baseline_throughput)

    # NOTE: Do not call accelerator.prepare() directly if using `SFTTrainer`
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
        "parallelization": "accelerate+deepspeed_tensor"
    }

    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)

    wandb.finish()
    return summary_metrics