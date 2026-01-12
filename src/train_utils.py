import os
import datetime
import torch
from peft import LoraConfig
from trl import SFTConfig
from src.metrics import MetricsCallback


def get_compute_dtype(args):
    """Determine compute dtype from args."""
    if args.bf16:
        return torch.bfloat16
    elif args.fp16:
        return torch.float16
    else:
        return torch.float32


def create_output_dir(args):
    """Create timestamped output directory."""
    model_name = args.model_id.split("/")[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_prefix = {
        "single_gpu": "lora",
        "pipeline": "dspp_lora",
        "model": "dstp_lora",
        "hybrid": "dshybrid_lora"
    }.get(args.parallelization_strategy, "lora")

    output_dir = f"{args.output_dir}/{model_name}_{strategy_prefix}_r{args.lora_r}_ep{args.num_train_epochs}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_training_config(args, output_dir):
    """Create SFTConfig with common training arguments."""
    return SFTConfig(
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


def create_lora_config(args):
    """Create LoRA configuration."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def create_metrics_callback(args):
    """Create and configure metrics callback."""
    metrics_callback = MetricsCallback(
        log_interval=args.metrics_log_interval,
        target_loss=args.target_loss
    )

    if args.baseline_throughput:
        metrics_callback.set_baseline_throughput(args.baseline_throughput)

    return metrics_callback


def generate_summary_metrics(train_result, train_dataset, total_train_time, args, parallelization_type):
    """Generate summary metrics dictionary."""
    throughput = len(train_dataset) / total_train_time
    metrics = train_result.metrics

    return {
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
        "parallelization": parallelization_type
    }
