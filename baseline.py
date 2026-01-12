import os
import time
import datetime
import random
import torch
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from multiprocessing import cpu_count
import wandb
import psutil
import gc
import json
import GPUtil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, NVMLError

try:
    nvmlInit()
    nvml_initialized = True
except:
    nvml_initialized = False
    print("NVML initialization failed. Some GPU metrics may not be available.")


class MetricsCallback(TrainerCallback):
    def __init__(self, log_interval=10, target_loss=None):
        self.log_interval = log_interval
        self.target_loss = target_loss
        self.step_times = []
        self.training_start_time = time.time()
        self.last_step_time = time.time()
        self.samples_processed = 0
        self.loss_values = []
        self.reached_target_loss = False
        self.time_to_target = None
        self.total_flops = 0
        self.useful_flops = 0
        self.comm_times = []
        self.compute_times = []
        self.gpu_utils = []
        self.gpu_mems = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_begin_time = time.time()
        self._log_gpu_metrics()

    def on_step_end(self, args, state, control, **kwargs):
        step_end_time = time.time()
        step_time = step_end_time - self.step_begin_time
        self.step_times.append(step_time)
        
        if state.log_history and len(state.log_history) > 0:
            for log in reversed(state.log_history):
                if 'loss' in log:
                    loss = log['loss']
                    self.loss_values.append(loss)
                    
                    if self.target_loss is not None and loss <= self.target_loss and not self.reached_target_loss:
                        self.reached_target_loss = True
                        self.time_to_target = step_end_time - self.training_start_time
                    break
        
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if torch.cuda.device_count() > 0:
            batch_size *= torch.cuda.device_count()
        
        self.samples_processed += batch_size
        
        if torch.cuda.device_count() > 1:
            comm_time = step_time * 0.1
            compute_time = step_time - comm_time
        else:
            comm_time = 0
            compute_time = step_time

        self.comm_times.append(comm_time)
        self.compute_times.append(compute_time)

        if hasattr(self, 'total_model_params'):
            flops_per_step = 2 * self.total_model_params * batch_size * 12
            self.total_flops += flops_per_step

            useful_flops = flops_per_step * (compute_time / step_time)
            self.useful_flops += useful_flops
        
        if state.global_step % self.log_interval == 0:
            self._log_training_metrics(step_time, batch_size, loss if 'loss' in locals() else None)
        
        self.last_step_time = step_end_time
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        self.last_step_time = self.training_start_time

        if 'model' in kwargs:
            model = kwargs['model']
            self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.total_model_params = sum(p.numel() for p in model.parameters())

            param_efficiency = (self.trainable_params / self.total_model_params) * 100 if self.total_model_params > 0 else 0
            wandb.log({
                "model/trainable_parameters": self.trainable_params,
                "model/total_parameters": self.total_model_params,
                "model/parameter_efficiency_pct": param_efficiency
            })

            print(f"Model has {self.trainable_params:,} trainable parameters out of {self.total_model_params:,} total parameters")
            print(f"Parameter efficiency: {param_efficiency:.2f}%")

    def on_train_end(self, args, state, control, **kwargs):
        end_time = time.time()
        total_training_time = end_time - self.training_start_time

        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        throughput = self.samples_processed / total_training_time if total_training_time > 0 else 0

        convergence_rate = None
        if self.target_loss is not None:
            if self.reached_target_loss:
                convergence_rate = self.samples_processed / self.time_to_target
            else:
                print("Warning: Target loss was not reached during training")

        scaling_efficiency = None
        if hasattr(self, 'baseline_throughput') and self.baseline_throughput > 0:
            num_gpus = max(1, torch.cuda.device_count())
            scaling_efficiency = (throughput / self.baseline_throughput) / num_gpus

        comm_overhead = sum(self.comm_times) / sum(self.step_times) * 100 if self.step_times else 0
        goodput = (self.useful_flops / self.total_flops) * 100 if self.total_flops > 0 else 0
        avg_gpu_util = np.mean(self.gpu_utils) if self.gpu_utils else 0
        avg_gpu_mem = np.mean(self.gpu_mems) if self.gpu_mems else 0
        
        final_metrics = {
            "final/total_training_time_seconds": total_training_time,
            "final/total_training_time_formatted": str(datetime.timedelta(seconds=int(total_training_time))),
            "final/avg_step_time_seconds": avg_step_time,
            "final/throughput_samples_per_second": throughput,
            "final/total_samples_processed": self.samples_processed,
            "final/communication_overhead_pct": comm_overhead,
            "final/gpu_utilization_avg_pct": avg_gpu_util,
            "final/gpu_memory_usage_avg_pct": avg_gpu_mem,
            "final/final_loss": self.loss_values[-1] if self.loss_values else None,
            "final/goodput_pct": goodput
        }
        
        if convergence_rate is not None:
            final_metrics["final/convergence_rate"] = convergence_rate
            final_metrics["final/time_to_target_loss"] = self.time_to_target
        
        if scaling_efficiency is not None:
            final_metrics["final/scaling_efficiency"] = scaling_efficiency
        
        wandb.log(final_metrics)
        
        print("\n=== TRAINING METRICS SUMMARY ===")
        for key, value in final_metrics.items():
            if value is not None:
                print(f"{key.split('/')[-1]}: {value}")
        
        output_dir = args.output_dir
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")
    
    def set_baseline_throughput(self, throughput):
        self.baseline_throughput = throughput

    def _log_gpu_metrics(self):
        if not torch.cuda.is_available():
            return
        
        try:
            gpu_util_values = []
            gpu_mem_values = []
            
            for i in range(torch.cuda.device_count()):
                if nvml_initialized:
                    try:
                        handle = nvmlDeviceGetHandleByIndex(i)
                        util = nvmlDeviceGetUtilizationRates(handle)
                        gpu_util_values.append(util.gpu)
                        
                        mem_info = nvmlDeviceGetMemoryInfo(handle)
                        mem_pct = (mem_info.used / mem_info.total) * 100
                        gpu_mem_values.append(mem_pct)
                    except NVMLError:
                        gpus = GPUtil.getGPUs()
                        if i < len(gpus):
                            gpu_util_values.append(gpus[i].load * 100)
                            gpu_mem_values.append(gpus[i].memoryUtil * 100)
                else:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_util_values.append(gpus[i].load * 100)
                        gpu_mem_values.append(gpus[i].memoryUtil * 100)
            
            self.gpu_utils.append(np.mean(gpu_util_values) if gpu_util_values else 0)
            self.gpu_mems.append(np.mean(gpu_mem_values) if gpu_mem_values else 0)
            
        except Exception as e:
            print(f"Error logging GPU metrics: {e}")
    
    def _log_training_metrics(self, step_time, batch_size, loss):
        elapsed_time = time.time() - self.training_start_time
        
        metrics = {
            "training/step_time_seconds": step_time,
            "training/elapsed_time_seconds": elapsed_time,
            "training/elapsed_time_formatted": str(datetime.timedelta(seconds=int(elapsed_time))),
            "training/samples_processed": self.samples_processed,
            "training/throughput_samples_per_second": batch_size / step_time,
        }
        
        if loss is not None:
            metrics["training/loss"] = loss

        if torch.cuda.is_available():
            metrics["hardware/gpu_utilization_pct"] = self.gpu_utils[-1] if self.gpu_utils else 0
            metrics["hardware/gpu_memory_used_pct"] = self.gpu_mems[-1] if self.gpu_mems else 0
            metrics["hardware/gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated()
            metrics["hardware/gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved()

        metrics["hardware/cpu_utilization_pct"] = psutil.cpu_percent()
        metrics["hardware/ram_used_pct"] = psutil.virtual_memory().percent
        metrics["hardware/communication_time_seconds"] = self.comm_times[-1] if self.comm_times else 0
        metrics["hardware/computation_time_seconds"] = self.compute_times[-1] if self.compute_times else 0
        metrics["hardware/communication_overhead_pct"] = (self.comm_times[-1] / step_time) * 100 if step_time > 0 else 0
        
        wandb.log(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline training script for Llama model with LoRA")

    parser.add_argument('--output_dir', type=str, default='results/baseline',
                        help='Path to save the model and checkpoints.')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to use from dataset.')                    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--eval_ratio', type=float, default=0.2,
                        help='Ratio of data to use for evaluation.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='Batch size per GPU for training.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                        help='Batch size per GPU for evaluation.')
    parser.add_argument('--eval_steps', type=int, default=50,
                        help='Perform evaluation every X steps.')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log metrics every X steps.')
    parser.add_argument('--save_strategy', type=str, default="steps",
                        help='When to save the model checkpoint (steps/epoch).')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint every X steps.')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use fp16 (mixed precision) during training.')
    parser.add_argument('--bf16', action='store_true',
                        help='Whether to use bf16 during training (better for A100/H100).')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model during training.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of update steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization.')
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help='Model ID for loading the pre-trained Llama model.')
    parser.add_argument('--dataset_name', type=str, default="databricks/databricks-dolly-15k",
                        help='Name of the HuggingFace dataset to use.')
    parser.add_argument('--sequence_length', type=int, default=1024,
                        help='Maximum sequence length to use for training.')
    
    parser.add_argument('--lora_r', type=int, default=64,
                        help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    
    # Metrics tracking args
    parser.add_argument('--target_loss', type=float, default=None,
                        help='Target loss value for time-to-accuracy measurement')
    parser.add_argument('--baseline_throughput', type=float, default=None,
                        help='Baseline throughput for scaling efficiency calculation')
    parser.add_argument('--metrics_log_interval', type=int, default=10,
                        help='How often to log detailed metrics (steps)')
    
    # Wandb configuration
    parser.add_argument('--wandb_project', type=str, default="lora-parallelization",
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity name')

    args = parser.parse_args()
    return args


def format_dolly_dataset(example):
    system_message = "You are a helpful AI assistant that provides detailed and informative responses."
    
    user_message = example["instruction"]
    if example.get("context") and example["context"].strip():
        user_message += f"\n\nContext: {example['context']}"
    
    assistant_message = example["response"]
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    
    return {
        "messages": messages,
        "system": system_message
    }


def main():
    args = parse_args()
    
    run_name = args.wandb_name
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lora_r{args.lora_r}_bs{args.per_device_train_batch_size}_{timestamp}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args)
    )

    print(f"Running with arguments: {args}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    if args.max_samples and args.max_samples < len(dataset["train"]):
        print(f"Using {args.max_samples} samples from the dataset")
        dataset["train"] = dataset["train"].select(range(args.max_samples))
    
    print("Formatting dataset to chat format")
    formatted_dataset = dataset["train"].map(
        format_dolly_dataset,
        remove_columns=dataset["train"].column_names,
        desc="Formatting dataset",
    )
    
    print(f"\nSample from the formatted dataset:")
    sample_idx = random.randint(0, len(formatted_dataset) - 1)
    print(f"Sample {sample_idx}:")
    for message in formatted_dataset[sample_idx]["messages"]:
        print(f"{message['role'].upper()}: {message['content'][:100]}...")
    
    print(f"\nLoading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token_id is None:
        print("Setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    
    max_length = min(args.sequence_length, getattr(tokenizer, "model_max_length", 2048))
    print(f"Using maximum sequence length: {max_length}")
    
    print(f"Splitting dataset with eval ratio: {args.eval_ratio}")
    train_val_split = formatted_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

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

    model_kwargs = dict(
        torch_dtype=compute_dtype,
        use_cache=False,
        device_map="auto",
    )

    model_name = args.model_id.split("/")[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{model_name}_lora_r{args.lora_r}_ep{args.num_train_epochs}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}_{timestamp}"
    print(f"Output directory: {output_dir}")

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
    print(f"Training arguments configured")
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    print(f"LoRA config: rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
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
    
    metrics_callback = MetricsCallback(
        log_interval=args.metrics_log_interval,
        target_loss=args.target_loss
    )
    
    if args.baseline_throughput is not None:
        metrics_callback.set_baseline_throughput(args.baseline_throughput)
    
    print("\nCreating SFT Trainer...")
    trainer = SFTTrainer(
        model=args.model_id,
        model_init_kwargs=model_kwargs,
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

    # Train the model
    print("\nStarting training...")
    train_start = time.time()
    train_result = trainer.train()
    train_end = time.time()
    
    total_train_time = train_end - train_start
    print(f"\nTotal training time: {total_train_time:.2f} seconds")
    wandb.log({"final/total_wall_time": total_train_time})
    
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
    }
    
    # Save summary metrics to file
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)
    
    print("\nFinal training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nBaseline training complete!")
    
    wandb.finish()


if __name__ == "__main__":
    main()