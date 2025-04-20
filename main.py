import os
import argparse
from src.baseline import train_single_gpu
from src.pipeline import train_pipeline_parallel
from src.hybrid import train_hybrid_parallel
from src.model import train_model_parallel  

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA parallelization strategies for LLM fine-tuning")
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
    
    # Parallelization strategy
    parser.add_argument('--parallelization_strategy', type=str, default="single_gpu",
                        choices=["single_gpu", "data_parallel", "ddp", "fsdp", "deepspeed", "pipeline", "hybrid", "model"],
                        help='Parallelization strategy to use')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str)
    
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

def main():
    args = parse_args()
    
    # Select the appropriate training function based on the parallelization strategy
    if args.parallelization_strategy == "single_gpu":
        metrics = train_single_gpu(args)
        print(f"\nBaseline training complete!")
    elif args.parallelization_strategy == "data_parallel":
        # Uncomment when implemented
        # metrics = train_data_parallel(args)
        print(f"\nData Parallel training complete!")
    elif args.parallelization_strategy == "ddp":
        # Uncomment when implemented
        # metrics = train_ddp(args)
        print(f"\nDDP training complete!")
    elif args.parallelization_strategy == "fsdp":
        # Uncomment when implemented
        # metrics = train_fsdp(args)
        print(f"\nFSDP training complete!")
    elif args.parallelization_strategy == "deepspeed":
        # Uncomment when implemented
        # metrics = train_deepspeed(args)
        print(f"\nDeepSpeed training complete!")
    elif args.parallelization_strategy == "pipeline":
        metrics = train_pipeline_parallel(args)
        print(f"\nPipeline Parallel training complete!")
    elif args.parallelization_strategy == "hybrid":
        metrics = train_hybrid_parallel(args)
        print(f"\nHybrid Parallel training complete!")
    elif args.parallelization_strategy == "model":
        metrics = train_model_parallel(args)
        print(f"\nModel Parallel training complete!")
    else:
        raise ValueError(f"Unknown parallelization strategy: {args.parallelization_strategy}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"Summary for strategy: {args.parallelization_strategy}")
    print("="*50)
    print(f"Training throughput: {metrics.get('training_throughput', 'N/A'):.2f} samples/second")
    print(f"Total wall time: {metrics.get('wall_time', 'N/A'):.2f} seconds")
    print(f"Final loss: {metrics.get('final_loss', 'N/A')}")
    print(f"GPU count: {metrics.get('num_gpus', 'N/A')}")
    print(f"Effective batch size: {metrics.get('effective_batch_size', 'N/A')}")
    if 'scaling_efficiency' in metrics:
        print(f"Scaling efficiency: {metrics['scaling_efficiency']:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()