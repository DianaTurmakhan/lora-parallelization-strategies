
import os
import time
import datetime
import random
import torch
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback



def format_dolly_dataset(example):
    """Format Databricks Dolly dataset into a chat format."""
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


def load_and_prepare_dataset(args):
    """Load and prepare dataset for training."""
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
    
    # Print a sample for verification
    print(f"\nSample from the formatted dataset:")
    sample_idx = random.randint(0, len(formatted_dataset) - 1)
    print(f"Sample {sample_idx}:")
    for message in formatted_dataset[sample_idx]["messages"]:
        print(f"{message['role'].upper()}: {message['content'][:100]}...")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Configure tokenizer
    if tokenizer.pad_token_id is None:
        print("Setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    
    # Set sequence length
    max_length = min(args.sequence_length, getattr(tokenizer, "model_max_length", 2048))
    print(f"Using maximum sequence length: {max_length}")
    
    # Split the dataset
    print(f"Splitting dataset with eval ratio: {args.eval_ratio}")
    train_val_split = formatted_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset, tokenizer, max_length