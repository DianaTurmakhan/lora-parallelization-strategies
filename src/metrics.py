import time
import datetime
import numpy as np
import torch
import wandb
import psutil
import os
import json
from transformers import TrainerCallback
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, NVMLError
import GPUtil

# Initialize NVML for GPU metrics
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
        self.compute_times = []
        # Initialize GPU tracking
        self.gpu_utils = []
        self.gpu_mems = []

        self.step_table = wandb.Table(columns=["Step", "Elapsed Time"])
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Record time at the beginning of a step"""
        self.step_begin_time = time.time()
        # Log GPU utilization at step begin
        self._log_gpu_metrics()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Record metrics at the end of a step"""
        step_end_time = time.time()
        step_time = step_end_time - self.step_begin_time
        self.step_times.append(step_time)
     

        if state.log_history and len(state.log_history) > 0:
            for log in reversed(state.log_history):
                if 'loss' in log:
                    loss = log['loss']
                    self.loss_values.append(loss)
                break
                    
                  
        # Calculate batch size
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if torch.cuda.device_count() > 0:
            batch_size *= torch.cuda.device_count()
        
        self.samples_processed += batch_size
        # Log GPU utilization at step end
        
        self.compute_times.append(step_time)

        if state.global_step % self.log_interval == 0:
            self._log_training_metrics(self.step_table, step_time, state.global_step, batch_size, loss if 'loss' in locals() else None)
        
        self.last_step_time = step_end_time

    def on_train_begin(self, args, state, control, **kwargs):
        """Record the start of training and set up initial metrics"""
        self.training_start_time = time.time()
        self.last_step_time = self.training_start_time

        # Record model parameter counts if available
        if 'model' in kwargs:
            model = kwargs['model']
            self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.total_model_params = sum(p.numel() for p in model.parameters())
            
            # Log parameter efficiency
            param_efficiency = (self.trainable_params / self.total_model_params) * 100 if self.total_model_params > 0 else 0
          
            model_param_table = wandb.Table(columns=["Metric", "Value"])
            model_param_table.add_data("model/trainable_parameters", self.trainable_params)
            model_param_table.add_data("model/total_parameters", self.total_model_params)
            model_param_table.add_data("model/parameter_efficiency_pct", param_efficiency)

            wandb.log({"model_parameter_table": model_param_table})
            
            print(f"Model has {self.trainable_params:,} trainable parameters out of {self.total_model_params:,} total parameters")
            print(f"Parameter efficiency: {param_efficiency:.2f}%")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Record final metrics at the end of training"""
        end_time = time.time()
        total_training_time = end_time - self.training_start_time
        
        # Calculate final metrics
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        throughput = self.samples_processed / total_training_time if total_training_time > 0 else 0
    
        # Calculate GPU utilization stats
        avg_gpu_util = np.mean(self.gpu_utils) if self.gpu_utils else 0
        avg_gpu_mem = np.mean(self.gpu_mems) if self.gpu_mems else 0
        
        # Log final metrics to wandb
        final_metrics = {
            "total_training_time_seconds": total_training_time,
            "total_training_time_formatted": str(datetime.timedelta(seconds=int(total_training_time))),
            "avg_step_time_seconds": avg_step_time,
            "total_samples_processed": self.samples_processed,
            "avg_throughput": throughput,
            "avg_gpu_utilization": avg_gpu_util,
            "avg_gpu_memory_usage": avg_gpu_mem,
            "final_loss": self.loss_values[-1] if self.loss_values else None,
        }

        # wandb.log(final_metrics)
        final_metrics_table = wandb.Table(columns=["Metric", "Value"])
        for key, value in final_metrics.items():
            final_metrics_table.add_data(key, str(value))

        wandb.log({"final_metrics_table": final_metrics_table})
        wandb.log({"training_elapsed_time_table": self.step_table})
       
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
    
  
    def _log_gpu_metrics(self):
        """Log GPU utilization and memory usage"""
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
                    # Fall back to GPUtil if NVML is not initialized
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_util_values.append(gpus[i].load * 100)
                        gpu_mem_values.append(gpus[i].memoryUtil * 100)
            
            # Store values for averaging later
            self.gpu_utils.append(np.mean(gpu_util_values) if gpu_util_values else 0)
            self.gpu_mems.append(np.mean(gpu_mem_values) if gpu_mem_values else 0)
            
        except Exception as e:
            print(f"Error logging GPU metrics: {e}")
    
    def _log_training_metrics(self, elapsed_time_table, step_time, step,  batch_size, loss):
        """Log training metrics to wandb"""
        elapsed_time = time.time() - self.training_start_time
        

        # TROUGHPUT
        throughput = batch_size / step_time if step_time > 0 else 0

        # STAT EFFICIENCY
        se = (np.abs(self.loss_values[-1] - self.loss_values[-2])) / batch_size if len(self.loss_values) > 1 else 0

        # GOODPUT
        goodput = throughput * se

        # Log GPU utilization at step end
        self.compute_times.append(step_time)
        
        
        metrics = {
            "training/step_time_seconds": step_time,
            "training/elapsed_time_seconds": elapsed_time,
            "training/samples_processed": self.samples_processed,
            "training/throughput": throughput,
            "training/goodput": goodput,
            "training/stat_efficiency": se,
        }

    
        elapsed_time_table.add_data(step, str(datetime.timedelta(seconds=int(elapsed_time))))
                
        if loss is not None:
            metrics["training/loss"] = loss
        
        # Add GPU metrics
        if torch.cuda.is_available():
            metrics["hardware/gpu_utilization_pct"] = self.gpu_utils[-1] if self.gpu_utils else 0
            metrics["hardware/gpu_memory_used_pct"] = self.gpu_mems[-1] if self.gpu_mems else 0
            metrics["hardware/gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated()
            metrics["hardware/gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved()
        
        # Add CPU metrics
        metrics["hardware/cpu_utilization_pct"] = psutil.cpu_percent()
        metrics["hardware/ram_used_pct"] = psutil.virtual_memory().percent
        
        # Communication vs computation
        metrics["hardware/computation_time_seconds"] = self.compute_times[-1] if self.compute_times else 0
      
        wandb.log(metrics)