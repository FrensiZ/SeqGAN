import os
import torch as th
import subprocess
from pathlib import Path
import json
import itertools
import time
import hashlib
import torch

N_SEEDS = 3  # Number of random seeds to test each configuration

def get_config_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

def get_free_gpu():
    try:
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            return None
        
        # Get total number of GPUs
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            # Get memory usage
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            
            # Check GPU utilization
            gpu_util = torch.cuda.utilization(i)
            
            # Criteria for a "free" GPU
            if allocated_memory / total_memory < 0.1 and gpu_util < 5:
                return i
        
        return None
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return None

def generate_configs():
    # Grid search for discriminator parameters
    param_grid = {
        'disc_type': ['simple', 'lstm', 'cnn'],     # Discriminator architecture types
        'batch_size': [64, 128],                    # Batch sizes for training
        'learning_rate': [1e-4, 5e-4],              # Learning rates for optimizer
        'embedding_dim': [64, 128],                 # Embedding dimensions
        'hidden_dim': [128, 256],                   # Hidden dimensions
        'dropout_rate': [0.1, 0.3],                 # Dropout rates
        'outer_epochs': [10, 25],                   # Number of outer training epochs
        'inner_epochs': [2, 3]                      # Number of inner training epochs
    }

    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def run_training(config, gpu_id, seed, config_id):

    base_dir = Path(os.getenv('WORKING_DIR', '.'))
    output_dir = base_dir / f"disc_outputs/config_{config_id}/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    os.chmod(output_dir, 0o775)
    
    # Save the configuration JSON
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),  # Keep this
        "CUDA_DEVICE": str(gpu_id),  # Add this line
        "GEN_PATH": str(base_dir / "saved_models/generator_pretrained.pth"),
        "OUTPUT_DIR": str(output_dir),
        "CONFIG_PATH": str(output_dir / "config.json"),
        "SEED": str(seed)
    })
    
    process = subprocess.Popen(
        ["python3", str(base_dir / "test_discriminator.py")],
        env=env,
        cwd=str(output_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process, output_dir

def main():
    base_dir = Path(os.getenv('WORKING_DIR', '.'))
    os.makedirs(base_dir / "disc_outputs", exist_ok=True)

    # Check if pretrained generator exists
    gen_path = base_dir / "saved_models/generator_pretrained.pth"
    if not os.path.exists(gen_path):
        print(f"Pretrained generator not found at {gen_path}. Aborting.")
        return

    configs = generate_configs()
    active_processes = {}
    completed = set()
    gpu_in_use = set()
    all_processes_started = False
    
    print(f"Starting discriminator hyperparameter search with {len(configs)} configurations")
    print(f"Each configuration will be tested with {N_SEEDS} seeds")
    print(f"Total runs: {len(configs) * N_SEEDS}")
    
    while len(completed) < len(configs) * N_SEEDS:
        if not all_processes_started:
            for config_id, config in enumerate(configs):
                for seed in range(N_SEEDS):
                    if (config_id, seed) not in completed and (config_id, seed) not in active_processes:
                        gpu_id = get_free_gpu()
                        if gpu_id is not None and gpu_id not in gpu_in_use:
                            try:
                                process, output_dir = run_training(config, gpu_id, seed, config_id)
                                active_processes[(config_id, seed)] = {
                                    'process': process,
                                    'gpu': gpu_id,
                                    'output_dir': output_dir,
                                    'config': config
                                }
                                gpu_in_use.add(gpu_id)
                                print(f"Started config {config_id} seed {seed} on GPU {gpu_id}")
                            except Exception as e:
                                print(f"Failed to start config {config_id} seed {seed}: {e}")
                                continue

            if len(active_processes) + len(completed) == len(configs) * N_SEEDS:
                all_processes_started = True
                print("\nAll processes started, waiting for completion...")

        for key in list(active_processes.keys()):
            process_info = active_processes[key]
            if process_info['process'].poll() is not None:
                stdout, stderr = process_info['process'].communicate()
                with open(process_info['output_dir'] / "logs.txt", 'w') as f:
                    f.write(stdout + "\nSTDERR:\n" + stderr)
                
                if process_info['process'].returncode == 0:
                    completed.add(key)
                    print(f"Completed config {key[0]} seed {key[1]}")
                else:
                    print(f"Failed config {key[0]} seed {key[1]}")
                
                gpu_in_use.remove(process_info['gpu'])
                del active_processes[key]
        
        time.sleep(4)
    
    print("\nAll processes completed!")

if __name__ == "__main__":
    main()