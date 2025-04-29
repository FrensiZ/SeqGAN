import os
import torch as th
import random
import numpy as np
import subprocess
from pathlib import Path
import json
import itertools
import time
import hashlib

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = BASE_DIR / "saved_models"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============= FIXED PARAMETERS =============
# Data parameters
VOCAB_SIZE = 5000
SEQ_LENGTH = 20
START_TOKEN = 0
GENERATED_NUM = 10000  # Number of samples to generate for testing

# Oracle/Generator model parameters
# These should be fixed as your generator is already pretrained
ORACLE_EMB_DIM = 32
ORACLE_HIDDEN_DIM = 32

# Paths for models
GEN_PRETRAIN_PATH = SAVE_DIR / 'generator_pretrained.pth'
TARGET_PARAMS_PATH = SAVE_DIR / 'target_params.pkl'

# ============= DEFAULT DISCRIMINATOR PARAMETERS =============
# These are the default parameters that can be overridden in experiments
DEFAULT_DISCRIMINATOR_CONFIG = {
    'disc_type': 'lstm',           # Options: 'simple', 'lstm', 'cnn'
    'embedding_dim': 64,           # Embedding dimension
    'hidden_dim': 128,             # Hidden state dimension
    'dropout_rate': 0.2,           # Dropout rate for regularization
    'batch_size': 64,              # Training batch size
    'learning_rate': 1e-4,         # Learning rate for optimizer
    'outer_epochs': 10,            # Number of outer training loops
    'inner_epochs': 3,             # Number of inner training epochs per outer loop
}

# ============= PARALLEL TRAINING PARAMETERS =============
# Settings for hyperparameter search
PARALLEL_CONFIG = {
    'num_seeds': 3,                # Number of random seeds to test each configuration
    'param_grid': {
        'disc_type': ['simple', 'lstm', 'cnn'],
        'batch_size': [64, 128], 
        'learning_rate': [1e-4, 5e-4],
        'embedding_dim': [64, 128],
        'hidden_dim': [128, 256],
        'dropout_rate': [0.1, 0.3],
        'outer_epochs': [10, 25],
        'inner_epochs': [2, 3]
    },
    'output_dir': RESULTS_DIR / "discriminator_search",
}

# ============= UTILITY FUNCTIONS =============

def get_full_config(override_params=None):
    """
    Create a complete configuration by starting with defaults and applying overrides.
    
    Args:
        override_params: Dictionary of parameters to override the defaults
        
    Returns:
        Dictionary with complete configuration
    """
    # Start with default config
    config_dict = DEFAULT_DISCRIMINATOR_CONFIG.copy()
    
    # Apply overrides if provided
    if override_params:
        config_dict.update(override_params)
        
    return config_dict

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def get_config_hash(config_dict):
    """Create a unique hash for a configuration to use in folder names."""
    # Sort keys for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

def generate_configs(param_grid=None):
    """
    Generate all combinations of parameters for grid search.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of configuration dictionaries
    """
    if param_grid is None:
        param_grid = PARALLEL_CONFIG['param_grid']
        
    keys = param_grid.keys()
    values = param_grid.values()
    
    # Generate all combinations
    configs = []
    for items in itertools.product(*values):
        config_dict = dict(zip(keys, items))
        configs.append(config_dict)
    
    return configs

def get_gpu_memory():
    """Get memory usage for each GPU."""
    if not th.cuda.is_available():
        return []
    
    memory_info = []
    num_gpus = th.cuda.device_count()
    
    for i in range(num_gpus):
        # Get memory info
        total_memory = th.cuda.get_device_properties(i).total_memory
        reserved_memory = th.cuda.memory_reserved(i)
        allocated_memory = th.cuda.memory_allocated(i)
        free_mem = total_memory - reserved_memory - allocated_memory
        
        memory_info.append({
            'id': i,
            'total': total_memory,
            'reserved': reserved_memory,
            'allocated': allocated_memory,
            'free': free_mem,
            'free_percent': free_mem / total_memory
        })
    
    return memory_info

def get_free_gpu():
    """Find a free GPU with the most available memory."""
    try:
        gpu_info = get_gpu_memory()
        
        # No GPUs available
        if not gpu_info:
            return None
        
        # Sort by free memory (highest first)
        sorted_gpus = sorted(gpu_info, key=lambda x: x['free'], reverse=True)
        
        # Return GPU with most free memory if it's at least 1GB free
        if sorted_gpus[0]['free'] > 1e9:  # 1GB
            return sorted_gpus[0]['id']
        
        return None
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return None

def run_subprocess(config_dict, seed, gpu_id, output_dir):
    """Run training in a separate process to isolate GPU memory usage."""
    # Create a unique ID for this run
    config_id = get_config_hash(config_dict)
    
    # Create output directory
    config_dir = output_dir / f"config_{config_id}"
    seed_dir = config_dir / f"seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    
    # Save configuration
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Create a small Python script to run the training
    script_path = seed_dir / "run_training.py"
    
    with open(script_path, "w") as f:
        f.write(f"""
import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path

# Set up directories that would normally be in config
sys.path.append("{BASE_DIR}")

# Import the training function
from train_discriminator import train_discriminator

# Set fixed parameters (normally from config)
VOCAB_SIZE = {VOCAB_SIZE}
SEQ_LENGTH = {SEQ_LENGTH}
START_TOKEN = {START_TOKEN}
GENERATED_NUM = {GENERATED_NUM}
ORACLE_EMB_DIM = {ORACLE_EMB_DIM}
ORACLE_HIDDEN_DIM = {ORACLE_HIDDEN_DIM}
GEN_PRETRAIN_PATH = "{GEN_PRETRAIN_PATH}"
TARGET_PARAMS_PATH = "{TARGET_PARAMS_PATH}"

# Set seed
seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Load configuration
with open("{seed_dir / 'config.json'}", 'r') as f:
    config_dict = json.load(f)

# Run training
try:
    # Monkey patch config module attributes into globals for train_discriminator
    import sys
    import types
    config_module = types.ModuleType('config')
    config_module.VOCAB_SIZE = VOCAB_SIZE
    config_module.SEQ_LENGTH = SEQ_LENGTH
    config_module.START_TOKEN = START_TOKEN
    config_module.GENERATED_NUM = GENERATED_NUM
    config_module.ORACLE_EMB_DIM = ORACLE_EMB_DIM
    config_module.ORACLE_HIDDEN_DIM = ORACLE_HIDDEN_DIM
    config_module.TARGET_PARAMS_PATH = TARGET_PARAMS_PATH
    config_module.GEN_PRETRAIN_PATH = GEN_PRETRAIN_PATH
    sys.modules['config'] = config_module
    
    results = train_discriminator(
        disc_config=config_dict,
        device=device,
        log_dir=Path("{seed_dir}")
    )
    
    print("Training completed successfully.")
    sys.exit(0)
except Exception as e:
    print(f"Error during training: {{str(e)}}")
    with open("{seed_dir / 'error.txt'}", 'w') as f:
        f.write(str(e))
    sys.exit(1)
""")
    
    # Create environment variables for subprocess
    env = os.environ.copy()
    
    # Set CUDA_VISIBLE_DEVICES to control which GPU is used
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Run the subprocess
    process = subprocess.Popen(
        ["python", str(script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process, seed_dir

def run_parallel_training():
    """Run training in parallel using multiple GPUs."""
    # Generate all configurations
    configs = generate_configs()
    output_dir = Path(PARALLEL_CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running {len(configs)} configurations with {PARALLEL_CONFIG['num_seeds']} seeds each")
    print(f"Total runs: {len(configs) * PARALLEL_CONFIG['num_seeds']}")
    
    # Check that required model files exist
    if not os.path.exists(GEN_PRETRAIN_PATH):
        print(f"ERROR: Pretrained generator not found at {GEN_PRETRAIN_PATH}")
        return
    
    if not os.path.exists(TARGET_PARAMS_PATH):
        print(f"ERROR: Target parameters not found at {TARGET_PARAMS_PATH}")
        return
    
    # Prepare all training tasks
    all_tasks = []
    for config_id, config_dict in enumerate(configs):
        config_hash = get_config_hash(config_dict)
        config_dir = output_dir / f"config_{config_hash}"
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configuration summary
        with open(config_dir / "config_summary.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        for seed in range(PARALLEL_CONFIG['num_seeds']):
            all_tasks.append((config_dict, seed, None, output_dir))  # GPU will be assigned later
    
    # Detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("No GPUs found, using CPU")
        num_gpus = 0
    
    # Track active processes and GPU usage
    active_processes = {}  # task_id -> (process, output_dir, gpu_id)
    completed_tasks = []
    task_status = {i: "pending" for i in range(len(all_tasks))}
    gpu_in_use = {i: False for i in range(num_gpus)} if num_gpus > 0 else {}
    
    # Track timing for progress estimation
    start_time = time.time()
    task_times = []
    
    # Main loop - keep going until all tasks are complete
    while len(completed_tasks) < len(all_tasks):
        # Start new tasks if GPUs are available
        for i, task in enumerate(all_tasks):
            if task_status[i] == "pending":
                # Find a free GPU
                free_gpu = None
                for gpu_id in range(num_gpus):
                    if not gpu_in_use.get(gpu_id, True):  # Default to True (in use) if key doesn't exist
                        free_gpu = gpu_id
                        break
                
                if free_gpu is not None or num_gpus == 0:  # Either we found a GPU or we're using CPU
                    # Update the task with the assigned GPU
                    config_dict, seed, _, output_dir = task
                    task_with_gpu = (config_dict, seed, free_gpu, output_dir)
                    
                    # Start the process
                    process, log_dir = run_subprocess(*task_with_gpu)
                    
                    # Update tracking
                    active_processes[i] = (process, log_dir, free_gpu)
                    task_status[i] = "running"
                    if free_gpu is not None:
                        gpu_in_use[free_gpu] = True
                    
                    print(f"Started task {i+1}/{len(all_tasks)}: config {get_config_hash(config_dict)}, seed {seed} on {'GPU '+str(free_gpu) if free_gpu is not None else 'CPU'}")
        
        # Check for completed processes
        for i in list(active_processes.keys()):
            process, log_dir, gpu_id = active_processes[i]
            if process.poll() is not None:
                # Get the task details
                config_dict, seed, _, _ = all_tasks[i]
                
                # Process output
                stdout, stderr = process.communicate()
                with open(log_dir / "process_output.log", "w") as f:
                    f.write(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
                
                # Record task completion time
                task_time = time.time() - start_time - sum(task_times)
                task_times.append(task_time)
                
                # Mark GPU as free
                if gpu_id is not None:
                    gpu_in_use[gpu_id] = False
                
                # Update status
                if process.returncode == 0:
                    task_status[i] = "completed"
                    print(f"Completed task {i+1}/{len(all_tasks)}: config {get_config_hash(config_dict)}, seed {seed}")
                else:
                    task_status[i] = "failed"
                    print(f"Task {i+1}/{len(all_tasks)} failed: config {get_config_hash(config_dict)}, seed {seed}")
                    print(f"Error details saved to {log_dir}/process_output.log and {log_dir}/error.txt")
                
                completed_tasks.append(i)
                
                # Remove from active processes
                del active_processes[i]
        
        # Show progress
        elapsed_time = time.time() - start_time
        completed = len(completed_tasks)
        total = len(all_tasks)
        
        if completed > 0:
            # Calculate estimated time remaining
            avg_time_per_task = sum(task_times) / completed
            est_remaining = avg_time_per_task * (total - completed)
            
            print(f"\rProgress: {completed}/{total} ({completed/total*100:.1f}%) - "
                  f"Elapsed: {elapsed_time/60:.1f} min - "
                  f"Est. remaining: {est_remaining/60:.1f} min - "
                  f"Active tasks: {len(active_processes)}", end="")
        
        # Avoid hammering the CPU with constant checks
        time.sleep(5)
        
        # Print a newline occasionally to keep log readable
        if completed % 5 == 0 and completed > 0:
            print()
    
    print("\n\nAll tasks completed!")
    
    # Analyze and summarize results
    print("\nAnalyzing results...")
    summarize_results(output_dir, configs)

def summarize_results(output_dir, configs):
    """Analyze all results and create summary reports."""
    # Collect results for each configuration
    config_results = {}
    
    # Process each configuration
    for config_dict in configs:
        config_hash = get_config_hash(config_dict)
        config_dir = output_dir / f"config_{config_hash}"
        
        if not os.path.exists(config_dir):
            print(f"Warning: Results directory not found for config {config_hash}")
            continue
        
        # Initialize results for this config
        config_results[config_hash] = {
            'config': config_dict,
            'seeds': []
        }
        
        # Collect results from each seed
        successful_seeds = 0
        accuracies = []
        real_probs = []
        fake_probs = []
        training_times = []
        
        for seed in range(PARALLEL_CONFIG['num_seeds']):
            seed_dir = config_dir / f"seed_{seed}"
            results_path = seed_dir / "results.json"
            
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    
                    # Extract metrics
                    final_metrics = results.get('final_metrics', {})
                    accuracy = final_metrics.get('accuracy', 0)
                    real_prob = final_metrics.get('real_prob', 0)
                    fake_prob = final_metrics.get('fake_prob', 0)
                    training_time = results.get('training_time', 0)
                    
                    seed_result = {
                        'seed': seed,
                        'success': True,
                        'accuracy': accuracy,
                        'real_prob': real_prob,
                        'fake_prob': fake_prob,
                        'training_time': training_time
                    }
                    
                    config_results[config_hash]['seeds'].append(seed_result)
                    
                    # Add to aggregate metrics
                    successful_seeds += 1
                    accuracies.append(accuracy)
                    real_probs.append(real_prob)
                    fake_probs.append(fake_prob)
                    training_times.append(training_time)
                    
                except Exception as e:
                    print(f"Error loading results for config {config_hash}, seed {seed}: {e}")
                    config_results[config_hash]['seeds'].append({
                        'seed': seed,
                        'success': False,
                        'error': str(e)
                    })
            else:
                config_results[config_hash]['seeds'].append({
                    'seed': seed,
                    'success': False,
                    'error': "Results file not found"
                })
        
        # Calculate average metrics
        if successful_seeds > 0:
            avg_metrics = {
                'accuracy': sum(accuracies) / successful_seeds,
                'real_prob': sum(real_probs) / successful_seeds,
                'fake_prob': sum(fake_probs) / successful_seeds,
                'training_time': sum(training_times) / successful_seeds,
                'successful_seeds': successful_seeds
            }
            config_results[config_hash]['avg_metrics'] = avg_metrics
        else:
            config_results[config_hash]['avg_metrics'] = {'error': "No successful runs"}
        
        # Save summary for this configuration
        with open(config_dir / "summary.json", "w") as f:
            json.dump(config_results[config_hash], f, indent=2)
    
    # Find the best configuration based on accuracy
    best_config = None
    best_accuracy = -1
    
    for config_hash, result in config_results.items():
        avg_metrics = result.get('avg_metrics', {})
        if 'accuracy' in avg_metrics:
            accuracy = avg_metrics['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config_hash
    
    # Create the overall summary
    overall_summary = {
        'total_configs': len(configs),
        'seeds_per_config': PARALLEL_CONFIG['num_seeds'],
        'best_config_hash': best_config,
        'best_accuracy': best_accuracy,
        'best_config': config_results.get(best_config, {}).get('config') if best_config else None,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save overall summary
    with open(output_dir / "overall_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)
    
    # Save detailed results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(config_results, f, indent=2)
    
    # Print the best configuration
    if best_config:
        print(f"\nBest configuration (accuracy: {best_accuracy:.4f}):")
        for k, v in config_results[best_config]['config'].items():
            print(f"  {k}: {v}")
    else:
        print("\nNo successful configurations found.")
    
    print(f"\nAll results saved to {output_dir}")
    print(f"  - Summary: {output_dir / 'overall_summary.json'}")
    print(f"  - Detailed results: {output_dir / 'all_results.json'}")

def main():
    # You can customize the hyperparameter search here
    # For example, to try different configurations:
    # PARALLEL_CONFIG['param_grid']['disc_type'] = ['lstm', 'cnn']  # Remove 'simple'
    # PARALLEL_CONFIG['param_grid']['batch_size'] = [64, 128, 256]  # Add a larger batch size
    
    # Set the number of seeds per configuration
    # PARALLEL_CONFIG['num_seeds'] = 5  # Increase for more robust results
    
    # Run the parallel training
    run_parallel_training()

if __name__ == "__main__":
    main()