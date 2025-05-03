import os
import torch as th
import subprocess
from pathlib import Path
import json
import numpy as np
import itertools
import time
import hashlib
import pickle

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============= PARALLEL TRAINING PARAMETERS =============
# Settings for hyperparameter search
PARALLEL_CONFIG = {
    'num_seeds': 8,                # Number of random seeds to test each configuration
    'param_grid': {
        'disc_type': ['simple'],
        'batch_size': [64], 
        'learning_rate': [1e-3],
        'embedding_dim': [128],
        'hidden_dim': [256],
        'dropout_rate': [0.01],
        'outer_epochs': [200],
        'inner_epochs': [1],
        'lr_patience':[5],
        'lr_decay':[0.5],
    },
    'output_dir': RESULTS_DIR / "discriminator_search",
}

def get_config_hash(config):
    """Generate a unique hash for a configuration."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

def get_free_gpus():
    """Find all free GPUs to use from the allowed GPUs."""
    allowed_gpus = [1,2,3,4]  # Only use these GPUs
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,nounits,noheader'], 
            capture_output=True, text=True, timeout=3
        )
        
        if result.returncode != 0:
            return allowed_gpus  # Default to all allowed GPUs if check fails
        
        free_gpus = []
        
        for i, line in enumerate(result.stdout.strip().split('\n')):
            if i in allowed_gpus:  # Only check allowed GPUs
                used, free, util = map(int, line.split(','))
                if used < 100 and util < 5:  # Consider GPU free if low usage
                    free_gpus.append(i)
                
        return free_gpus if free_gpus else allowed_gpus
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        return allowed_gpus  # Default to all allowed GPUs


def generate_configs(param_grid):
    """Generate all possible configurations from the parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]    

def run_training(config, gpu_id, seed, output_dir):
    """
    Run a single training job as a subprocess.
    
    Args:
        config: Configuration dictionary
        gpu_id: GPU ID to use
        seed: Random seed value
        output_dir: Directory to save results
    
    Returns:
        subprocess.Popen: Running process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration to file
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Prepare environment variables
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "CONFIG_PATH": str(config_path),
        "OUTPUT_DIR": str(output_dir),
        "SEED": str(seed),
        "WORKING_DIR": str(BASE_DIR)
    })
    
    # Start training process
    process = subprocess.Popen(
        ["python3", str(BASE_DIR / "disc_train.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def analyze_results(output_dir, configs, num_seeds):
    """
    Analyze results from all training runs.
    
    Args:
        output_dir: Base output directory
        configs: List of configurations tested
        num_seeds: Number of seeds used per configuration
    """
    print("\nAnalyzing results...")
    
    # Collect results for all configurations and seeds
    results = []
    
    for config_id, config in enumerate(configs):
        config_hash = get_config_hash(config)
        config_results = []
        
        # Collect results for each seed
        for seed in range(num_seeds):
            seed_dir = output_dir / f"config_{config_id}_seed_{seed}"
            results_path = seed_dir / "results.json"
            
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        seed_result = json.load(f)
                    
                    config_results.append({
                        'seed': seed,
                        'metrics': seed_result['final_metrics'],
                        'training_time': seed_result['training_time']
                    })
                except Exception as e:
                    print(f"Error reading results for config {config_id}, seed {seed}: {e}")
        
        # Skip if no results for this configuration
        if not config_results:
            continue
        
        # Calculate average metrics
        avg_accuracy = np.mean([r['metrics']['accuracy'] for r in config_results])
        std_accuracy = np.std([r['metrics']['accuracy'] for r in config_results])
        avg_real_prob = np.mean([r['metrics']['real_prob'] for r in config_results])
        avg_fake_prob = np.mean([r['metrics']['fake_prob'] for r in config_results])
        avg_time = np.mean([r['training_time'] for r in config_results])
        
        # Calculate discriminator capability (higher is better)
        # We want high real_prob and low fake_prob
        disc_capability = avg_real_prob - avg_fake_prob
        
        results.append({
            'config_id': config_id,
            'config': config,
            'hash': config_hash,
            'avg_accuracy': float(avg_accuracy),
            'std_accuracy': float(std_accuracy),
            'avg_real_prob': float(avg_real_prob),
            'avg_fake_prob': float(avg_fake_prob),
            'disc_capability': float(disc_capability),
            'avg_training_time': float(avg_time),
            'seeds_completed': len(config_results)
        })
    
    # Save all results
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Sort by discriminator capability (higher is better)
    results.sort(key=lambda x: x['disc_capability'], reverse=True)
    
    # Create top configs report
    report_path = output_dir / "top_configs.txt"
    with open(report_path, 'w') as f:
        f.write("TOP DISCRIMINATOR CONFIGURATIONS\n")
        f.write("=" * 40 + "\n\n")
        
        for i, result in enumerate(results[:5]):
            output = (
                f"Rank {i+1}: {result['config']['disc_type']} (ID: {result['config_id']})\n"
                f"  Discriminator Capability: {result['disc_capability']:.4f}\n"
                f"  Accuracy: {result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n"
                f"  Real Prob: {result['avg_real_prob']:.4f}, Fake Prob: {result['avg_fake_prob']:.4f}\n"
                f"  Avg Training Time: {result['avg_training_time']:.1f} seconds\n"
                f"  Seeds Completed: {result['seeds_completed']}/{num_seeds}\n"
                f"  Configuration:\n"
            )
            
            # Add configuration details with indentation
            for k, v in result['config'].items():
                output += f"    {k}: {v}\n"
            
            output += "\n" + "-" * 40 + "\n\n"
            
            f.write(output)
            print(output)
    
    # Find best for each discriminator type
    best_by_type = {}
    for result in results:
        disc_type = result['config']['disc_type']
        if disc_type not in best_by_type:
            best_by_type[disc_type] = result
    
    # Create best by type report
    type_report_path = output_dir / "best_by_type.txt"
    with open(type_report_path, 'w') as f:
        f.write("BEST CONFIGURATION BY DISCRIMINATOR TYPE\n")
        f.write("=" * 40 + "\n\n")
        
        for disc_type, result in best_by_type.items():
            output = (
                f"Best {disc_type.upper()} Configuration:\n"
                f"  Config ID: {result['config_id']}\n"
                f"  Discriminator Capability: {result['disc_capability']:.4f}\n"
                f"  Accuracy: {result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n"
                f"  Configuration:\n"
            )
            
            # Add configuration details with indentation
            for k, v in result['config'].items():
                output += f"    {k}: {v}\n"
            
            output += "\n" + "-" * 40 + "\n\n"
            
            f.write(output)
            print(output)
    
    print(f"Analysis complete. Reports saved to {output_dir}")

def main():
    """Main function to run parallel training."""
    # Create output directory
    output_dir = PARALLEL_CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all configurations
    configs = generate_configs(PARALLEL_CONFIG['param_grid'])
    
    # Save configurations
    with open(output_dir / "all_configs.json", 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated {len(configs)} configurations")
    print(f"Running {PARALLEL_CONFIG['num_seeds']} seeds for each configuration")
    print(f"Total runs: {len(configs) * PARALLEL_CONFIG['num_seeds']}")
    
    # Track active processes
    active_processes = {}  # (config_id, seed): {'process': process, 'gpu': gpu_id, 'output_dir': output_dir}
    completed_runs = set()  # Set of (config_id, seed) that have completed


    while len(completed_runs) < len(configs) * PARALLEL_CONFIG['num_seeds']:
        # Get all available GPUs
        free_gpus = get_free_gpus()
        
        # Start a job on each free GPU if we have pending runs
        for gpu_id in free_gpus:
            # Skip if this GPU already has a process running
            if any(info['gpu'] == gpu_id for info in active_processes.values()):
                continue
                
            # Find a configuration and seed to run
            found_job = False
            for config_id, config in enumerate(configs):
                for seed in range(PARALLEL_CONFIG['num_seeds']):
                    run_key = (config_id, seed)
                    
                    if run_key not in completed_runs and run_key not in active_processes:
                        # Create output directory for this run
                        run_dir = output_dir / f"config_{config_id}_seed_{seed}"
                        
                        # Start the training process
                        try:
                            print(f"Starting config {config_id} (type: {config['disc_type']}), seed {seed} on GPU {gpu_id}")
                            process = run_training(config, gpu_id, seed, run_dir)
                            
                            # Track the process
                            active_processes[run_key] = {
                                'process': process,
                                'gpu': gpu_id,
                                'output_dir': run_dir,
                                'start_time': time.time()
                            }
                            
                            found_job = True
                            break
                        except Exception as e:
                            print(f"Error starting run for config {config_id}, seed {seed}: {e}")
                            continue
                            
                if found_job:
                    break
        
        # Check active processes for completion
        for run_key in list(active_processes.keys()):
            process_info = active_processes[run_key]
            process = process_info['process']
            
            # Check if process has completed
            if process.poll() is not None:
                config_id, seed = run_key
                run_dir = process_info['output_dir']
                
                # Get output from process
                stdout, stderr = process.communicate()
                
                # Save logs
                with open(run_dir / "stdout.log", 'w') as f:
                    f.write(stdout)
                with open(run_dir / "stderr.log", 'w') as f:
                    f.write(stderr)
                
                # Check if successful
                if process.returncode == 0:
                    elapsed = time.time() - process_info['start_time']
                    print(f"Completed config {config_id}, seed {seed} in {elapsed:.1f} seconds")
                    completed_runs.add(run_key)
                else:
                    print(f"Failed config {config_id}, seed {seed}, return code: {process.returncode}")
                    # Still mark as completed to avoid retrying
                    completed_runs.add(run_key)
                
                # Remove from active processes
                del active_processes[run_key]
        
        # Print status update periodically
        if len(completed_runs) % 5 == 0:
            print(f"Progress: {len(completed_runs)}/{len(configs) * PARALLEL_CONFIG['num_seeds']} runs completed")
            print(f"Active processes: {len(active_processes)}")
        
        # Sleep to prevent CPU spinning
        time.sleep(4)
    
    print("\nAll training runs completed!")
    
    # Analyze results
    analyze_results(output_dir, configs, PARALLEL_CONFIG['num_seeds'])
    
    print(f"All done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()