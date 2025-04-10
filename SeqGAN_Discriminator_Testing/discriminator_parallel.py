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

N_SEEDS = 3  # Number of random seeds to test each configuration

def get_config_hash(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        
        free_gpus = []
        for i, line in enumerate(result.stdout.strip().split('\n')):
            used, free, util = map(int, line.split(','))
            if used < 100 and util < 5:
                free_gpus.append(i)
        return free_gpus[0] if free_gpus else None
    except:
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
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
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

def analyze_results(base_dir):
    results = []
    
    config_ids = set()
    for seed_dir in base_dir.glob("disc_outputs/config_*/seed_*"):
        config_id = int(seed_dir.parent.name.split('_')[1])
        config_ids.add(config_id)
    
    for config_id in sorted(config_ids):
        try:
            first_seed_dir = base_dir / f"disc_outputs/config_{config_id}/seed_0"
            with open(first_seed_dir / "config.json") as f:
                config = json.load(f)
            
            # Gather metrics for all seeds
            all_metrics = []
            for seed in range(N_SEEDS):
                seed_dir = base_dir / f"disc_outputs/config_{config_id}/seed_{seed}"
                try:
                    with open(seed_dir / "discriminator_results.pkl", "rb") as f:
                        data = pickle.load(f)
                    
                    # Extract the latest evaluation metrics
                    final_metrics = {
                        'accuracy': data['metrics']['accuracy'][-1],
                        'real_prob': data['metrics']['real_prob'][-1],
                        'fake_prob': data['metrics']['fake_prob'][-1],
                    }
                    all_metrics.append(final_metrics)
                except Exception as e:
                    print(f"Error processing seed {seed} for config {config_id}: {e}")
            
            # Calculate aggregate metrics across seeds
            if all_metrics:
                # Convert to numpy arrays
                accuracy = np.array([m['accuracy'] for m in all_metrics])
                real_prob = np.array([m['real_prob'] for m in all_metrics])
                fake_prob = np.array([m['fake_prob'] for m in all_metrics])
                
                # Calculate separation score: high real_prob, low fake_prob, and high accuracy
                separation = real_prob - fake_prob
                
                # Calculate means and stds
                avg_accuracy = np.mean(accuracy)
                std_accuracy = np.std(accuracy)
                avg_separation = np.mean(separation)
                std_separation = np.std(separation)
                
                # Calculate combined score (higher is better)
                combined_score = avg_accuracy + 0.5 * avg_separation
                
                results.append({
                    'config_id': config_id,
                    'config': config,
                    'accuracy': f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
                    'separation': f"{avg_separation:.4f} ± {std_separation:.4f}",
                    'combined_score': combined_score,
                })
            
        except Exception as e:
            print(f"Error processing config {config_id}: {e}")
    
    # Sort by combined score (descending)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print("\nTop 5 discriminator configurations:")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Config {result['config_id']}")
        print(f"   Accuracy: {result['accuracy']}")
        print(f"   Real-Fake Separation: {result['separation']}")
        print(f"   Combined Score: {result['combined_score']:.4f}")
        print(json.dumps(result['config'], indent=2))
    
    top_configs_path = base_dir / "top_discriminator_configs.txt"
    with open(top_configs_path, 'w') as f:
        f.write("Top 5 discriminator configurations:\n")
        for i, result in enumerate(results[:5]):
            f.write(f"\n{i+1}. Config {result['config_id']}\n")
            f.write(f"   Accuracy: {result['accuracy']}\n")
            f.write(f"   Real-Fake Separation: {result['separation']}\n")
            f.write(f"   Combined Score: {result['combined_score']:.4f}\n")
            f.write(json.dumps(result['config'], indent=2) + "\n")

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
    
    print("\nAll processes completed, analyzing results...")
    analyze_results(base_dir)

if __name__ == "__main__":
    main()