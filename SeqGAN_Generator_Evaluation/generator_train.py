import os
import time
import json
import random
import numpy as np
import torch as th
from pathlib import Path
import sys
from torch.utils.data import DataLoader, TensorDataset

# Import models
from oracle import Oracle
from generator import Generator, generator_adversarial_update
from discriminator import Discriminator_LSTM_Frensi, evaluate_discriminator, pretrain_discriminator
from rollout import Rollout
from generator import pretrain_generator

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.getenv('WORKING_DIR', Path(os.path.dirname(os.path.abspath(__file__)))))
SAVE_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============= FIXED PARAMETERS =============
# Data parameters
VOCAB_SIZE = 5000
SEQ_LENGTH = 20
START_TOKEN = 0
GENERATED_NUM = 5000  # Number of samples to generate for testing

# Oracle/Generator model parameters
ORACLE_EMB_DIM = 32
ORACLE_HIDDEN_DIM = 32

# Paths for models
ORACLE_PARAMS_PATH = SAVE_DIR / 'target_params.pkl'
GEN_PRETRAIN_PATH = SAVE_DIR / 'generator_pretrained.pth'
DISC_PRETRAIN_PATH = SAVE_DIR / 'discriminator_pretrained.pth'

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def main():
    """Main function to run a single generator training."""
    
    # Get environment variables
    config_path = os.getenv('CONFIG_PATH')
    seed = int(os.getenv('SEED', '0'))
    output_dir = Path(os.getenv('OUTPUT_DIR', RESULTS_DIR / "generator_runs"))
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load configuration
    print(f"Loading config from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if th.cuda.is_available():
        gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        device = th.device("cuda:0")
        print(f"Using GPU: {gpu_id}")
    else:
        device = th.device("cpu")
        print("Using CPU")
    
    # Start timing
    start_time = time.time()
    
    # Create log file paths
    log_folder = output_dir
    log_file = os.path.join(log_folder, "training.log")
    gen_pretrain_log = os.path.join(log_folder, "generator_pretrain.txt")
    disc_pretrain_log = os.path.join(log_folder, "discriminator_pretrain.txt")
    adversarial_log = os.path.join(log_folder, "adversarial_training_log.txt")
    reward_log = os.path.join(log_folder, "rewards_log.txt")
    
    # Print training configuration
    print(f"Training generator with:")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Generator Learning Rate: {config['g_learning_rate']}")
    print(f"  Discriminator Learning Rate: {config['d_learning_rate']}")
    print(f"  Pre-training Epochs: {config['pretrain_epochs']}")
    print(f"  Rollout Number: {config['rollout_num']}")
    print(f"  Total Adversarial Epochs: {config['adv_epochs']}")
    
    # Create Oracle
    oracle = Oracle(
        vocab_size=VOCAB_SIZE,
        embedding_dim=ORACLE_EMB_DIM,
        hidden_dim=ORACLE_HIDDEN_DIM,
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        device=device
    )
    
    # Load oracle parameters
    print(f"Loading oracle parameters from {ORACLE_PARAMS_PATH}...")
    try:
        oracle.load_params(ORACLE_PARAMS_PATH)
        if not os.path.exists(ORACLE_PARAMS_PATH):
            raise FileNotFoundError(f"Oracle parameter file not found: {ORACLE_PARAMS_PATH}")
    except Exception as e:
        print(f"Error loading oracle parameters: {e}")
        sys.exit(1)  # Exit with error code
    
    # Create Generator
    generator = Generator(
        vocab_size=VOCAB_SIZE,
        embedding_dim=config.get('g_embedding_dim', ORACLE_EMB_DIM),
        hidden_dim=config.get('g_hidden_dim', ORACLE_HIDDEN_DIM),
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        device=device
    )
    
    # Create discriminator
    discriminator = Discriminator_LSTM_Frensi(
        vocab_size=VOCAB_SIZE,
        embedding_dim=config.get('d_embedding_dim', 128),
        hidden_dim=config.get('d_hidden_dim', 256),
        dropout_rate=config.get('dropout_rate', 0.02),
        device=device
    )
    
    # Initialize optimizers
    g_optimizer_pretrain = th.optim.Adam(generator.parameters(), lr=config.get('g_pretrain_lr', 1e-3))
    d_optimizer_pretrain = th.optim.Adam(discriminator.parameters(), lr=config.get('d_pretrain_lr', 4e-4))
    
    # Pretraining phase
    if config.get('do_pretrain', True):

        print("Starting generator pretraining...")
        pretrain_generator(
            target_lstm=oracle,
            generator=generator,
            optimizer=g_optimizer_pretrain,
            pre_epoch_num=config['pretrain_epochs'],
            batch_size=config['batch_size'],
            generated_num=GENERATED_NUM,
            eval_freq=config.get('eval_freq', 5),
            lr_patience=config.get('lr_patience', 5),
            lr_decay=config.get('lr_decay', 0.5),
            log_path=gen_pretrain_log
        )
        
        print("Starting discriminator pretraining...")
        pretrain_discriminator(
            target_lstm=oracle,
            generator=generator,
            discriminator=discriminator,
            optimizer=d_optimizer_pretrain,
            outer_epochs=config.get('d_outer_epochs', 10),
            inner_epochs=config.get('d_inner_epochs', 2),
            batch_size=config['batch_size'],
            generated_num=GENERATED_NUM,
            log_file=disc_pretrain_log
        )
        
        # Save pretrained models
        gen_save_path = os.path.join(output_dir, "generator_pretrained.pth")
        disc_save_path = os.path.join(output_dir, "discriminator_pretrained.pth")
        th.save(generator.state_dict(), gen_save_path)
        th.save(discriminator.state_dict(), disc_save_path)
        print(f"Saved pretrained models to {output_dir}")

    else:
        # Load pretrained models if not doing pretraining
        try:
            print("Loading pretrained models...")
            gen_load_path = config.get('gen_load_path', GEN_PRETRAIN_PATH)
            disc_load_path = config.get('disc_load_path', DISC_PRETRAIN_PATH)
            
            generator.load_state_dict(th.load(gen_load_path, map_location=device))
            
            checkpoint = th.load(disc_load_path, map_location=device)
            discriminator.load_state_dict(checkpoint['model_state_dict'])
            d_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            #discriminator.load_state_dict(th.load(disc_load_path, map_location=device))
            
        except Exception as e:
            print(f"Error loading pretrained models: {e}")
            sys.exit(1)
    
    # Initialize rollout module
    rollout = Rollout(
        generator=generator,
        discriminator=discriminator,
        rollout_num=config['rollout_num'],
        update_rate=config.get('rollout_update_rate', 0.8),
        device=device
    )
    
    # Initialize optimizers for adversarial training (potentially with different learning rates)
    g_optimizer = th.optim.Adam(generator.parameters(), lr=config['g_learning_rate'])
    d_optimizer = th.optim.Adam(discriminator.parameters(), lr=config['d_learning_rate'])
    
    # Start adversarial training
    print("Starting adversarial training...")
    
    train_seqgan(
        generator=generator,
        discriminator=discriminator,
        rollout=rollout,
        target_lstm=oracle,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        num_epochs=config['adv_epochs'],
        batch_size=config['batch_size'],
        generated_num=GENERATED_NUM,
        g_steps=config.get('g_steps', 1),
        d_steps=config.get('d_steps', 5),
        k_epochs=config.get('k_epochs', 3),
        log_path=adversarial_log,
        log_path_reward=reward_log,
        device=device
    )
    
    # Save final models
    final_gen_path = os.path.join(output_dir, "generator_final.pth")
    final_disc_path = os.path.join(output_dir, "discriminator_final.pth")
    th.save(generator.state_dict(), final_gen_path)
    th.save(discriminator.state_dict(), final_disc_path)
    
    # Record training time
    training_time = time.time() - start_time
    
    # Add seed to config for results
    config_with_seed = config.copy()
    config_with_seed['seed'] = seed
    
    # Generate final samples for evaluation
    generator.eval()
    final_samples = generator.generate(GENERATED_NUM)
    final_nll = oracle.calculate_nll(final_samples)
    
    # Create results summary
    results = {
        "config": config_with_seed,
        "training_time": training_time,
        "final_metrics": {
            "nll": final_nll,
            "discriminator": evaluate_discriminator(discriminator, oracle, generator, num_samples=1000)
        },
        "model_paths": {
            "generator": str(final_gen_path),
            "discriminator": str(final_disc_path)
        }
    }
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed in {training_time:.2f} seconds!")
    print(f"Results saved to {output_dir}")
    print(f"Final NLL: {final_nll:.4f}")

if __name__ == "__main__":
    main()