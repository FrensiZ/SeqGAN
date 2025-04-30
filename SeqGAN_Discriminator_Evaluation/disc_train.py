import os
import time
import json
import random
import numpy as np
import torch as th
from pathlib import Path
import sys

# Import models
from oracle import Oracle
from generator import Generator
from discriminator import (
    Discriminator_LSTM_Frensi, 
    Discriminator_CNN, 
    Discriminator_LSTM_Surag,
    pretrain_discriminator,
    evaluate_discriminator
)

# ============= BASE DIRECTORIES =============
BASE_DIR = Path(os.getenv('WORKING_DIR', Path(os.path.dirname(os.path.abspath(__file__)))))
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
ORACLE_EMB_DIM = 32
ORACLE_HIDDEN_DIM = 32

# Paths for models
GEN_PRETRAIN_PATH = SAVE_DIR / 'generator_pretrained.pth'
TARGET_PARAMS_PATH = SAVE_DIR / 'target_params.pkl'

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

def create_discriminator(disc_type, vocab_size, embedding_dim, hidden_dim, dropout, device):

    if disc_type == 'simple':
        return Discriminator_LSTM_Frensi(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            device=device
        )
    elif disc_type == 'lstm':
        return Discriminator_LSTM_Surag(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            device=device
        )
    elif disc_type == 'cnn':
        return Discriminator_CNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            sequence_length=SEQ_LENGTH,
            device=device,
            dropout_prob=dropout
        )
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")

def main():
    """Main function to run a single discriminator training."""
    
    # Get environment variables
    config_path = os.getenv('CONFIG_PATH')
    seed = int(os.getenv('SEED', '0'))
    output_dir = Path(os.getenv('OUTPUT_DIR', RESULTS_DIR / "discriminator_runs"))
    
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
    
    # Create log file
    log_file = os.path.join(output_dir, "training.log")
    
    # Print training configuration
    print(f"Training discriminator with:")
    print(f"  Type: {config['disc_type']}")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Embedding Dim: {config['embedding_dim']}")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  Dropout Rate: {config['dropout_rate']}")
    print(f"  Outer Epochs: {config['outer_epochs']}")
    print(f"  Inner Epochs: {config['inner_epochs']}")
    
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
    print(f"Loading oracle parameters from {TARGET_PARAMS_PATH}...")
    try:
        oracle.load_params(TARGET_PARAMS_PATH)
        if not os.path.exists(TARGET_PARAMS_PATH):
            raise FileNotFoundError(f"Oracle parameter file not found: {TARGET_PARAMS_PATH}")
    except Exception as e:
        print(f"Error loading oracle parameters: {e}")
        sys.exit(1)  # Exit with error code
    
    # Create Generator
    generator = Generator(
        vocab_size=VOCAB_SIZE,
        embedding_dim=ORACLE_EMB_DIM,
        hidden_dim=ORACLE_HIDDEN_DIM,
        sequence_length=SEQ_LENGTH,
        start_token=START_TOKEN,
        device=device
    )
    
    # Load pretrained generator
    print(f"Loading pretrained generator from {GEN_PRETRAIN_PATH}...")
    try:
        if not os.path.exists(GEN_PRETRAIN_PATH):
            raise FileNotFoundError(f"Generator model file not found: {GEN_PRETRAIN_PATH}")
        generator.load_state_dict(th.load(GEN_PRETRAIN_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading pretrained generator: {e}")
        sys.exit(1)  # Exit with error code
    
    # Create discriminator
    discriminator = create_discriminator(
        disc_type=config['disc_type'],
        vocab_size=VOCAB_SIZE,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout_rate'],
        device=device
    )
    
    # Create optimizer
    optimizer = th.optim.Adam(discriminator.parameters(), lr=config['learning_rate'])
    
    # Train discriminator
    pretrain_discriminator(
        target_lstm=oracle,
        generator=generator,
        discriminator=discriminator,
        optimizer=optimizer,
        outer_epochs=config['outer_epochs'],
        inner_epochs=config['inner_epochs'],
        batch_size=config['batch_size'],
        generated_num=GENERATED_NUM,
        log_file=log_file
    )
    
    # Evaluate final model
    print("Evaluating final model...")
    final_metrics = evaluate_discriminator(
        discriminator=discriminator,
        target_lstm=oracle,
        generator=generator,
        num_samples=1000
    )
    
    # Save discriminator model
    model_path = os.path.join(output_dir, "discriminator_model.pth")
    th.save(discriminator.state_dict(), model_path)
    
    # Record training time
    training_time = time.time() - start_time
    
    # Add seed to config for results
    config_with_seed = config.copy()
    config_with_seed['seed'] = seed
    
    # Create results summary
    results = {
        "config": config_with_seed,
        "training_time": training_time,
        "final_metrics": final_metrics,
        "model_path": str(model_path)
    }
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed in {training_time:.2f} seconds!")
    print(f"Results saved to {output_dir}")
    print(f"Final metrics: {final_metrics}")

if __name__ == "__main__":
    main()