import torch
import random
import numpy as np
import os
import itertools
from datetime import datetime

# Import models and functions
from oracle import TargetLSTM
from generator import Generator
from discriminator import (
    Discriminator, 
    Discriminator_Simple, 
    Discriminator_CNN, 
    pretrain_discriminator 
    #evaluate_discriminator
)
import config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_discriminator(disc_type, vocab_size, embedding_dim, hidden_dim, dropout, device):
    """Create the specified discriminator type."""
    if disc_type == 'simple':
        return Discriminator_Simple(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            device=device
        )
    elif disc_type == 'lstm':
        return Discriminator(
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
            sequence_length=config.SEQ_LENGTH,
            dropout_prob=dropout,
            device=device
        )
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")

def main():

    # ============ PARAMETERS (edit these) ============
    # Grid search parameters
    DISC_TYPES = ['cnn', 'lstm', 'simple']     # Options: 'simple', 'lstm', 'cnn'
    SEEDS = [config.SEED]                      # Random seeds for reproducibility
    
    # Training parameters
    OUTER_EPOCHS = [50]                        # Number of outer training epochs
    INNER_EPOCHS = [3]                         # Number of inner training epochs
    BATCH_SIZES = [64, 128]                    # Batch sizes for training
    LEARNING_RATES = [1e-4, 5e-4]              # Learning rates for optimizer
    
    # Discriminator hyperparameters
    EMBEDDING_DIMS = [64, 128]                 # Embedding dimensions
    HIDDEN_DIMS = [128, 256]                   # Hidden dimensions
    DROPOUT_RATES = [0.1, 0.3]                 # Dropout rates
    # =================================================
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"discriminator_grid_search_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Use CPU for now (GPU assignment will be added later)
    device = torch.device('cpu')
    print("Using CPU")
    
    # Save grid search configuration
    with open(os.path.join(base_output_dir, 'grid_config.txt'), 'w') as f:
        f.write(f"Discriminator Types: {DISC_TYPES}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Outer Epochs: {OUTER_EPOCHS}\n")
        f.write(f"Inner Epochs: {INNER_EPOCHS}\n")
        f.write(f"Batch Sizes: {BATCH_SIZES}\n")
        f.write(f"Learning Rates: {LEARNING_RATES}\n")
        f.write(f"Embedding Dimensions: {EMBEDDING_DIMS}\n")
        f.write(f"Hidden Dimensions: {HIDDEN_DIMS}\n")
        f.write(f"Dropout Rates: {DROPOUT_RATES}\n")
    
    # Initialize oracle and generator (shared across all experiments)
    target_lstm = TargetLSTM(config.VOCAB_SIZE, config.EMB_DIM, config.HIDDEN_DIM, config.SEQ_LENGTH, config.START_TOKEN, device).to(device)
    generator = Generator(config.VOCAB_SIZE, config.EMB_DIM, config.HIDDEN_DIM, config.SEQ_LENGTH, config.START_TOKEN, device).to(device)
    
    # Load oracle parameters
    print("Loading oracle parameters...")
    target_lstm.load_params(config.TARGET_PARAMS_PATH)
    
    # If generator pretrained exists, load it
    if os.path.exists(config.GEN_PRETRAIN_PATH):
        print("Loading pretrained generator...")
        generator.load_state_dict(torch.load(config.GEN_PRETRAIN_PATH, map_location=device))
    else:
        # If generator isn't available, we can't proceed with discriminator testing
        print("Pretrained generator not found. Please run generator pretraining first.")
        return
    
    # Create a grid of all parameter combinations
    param_grid = list(itertools.product(
        DISC_TYPES, SEEDS, OUTER_EPOCHS, INNER_EPOCHS, BATCH_SIZES, 
        LEARNING_RATES, EMBEDDING_DIMS, HIDDEN_DIMS, DROPOUT_RATES
    ))
    
    print(f"Total number of experiments: {len(param_grid)}")
    
    # Run each experiment in the grid
    for i, params in enumerate(param_grid):

        (disc_type, seed, outer_epochs, inner_epochs, batch_size, 
         learning_rate, embedding_dim, hidden_dim, dropout_rate) = params
        
        print(f"\n===== Experiment {i+1}/{len(param_grid)} =====")
        print(f"Testing {disc_type} discriminator with:")
        print(f"  Seed: {seed}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Embedding Dim: {embedding_dim}")
        print(f"  Hidden Dim: {hidden_dim}")
        print(f"  Dropout Rate: {dropout_rate}")
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Create experiment directory
        exp_dir = os.path.join(base_output_dir, f"exp_{i+1}_{disc_type}_bs{batch_size}_lr{learning_rate}_emb{embedding_dim}_hid{hidden_dim}_drop{dropout_rate}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create discriminator
        discriminator = create_discriminator(disc_type=disc_type, vocab_size=config.VOCAB_SIZE, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout_rate, device=device)
        
        # Create optimizer
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        
        # Save configuration for this experiment
        with open(os.path.join(exp_dir, 'config.txt'), 'w') as f:
            f.write(f"Discriminator Type: {disc_type}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Outer Epochs: {outer_epochs}\n")
            f.write(f"Inner Epochs: {inner_epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Embedding Dimension: {embedding_dim}\n")
            f.write(f"Hidden Dimension: {hidden_dim}\n")
            f.write(f"Dropout Rate: {dropout_rate}\n")
        
        # Create log file
        log_file = os.path.join(exp_dir, f'{disc_type}_discriminator.log')
        
        # Train discriminator
        pretrain_discriminator(target_lstm=target_lstm, generator=generator, discriminator=discriminator, optimizer=d_optimizer, 
                               outer_epochs=outer_epochs, inner_epochs=inner_epochs, batch_size=batch_size, generated_num=config.GENERATED_NUM, 
                               log_file=log_file, device=device)
        
        # Save the trained discriminator
        model_path = os.path.join(exp_dir, f'{disc_type}_discriminator.pth')
        torch.save(discriminator.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    print(f"\nAll experiments completed. Results saved to: {base_output_dir}")

if __name__ == "__main__":
    main()