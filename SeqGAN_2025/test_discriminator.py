import torch
import random
import numpy as np
import os
import itertools

# Import models and functions
from oracle import TargetLSTM
from generator import Generator
from discriminator import (
    Discriminator, 
    Discriminator_Simple, 
    Discriminator_CNN, 
    pretrain_discriminator, 
    evaluate_discriminator
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
    OUTER_EPOCHS = [2]                         # Number of outer training epochs
    INNER_EPOCHS = [2]                         # Number of inner training epochs
    BATCH_SIZES = [64, 128]                    # Batch sizes for training
    LEARNING_RATES = [1e-4, 5e-4]              # Learning rates for optimizer
    
    # Discriminator hyperparameters
    EMBEDDING_DIMS = [64, 128]                 # Embedding dimensions
    HIDDEN_DIMS = [128, 256]                   # Hidden dimensions
    DROPOUT_RATES = [0.1, 0.3]                 # Dropout rates
    # =================================================
    
    # Create base output directory
    base_output_dir = "discriminator_test"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Use CPU for now (GPU assignment will be added later)
    device = torch.device('cpu')
    print("Using CPU")
    
    # Initialize oracle and generator (shared across all experiments)
    target_lstm = TargetLSTM(
        config.VOCAB_SIZE, 
        config.EMB_DIM, 
        config.HIDDEN_DIM, 
        config.SEQ_LENGTH, 
        config.START_TOKEN, 
        device
    )
    
    generator = Generator(
        config.VOCAB_SIZE, 
        config.EMB_DIM, 
        config.HIDDEN_DIM, 
        config.SEQ_LENGTH, 
        config.START_TOKEN, 
        device
    )
    
    # Load oracle parameters
    print("Loading oracle parameters...")
    target_lstm.load_params(config.TARGET_PARAMS_PATH)
    
    # If generator pretrained exists, load it
    if os.path.exists(config.GEN_PRETRAIN_PATH):
        print("Loading pretrained generator...")
        generator.load_state_dict(torch.load(config.GEN_PRETRAIN_PATH, map_location=device, weights_only=False))
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
        
        # Create discriminator
        discriminator = create_discriminator(
            disc_type=disc_type,
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_rate,
            device=device
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        
        # Create log file
        log_file = os.path.join(base_output_dir, f'{i+1}_{disc_type}_bs{batch_size}_lr{learning_rate}_emb{embedding_dim}_hid{hidden_dim}_drop{dropout_rate}.log')
        
        # Train discriminator
        pretrain_discriminator(
            target_lstm=target_lstm,
            generator=generator,
            discriminator=discriminator,
            optimizer=optimizer,
            outer_epochs=outer_epochs,
            inner_epochs=inner_epochs,
            batch_size=batch_size,
            generated_num=config.GENERATED_NUM,
            log_file=log_file,
            device=device
        )

    print(f"\nAll experiments completed. Results saved to: {base_output_dir}")

if __name__ == "__main__":
    main()