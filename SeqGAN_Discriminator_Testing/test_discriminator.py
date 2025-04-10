import torch
import random
import numpy as np
import os
import json
import pickle
from pathlib import Path

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
    # Get working directory
    base_dir = Path(os.getenv('OUTPUT_DIR', '.'))
    os.makedirs(base_dir, exist_ok=True)
    
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', '')
    if not config_path:
        raise ValueError("CONFIG_PATH environment variable not set")
    
    with open(config_path, 'r') as f:
        params = json.load(f)
    
    # Set seed for reproducibility
    seed = int(os.getenv('SEED', '0'))
    set_seed(seed)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract parameters from config
    disc_type = params.get('disc_type', 'cnn')
    batch_size = params.get('batch_size', 64)
    learning_rate = params.get('learning_rate', 1e-4)
    embedding_dim = params.get('embedding_dim', 64)
    hidden_dim = params.get('hidden_dim', 128)
    dropout_rate = params.get('dropout_rate', 0.1)
    outer_epochs = params.get('outer_epochs', 50)
    inner_epochs = params.get('inner_epochs', 3)
    
    print(f"Training {disc_type} discriminator with:")
    print(f"  Seed: {seed}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Embedding Dim: {embedding_dim}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Dropout Rate: {dropout_rate}")
    print(f"  Outer Epochs: {outer_epochs}")
    print(f"  Inner Epochs: {inner_epochs}")
    
    # Initialize oracle and generator
    target_lstm = TargetLSTM(
        config.VOCAB_SIZE, 
        config.EMB_DIM, 
        config.HIDDEN_DIM, 
        config.SEQ_LENGTH, 
        config.START_TOKEN, 
        device
    ).to(device)
    
    generator = Generator(
        config.VOCAB_SIZE, 
        config.EMB_DIM, 
        config.HIDDEN_DIM, 
        config.SEQ_LENGTH, 
        config.START_TOKEN, 
        device
    ).to(device)
    
    # Load oracle parameters
    print("Loading oracle parameters...")
    target_lstm.load_params(config.TARGET_PARAMS_PATH)
    
    # Load pretrained generator
    gen_path = os.getenv('GEN_PATH', '')
    if not gen_path:
        raise ValueError("GEN_PATH environment variable not set")
        
    print(f"Loading pretrained generator from {gen_path}...")
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    
    # Create discriminator
    discriminator = create_discriminator(
        disc_type=disc_type,
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout_rate,
        device=device
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    # Create log file
    log_file = base_dir / f"discriminator_training.log"
    
    # Store metrics for tracking progress
    metrics = {
        'loss': [],
        'accuracy': [],
        'real_prob': [],
        'fake_prob': []
    }
    
    # Initial evaluation
    initial_metrics = evaluate_discriminator(
        discriminator, 
        target_lstm, 
        generator, 
        num_samples=1000, 
        device=device
    )
    
    print(f"Initial metrics:")
    print(f"  Accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"  Real Probability: {initial_metrics['real_prob']:.4f}")
    print(f"  Fake Probability: {initial_metrics['fake_prob']:.4f}")
    
    # Store initial metrics
    for key in metrics:
        if key in initial_metrics:
            metrics[key].append(initial_metrics[key])
    
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
        device=device,
        metrics=metrics  # Pass metrics dictionary to store values
    )
    
    # Final evaluation
    final_metrics = evaluate_discriminator(
        discriminator, 
        target_lstm, 
        generator, 
        num_samples=2000, 
        device=device
    )
    
    print(f"\nFinal evaluation:")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Real Probability: {final_metrics['real_prob']:.4f}")
    print(f"  Fake Probability: {final_metrics['fake_prob']:.4f}")
    
    # Store results
    results = {
        'config': params,
        'seed': seed,
        'final_metrics': final_metrics,
        'metrics': metrics
    }
    
    # Save results
    results_path = base_dir / "discriminator_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save as JSON for easier inspection
    with open(base_dir / "discriminator_results.json", 'w') as f:
        # Convert numpy values to Python types
        json_results = {
            'config': params,
            'seed': seed,
            'final_metrics': {k: float(v) for k, v in final_metrics.items()},
            'metrics': {k: [float(val) for val in v] for k, v in metrics.items()}
        }
        json.dump(json_results, f, indent=2)
    
    # Save the trained discriminator
    torch.save(discriminator.state_dict(), base_dir / f"discriminator_{disc_type}.pth")
    
    print(f"Training completed! Results saved to {results_path}")

if __name__ == "__main__":
    main()