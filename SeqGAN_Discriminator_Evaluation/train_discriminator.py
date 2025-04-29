import torch
import os
from pathlib import Path
import time
import json

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
import config

def create_discriminator(disc_type, vocab_size, embedding_dim, hidden_dim, dropout, device):
    """
    Create the specified discriminator type.
    
    Args:
        disc_type: Type of discriminator ('simple', 'lstm', or 'cnn')
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden state
        dropout: Dropout rate
        device: PyTorch device
    
    Returns:
        Initialized discriminator model
    """
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
            sequence_length=config.SEQ_LENGTH,
            device=device,
            dropout_prob=dropout
        )
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")

def train_discriminator(disc_config, device, log_dir):
    """
    Train a discriminator model with the given configuration.
    
    Args:
        disc_config: Configuration for the discriminator and training
        device: PyTorch device to use for training
        log_dir: Directory to save logs and results
    
    Returns:
        Dictionary with training results and metrics
    """
    start_time = time.time()
    
    # Create log directory if needed
    if log_dir is None:
        log_dir = config.LOG_DIR / f"disc_{disc_config['disc_type']}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(disc_config, f, indent=2)
    
    # Create log file
    log_file = os.path.join(log_dir, "training.log")
    
    # Create Oracle
    oracle = Oracle(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.ORACLE_EMB_DIM,
        hidden_dim=config.ORACLE_HIDDEN_DIM,
        sequence_length=config.SEQ_LENGTH,
        start_token=config.START_TOKEN,
        device=device
    )
    
    # Load oracle parameters
    print("Loading oracle parameters...")
    oracle.load_params(config.TARGET_PARAMS_PATH)
    
    # Create Generator
    generator = Generator(
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=config.ORACLE_EMB_DIM,
        hidden_dim=config.ORACLE_HIDDEN_DIM,
        sequence_length=config.SEQ_LENGTH,
        start_token=config.START_TOKEN,
        device=device
    )
    
    # Load pretrained generator
    print(f"Loading pretrained generator from {config.GEN_PRETRAIN_PATH}...")
    generator.load_state_dict(torch.load(config.GEN_PRETRAIN_PATH, map_location=device))
    
    # Create discriminator
    discriminator = create_discriminator(
        disc_type=disc_config['disc_type'],
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=disc_config['embedding_dim'],
        hidden_dim=disc_config['hidden_dim'],
        dropout=disc_config['dropout_rate'],
        device=device
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_config['learning_rate'])
    
    # Print training configuration
    print(f"Training discriminator with:")
    print(f"  Type: {disc_config['disc_type']}")
    print(f"  Device: {device}")
    print(f"  Batch Size: {disc_config['batch_size']}")
    print(f"  Learning Rate: {disc_config['learning_rate']}")
    print(f"  Embedding Dim: {disc_config['embedding_dim']}")
    print(f"  Hidden Dim: {disc_config['hidden_dim']}")
    print(f"  Dropout Rate: {disc_config['dropout_rate']}")
    print(f"  Outer Epochs: {disc_config['outer_epochs']}")
    print(f"  Inner Epochs: {disc_config['inner_epochs']}")
    
    # Train discriminator
    pretrain_discriminator(
        target_lstm=oracle,
        generator=generator,
        discriminator=discriminator,
        optimizer=optimizer,
        outer_epochs=disc_config['outer_epochs'],
        inner_epochs=disc_config['inner_epochs'],
        batch_size=disc_config['batch_size'],
        generated_num=config.GENERATED_NUM,
        log_file=log_file,
        device=device
    )
    
    # Evaluate final model
    final_metrics = evaluate_discriminator(
        discriminator=discriminator,
        target_lstm=oracle,
        generator=generator,
        num_samples=1000,
        device=device
    )
    
    # Save discriminator model
    model_path = os.path.join(log_dir, "discriminator_model.pth")
    torch.save(discriminator.state_dict(), model_path)
    
    # Record training time
    training_time = time.time() - start_time
    
    # Create results summary
    results = {
        "config": disc_config,
        "training_time": training_time,
        "final_metrics": final_metrics,
        "model_path": model_path
    }
    
    # Save results
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed! Results saved to {log_dir}")
    print(f"Final metrics: {final_metrics}")
    
    return results

