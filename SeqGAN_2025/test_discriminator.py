import torch
import argparse
import random
import numpy as np
import os
from datetime import datetime

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test different discriminator architectures')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='discriminator_tests', help='Output directory')
    
    # Discriminator configuration
    parser.add_argument('--disc_type', type=str, default='cnn', 
                        help='Discriminator type to test (simple, lstm, cnn)')
    
    # Training parameters
    parser.add_argument('--outer_epochs', type=int, default=config.DIS_OUTER_EPOCHS, 
                        help='Number of outer training epochs')
    parser.add_argument('--inner_epochs', type=int, default=config.DIS_INNER_EPOCHS, 
                        help='Number of inner training epochs')
    parser.add_argument('--batch_size', type=int, default=config.DIS_BATCH_SIZE, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=config.DIS_LR, 
                        help='Learning rate')
    
    # Hyperparameters
    parser.add_argument('--emb_dim', type=int, default=config.DIS_EMB_DIM, 
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=config.DIS_DROPOUT, 
                        help='Dropout rate')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp and discriminator type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{args.disc_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set random seed
    set_seed(args.seed)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
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
    
    # If generator pretrained exists, load it
    if os.path.exists(config.GEN_PRETRAIN_PATH):
        print("Loading pretrained generator...")
        generator.load_state_dict(torch.load(config.GEN_PRETRAIN_PATH, map_location=device))
    else:
        # If generator isn't available, we can't proceed with discriminator testing
        print("Pretrained generator not found. Please run generator pretraining first.")
        return
    
    # Create discriminator based on specified type
    print(f"\n===== Testing {args.disc_type} discriminator =====")
    discriminator = create_discriminator(
        disc_type=args.disc_type,
        vocab_size=config.VOCAB_SIZE,
        embedding_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    
    # Create log file
    log_file = os.path.join(output_dir, f'{args.disc_type}_discriminator.log')
    
    # Train discriminator using the existing function
    pretrain_discriminator(
        target_lstm=target_lstm,
        generator=generator,
        discriminator=discriminator,
        optimizer=optimizer,
        outer_epochs=args.outer_epochs,
        inner_epochs=args.inner_epochs,
        batch_size=args.batch_size,
        generated_num=config.GENERATED_NUM,
        log_file=log_file,
        device=device
    )
    
    # Save the trained discriminator
    model_path = os.path.join(output_dir, f'{args.disc_type}_discriminator.pth')
    torch.save(discriminator.state_dict(), model_path)
    print(f"\nTraining completed. Model saved to: {model_path}")
    print(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main()