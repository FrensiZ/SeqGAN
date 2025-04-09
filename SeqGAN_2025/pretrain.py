import torch
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import your models
from oracle import TargetLSTM
from generator import Generator
from discriminator import Discriminator_CNN
from generator import pretrain_generator
from discriminator import pretrain_discriminator
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pretrain Generator and Discriminator for SeqGAN')
    parser.add_argument('--gen', action='store_true', help='Pretrain the generator')
    parser.add_argument('--dis', action='store_true', help='Pretrain the discriminator')
    parser.add_argument('--both', action='store_true', help='Pretrain both models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(config.SEED)
    
    # Initialize the oracle (target LSTM)
    target_lstm = TargetLSTM(
        config.VOCAB_SIZE, 
        config.EMB_DIM, 
        config.HIDDEN_DIM, 
        config.SEQ_LENGTH, 
        config.START_TOKEN, 
        device
    ).to(device)
    
    # Load oracle parameters
    target_lstm.load_params(config.TARGET_PARAMS_PATH)
    
    # Initialize the generator
    generator = Generator(
        config.VOCAB_SIZE, 
        config.EMB_DIM, 
        config.HIDDEN_DIM, 
        config.SEQ_LENGTH, 
        config.START_TOKEN, 
        device
    ).to(device)
    
    # Initialize the discriminator
    discriminator = Discriminator_CNN(
        config.VOCAB_SIZE, 
        config.DIS_EMB_DIM, 
        config.SEQ_LENGTH, 
        dropout_prob=config.DIS_DROPOUT, 
        device=device
    ).to(device)
    
    # Pretraining choice
    if args.both or args.gen:
        # Pretraining generator
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.GEN_LR)
        pretrain_generator(
            target_lstm=target_lstm,
            generator=generator,
            optimizer=g_optimizer,
            pre_epoch_num=config.PRE_EPOCH_NUM,
            batch_size=config.GEN_BATCH_SIZE,
            generated_num=config.GENERATED_NUM,
            eval_freq=config.EVAL_FREQ,
            lr_patience=config.GEN_LR_PATIENCE,
            lr_decay=config.GEN_LR_DECAY,
            log_path=config.GEN_PRETRAIN_LOG,
            device=device
        )
    
    if args.both or args.dis:
        # For discriminator pretraining, load the pretrained generator if it exists
        if args.dis and not args.gen and os.path.exists(config.GEN_PRETRAIN_PATH):
            print("Loading pretrained generator...")
            generator.load_state_dict(torch.load(config.GEN_PRETRAIN_PATH, map_location=device))
        
        # Pretraining discriminator
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.DIS_LR)
        pretrain_discriminator(
            target_lstm=target_lstm,
            generator=generator,
            discriminator=discriminator,
            optimizer=d_optimizer,
            outer_epochs=config.DIS_OUTER_EPOCHS,
            inner_epochs=config.DIS_INNER_EPOCHS,
            batch_size=config.DIS_BATCH_SIZE,
            generated_num=config.GENERATED_NUM,
            log_file=config.DIS_PRETRAIN_LOG,
            device=device
        )

if __name__ == "__main__":
    main()