import torch
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Device configuration - will be overridden by CUDA_VISIBLE_DEVICES in parallel script
DEVICE = torch.device(os.getenv('CUDA_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))

# Data parameters
VOCAB_SIZE = 5000
SEQ_LENGTH = 20
START_TOKEN = 0
GENERATED_NUM = 10000  # Number of samples to generate for testing

# Model hyperparameters for GENERATOR / ORACLE
# These should be fixed as your generator is already pretrained
EMB_DIM = 32
HIDDEN_DIM = 32

# Paths for saving/loading models
ORACLE_PATH = os.path.join(SAVE_DIR, 'oracle.pth')
GEN_PRETRAIN_PATH = os.path.join(SAVE_DIR, 'generator_pretrained.pth')

# Target LSTM params file
TARGET_PARAMS_PATH = os.path.join(SAVE_DIR, 'target_params.pkl')