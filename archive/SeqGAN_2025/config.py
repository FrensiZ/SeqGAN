import torch
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seeds for reproducibility
SEED = 88

# Data parameters
VOCAB_SIZE = 5000
SEQ_LENGTH = 20
START_TOKEN = 0
GENERATED_NUM = 10000

# Model hyperparameters GENERATOR / ORACLE
EMB_DIM = 32
HIDDEN_DIM = 32

# Generator training parameters
GEN_BATCH_SIZE = 64
GEN_LR = 6e-3
PRE_EPOCH_NUM = 120
GEN_LR_DECAY = 0.5
GEN_LR_PATIENCE = 5
EVAL_FREQ = 5

# Discriminator hyperparameters
DIS_EMB_DIM = 64
DIS_DROPOUT = 0.1
DIS_BATCH_SIZE = 64
DIS_LR = 1e-4

# Discriminator pre-training
DIS_OUTER_EPOCHS = 50
DIS_INNER_EPOCHS = 3

# Adversarial training parameters
ADV_TOTAL_EPOCHS = 200
ADV_G_STEPS = 1             # Generator updates per iteration
ADV_D_STEPS = 5             # Discriminator update iterations
ADV_D_EPOCHS = 3            # Discriminator epochs per update
ROLLOUT_NUM = 16            # Number of rollouts for reward estimation
ROLLOUT_UPDATE_RATE = 0.8   # Rate for updating rollout parameters

# Paths for saving models
ORACLE_PATH = os.path.join(SAVE_DIR, 'oracle.pth')
GEN_PRETRAIN_PATH = os.path.join(SAVE_DIR, 'generator_pretrained.pth')
DIS_PRETRAIN_PATH = os.path.join(SAVE_DIR, 'discriminator_pretrained.pth')

# Log files
GEN_PRETRAIN_LOG = os.path.join(LOG_DIR, 'generator_pretrain.txt')
DIS_PRETRAIN_LOG = os.path.join(LOG_DIR, 'discriminator_pretrain.txt')
ADV_TRAIN_LOG = os.path.join(LOG_DIR, 'adversarial_train.txt')
REWARD_LOG = os.path.join(LOG_DIR, 'rewards.txt')

# Target LSTM params file
TARGET_PARAMS_PATH = 'target_params.pkl'  # Assuming you're using the same file as in your implementation