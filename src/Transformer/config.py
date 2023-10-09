# Decided to use launch configurator instead

# Hyperparameters
# MODEL = "vtnhcpf"
# DATASET = "handcrop_poseflow"
MODEL = "vtnhc"
DATASET = "handcrop"
# MODEL = "vtnfb"
# DATASET = "fullbody"
LEARNING_RATE = 1e-4
GRADIENT_CLIP_VAL = 1
CNN = "rn34"
NUM_LAYERS = 4
NUM_HEADS = 8
BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 8

# Dataset
NUM_WORKERS = 4
SEQUENCE_LENGTH = 16
TEMPORAL_STRIDE = 2
LOG_DIR = "\\results"
DATA_DIR = "..\\..\\Data\\ELAR\\avi"

# Compute Related
GPUS = 1