# config/config.py

# Data
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Image settings
IMAGE_SIZE = (256, 256)
PATCH_GRID = (4, 4)
PATCH_SIZE = 64

# Difficulty mapping
DIFFICULTY = 'easy'
DIFFICULTY_MAP = {
    'easy': 2,
    'medium': 4,
    'difficult': 6
}

# Training
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# Loss
USE_SPATIAL_LOSS = True
SPATIAL_WEIGHT = 0.5

# Logging & Checkpoints
SAVE_DIR = "checkpoints"
