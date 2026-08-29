"""
Central configuration for DP-SGD training on MNIST.
"""

# System and Reproducibility
SEED = 42
DATA_DIR = "./data"

# Training Parameters
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 5
LR = 0.05
LEARNING_RATE = LR
MOMENTUM = 0.0  # Standard SGD without momentum for clean DP clipping baseline

# Differential Privacy Parameters (Opacus)
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.1
DELTA = 1e-5
