"""
Central configuration for DP-SGD training and MIA evaluation across datasets (MNIST, CIFAR-10).

NOTE: Existing MNIST results in experiments/checkpoints/, experiments/splits/, and
experiments/results/ should be manually moved to experiments/mnist/ by the user.
"""

# Dataset Selection
DATASET = "cifar10"  # Supported: "mnist", "cifar10"

# Normalization Constants
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# System and Reproducibility
SEED = 42
DATA_DIR = "./data"

# Training Parameters (apply to both datasets)
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 50
LR = 0.05
LEARNING_RATE = LR
MOMENTUM = 0.0  # Standard SGD without momentum for clean DP clipping baseline

# Data Split (subsample to force memorization for MIA signal; applies to both datasets)
N_TRAIN = 5000

# Differential Privacy Parameters (Opacus)
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.1
DELTA = 1e-5

# Dataset-Scoped Output Paths
# Note: Existing MNIST artifacts in experiments/checkpoints/, experiments/splits/, and
# experiments/results/ should be manually moved to experiments/mnist/ by the user.
OUTPUT_DIR = f"experiments/{DATASET}"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
SPLITS_DIR = f"{OUTPUT_DIR}/splits"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
