from pathlib import Path

# Project root is one level up from src directory
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train_solution_bounding_boxes (1).csv"
IMAGE_DIR = DATA_DIR / "training_images"
PROCESSED_DIR = DATA_DIR / "processed"

# Model configuration
#MODEL_NAME = "facebook/detr-resnet-50"
MODEL_NAME = "model/"
ID2LABEL = {0: "car"}
LABEL2ID = {"car": 0}
NUM_LABELS = len(ID2LABEL)
DTYPE = "bfloat16"

# Training configuration
TRAIN_SIZE = 0.9
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
EPOCHS = 50
