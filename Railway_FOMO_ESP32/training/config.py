from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "evaluation" / "results"


# ============================================================
# DATASET
# ============================================================

CLASS_NAMES = [
    "Cracks",
    "Scars",
    "breaks",
    "lightbands"
]

NUM_CLASSES = len(CLASS_NAMES)


# Original Roboflow → FOMO mapping
#
# 0 = Cracks       → 0
# 1 = Rails        → ignored
# 2 = Scars        → 1
# 3 = breaks       → 2
# 4 = lightbands   → 3

CLASS_MAP = {
    0: 0,
    2: 1,
    3: 2,
    4: 3
}


# ============================================================
# FOMO CONFIGURATION
# ============================================================

IMAGE_SIZE = 96

GRID_SIZE = 12

INPUT_CHANNELS = 3


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.001

NUM_WORKERS = 2

SEED = 42


# ============================================================
# OUTPUT
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY FOMO CONFIGURATION")
    print("=" * 60)

    print("Project      :", PROJECT_DIR)
    print("Train images :", TRAIN_IMAGES)
    print("Valid images :", VALID_IMAGES)

    print()
    print("Image size   :", IMAGE_SIZE)
    print("Grid size    :", GRID_SIZE)
    print("Classes      :", NUM_CLASSES)

    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i} -> {name}")

    print()
    print("Rails class  : IGNORED")