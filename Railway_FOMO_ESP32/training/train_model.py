from pathlib import Path
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_CLASSES,
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from fomo_loss import FOMOLoss


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 16
EPOCHS = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

OUTPUT_DIR = Path("evaluation/training")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL = (
    OUTPUT_DIR /
    "moderate_best_fomo_model.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class RailwayFOMODataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir
    ):

        self.converter = (
            RailwayFOMOConverter(
                image_dir,
                label_dir
            )
        )

        self.samples = (
            self.converter.samples
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label_path = (
            self.samples[index]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.from_numpy(
            image
        )

        # ----------------------------------------------------
        # FOMO target
        #
        # dataset.py returns:
        # (12, 12, 4)
        # ----------------------------------------------------

        target = (
            self.converter.encode_label(
                label_path
            )
        )

        target = torch.from_numpy(
            target
        )

        return image, target


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = RailwayFOMODataset(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

valid_dataset = RailwayFOMODataset(
    VALID_IMAGES,
    VALID_LABELS
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = FOMOModel(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = FOMOLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    samples = 0

    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            images
        )

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        # Prevent unstable gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        samples += batch_size

    return running_loss / samples


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    samples = 0

    with torch.no_grad():

        for images, targets in valid_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(
                images
            )

            loss = criterion(
                predictions,
                targets
            )

            batch_size = images.size(0)

            running_loss += (
                loss.item() *
                batch_size
            )

            samples += batch_size

    return running_loss / samples


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("RAILWAY ADAPTIVE FOMO TRAINING")
    print("=" * 70)

    print()
    print("Device          :", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version    :",
            torch.version.cuda
        )

    print()
    print("Training images :", len(train_dataset))
    print("Validation images:", len(valid_dataset))
    print("Image size      :", IMAGE_SIZE)
    print("Grid size       :", GRID_SIZE)
    print("Classes         :", NUM_CLASSES)
    print("Batch size      :", BATCH_SIZE)
    print("Epochs          :", EPOCHS)
    print("Learning rate   :", LEARNING_RATE)

    print()
    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print()
    print("Best model:")
    print(BEST_MODEL)

    print()
    print("=" * 70)

    best_validation_loss = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = (
            train_one_epoch()
        )

        validation_loss = (
            validate()
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train loss      : "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Learning rate   : "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save ONLY best model
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "validation_loss":
                    validation_loss,
                "train_loss":
                    train_loss,
                "image_size":
                    IMAGE_SIZE,
                "grid_size":
                    GRID_SIZE,
                "num_classes":
                    NUM_CLASSES,
            }

            torch.save(
                checkpoint,
                BEST_MODEL
            )

            print(
                "✓ New best model saved"
            )

    print()
    print("=" * 70)
    print("5-EPOCH TRAINING TRIAL COMPLETE")
    print("=" * 70)

    print()
    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}"
    )

    print()
    print(
        "Best model:",
        BEST_MODEL
    )


if __name__ == "__main__":
    main()