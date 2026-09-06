from pathlib import Path
import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image, ImageEnhance


# ============================================================
# PROJECT PATH
# ============================================================

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
    CLASS_MAP
)

from scar_binary_v2 import DedicatedScarBinaryCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path(
    "diagnostic_scar_binary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# IMPORTANT:
# This is a NEW checkpoint.
# It does NOT overwrite either previous model.
CHECKPOINT_PATH = (
    OUTPUT_DIR /
    "scar_mild_augmented_best.pth"
)

BATCH_SIZE = 16
EPOCHS = 20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

THRESHOLDS = [
    0.50,
    0.70,
    0.80,
    0.90
]


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# DATASET
# ============================================================

class MildAugmentedScarDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir,
        augment=False
    ):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.augment = augment

        self.samples = []

        for label_file in sorted(
            self.label_dir.glob("*.txt")
        ):

            image_file = self.find_image(
                label_file.stem
            )

            if image_file is not None:

                self.samples.append(
                    (
                        image_file,
                        label_file
                    )
                )

    # ========================================================
    # FIND IMAGE
    # ========================================================

    def find_image(self, stem):

        for ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        ]:

            image_path = (
                self.image_dir /
                (stem + ext)
            )

            if image_path.exists():

                return image_path

        return None

    # ========================================================
    # READ SCAR LOCATIONS
    # ========================================================

    def read_scar_points(
        self,
        label_file
    ):

        points = []

        text = label_file.read_text().strip()

        if not text:

            return points

        for line in text.splitlines():

            values = line.split()

            if len(values) != 5:

                continue

            original_class = int(
                values[0]
            )

            fomo_class = CLASS_MAP.get(
                original_class,
                -1
            )

            # Scar class
            if fomo_class != 1:

                continue

            xc = np.clip(
                float(values[1]),
                0.0,
                1.0
            )

            yc = np.clip(
                float(values[2]),
                0.0,
                1.0
            )

            points.append(
                (xc, yc)
            )

        return points

    # ========================================================
    # CREATE 12 x 12 TARGET
    # ========================================================

    def make_target(
        self,
        points
    ):

        target = np.zeros(
            (
                GRID_SIZE,
                GRID_SIZE,
                1
            ),
            dtype=np.float32
        )

        for xc, yc in points:

            gx = min(
                int(xc * GRID_SIZE),
                GRID_SIZE - 1
            )

            gy = min(
                int(yc * GRID_SIZE),
                GRID_SIZE - 1
            )

            target[
                gy,
                gx,
                0
            ] = 1.0

        return target

    # ========================================================
    # MILD AUGMENTATION
    # ========================================================

    def augment_image_and_points(
        self,
        image,
        points
    ):

        # ----------------------------------------------------
        # 1. HORIZONTAL FLIP
        # ----------------------------------------------------
        #
        # This is safe because the Scar remains a Scar.
        # We MUST update the x coordinate.
        #

        if random.random() < 0.50:

            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

            points = [
                (
                    1.0 - xc,
                    yc
                )
                for xc, yc in points
            ]

        # ----------------------------------------------------
        # 2. MILD BRIGHTNESS
        # ----------------------------------------------------
        #
        # Previous experiment:
        # 0.75 - 1.25
        #
        # New experiment:
        # 0.90 - 1.10
        #

        if random.random() < 0.30:

            factor = random.uniform(
                0.90,
                1.10
            )

            image = (
                ImageEnhance.Brightness(
                    image
                ).enhance(factor)
            )

        # ----------------------------------------------------
        # 3. MILD CONTRAST
        # ----------------------------------------------------

        if random.random() < 0.30:

            factor = random.uniform(
                0.90,
                1.10
            )

            image = (
                ImageEnhance.Contrast(
                    image
                ).enhance(factor)
            )

        # ----------------------------------------------------
        # NO ROTATION
        # ----------------------------------------------------
        #
        # We deliberately removed rotation because the
        # previous experiment produced poor spatial matching.
        #

        return image, points

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        image_file, label_file = (
            self.samples[idx]
        )

        image = Image.open(
            image_file
        ).convert("RGB")

        # Read original Scar coordinates
        points = self.read_scar_points(
            label_file
        )

        # ----------------------------------------------------
        # Resize to CNN input
        # ----------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Apply mild augmentation
        # ----------------------------------------------------

        if self.augment:

            image, points = (
                self.augment_image_and_points(
                    image,
                    points
                )
            )

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        img_arr = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        img_tensor = torch.from_numpy(
            np.transpose(
                img_arr,
                (2, 0, 1)
            )
        )

        # ----------------------------------------------------
        # Create target
        # ----------------------------------------------------

        target = self.make_target(
            points
        )

        target_tensor = torch.from_numpy(
            target
        ).permute(
            2,
            0,
            1
        )

        return (
            img_tensor,
            target_tensor
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )


# ============================================================
# COMPUTE POSITIVE WEIGHT
# ============================================================

def compute_pos_weight(dataset):

    total_cells = 0
    scar_cells = 0

    for _, target in dataset:

        total_cells += (
            GRID_SIZE *
            GRID_SIZE
        )

        scar_cells += int(
            torch.sum(
                target > 0.5
            ).item()
        )

    background_cells = (
        total_cells -
        scar_cells
    )

    if scar_cells == 0:

        return 1.0

    pos_weight = (
        background_cells /
        scar_cells
    )

    print()
    print(
        "[DATA STATISTICS]"
    )

    print(
        f"Total cells      : "
        f"{total_cells}"
    )

    print(
        f"Scar cells       : "
        f"{scar_cells}"
    )

    print(
        f"Background cells : "
        f"{background_cells}"
    )

    print(
        f"pos_weight       : "
        f"{pos_weight:.4f}"
    )

    return float(
        pos_weight
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    results = {}

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            probs = torch.sigmoid(
                outputs
            )

            all_probs.append(
                probs.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

    all_probs = torch.cat(
        all_probs,
        dim=0
    )

    all_targets = torch.cat(
        all_targets,
        dim=0
    )

    # ========================================================
    # TEST EACH THRESHOLD
    # ========================================================

    for threshold in THRESHOLDS:

        preds = (
            all_probs >= threshold
        )

        targets = (
            all_targets >= 0.5
        )

        tp = torch.sum(
            preds & targets
        ).item()

        fp = torch.sum(
            preds & ~targets
        ).item()

        fn = torch.sum(
            ~preds & targets
        ).item()

        precision = (
            tp /
            (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp /
            (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[threshold] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return results


# ============================================================
# TRAIN
# ============================================================

def train():

    print("=" * 60)
    print("MILD AUGMENTED SCAR CNN TRAINING")
    print("=" * 60)

    print()
    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # TRAINING DATA
    # ========================================================

    train_dataset = (
        MildAugmentedScarDataset(
            TRAIN_IMAGES,
            TRAIN_LABELS,
            augment=True
        )
    )

    # ========================================================
    # VALIDATION DATA
    # IMPORTANT: NO AUGMENTATION
    # ========================================================

    valid_dataset = (
        MildAugmentedScarDataset(
            VALID_IMAGES,
            VALID_LABELS,
            augment=False
        )
    )

    print()
    print(
        f"Training samples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(valid_dataset)}"
    )

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    stats_dataset = (
        MildAugmentedScarDataset(
            TRAIN_IMAGES,
            TRAIN_LABELS,
            augment=False
        )
    )

    pos_weight = compute_pos_weight(
        stats_dataset
    )

    pos_weight_tensor = torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=DEVICE
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        DedicatedScarBinaryCNN()
        .to(DEVICE)
    )

    print()
    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_f1 = -1.0
    best_epoch = -1
    best_threshold = None

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print()
    print("=" * 60)
    print("STARTING MILD AUGMENTED TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_loss = 0.0

        for images, targets in train_loader:

            images = images.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

        avg_loss = (
            epoch_loss /
            len(train_loader)
        )

        scheduler.step(
            avg_loss
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        metrics = evaluate(
            model,
            valid_loader
        )

        epoch_best_threshold = max(
            THRESHOLDS,
            key=lambda t:
            metrics[t]["f1"]
        )

        epoch_best_f1 = metrics[
            epoch_best_threshold
        ]["f1"]

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | Loss: {avg_loss:.4f}"
        )

        for threshold in THRESHOLDS:

            m = metrics[
                threshold
            ]

            print(
                f"  -> Thresh "
                f"{threshold:.2f} "
                f"| TP={m['tp']} "
                f"FP={m['fp']} "
                f"FN={m['fn']} "
                f"| Prec={m['precision']:.4f} "
                f"Rec={m['recall']:.4f} "
                f"F1={m['f1']:.4f}"
            )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if epoch_best_f1 > best_f1:

            best_f1 = epoch_best_f1
            best_epoch = epoch
            best_threshold = (
                epoch_best_threshold
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "pos_weight":
                        pos_weight,

                    "best_f1":
                        best_f1,

                    "best_threshold":
                        best_threshold,

                    "epoch":
                        best_epoch,

                    "image_size":
                        IMAGE_SIZE,

                    "grid_size":
                        GRID_SIZE
                },
                CHECKPOINT_PATH
            )

            print()
            print(
                "   [✓] NEW BEST "
                "MILD AUGMENTED MODEL SAVED"
            )

            print(
                f"       Epoch     : "
                f"{best_epoch}"
            )

            print(
                f"       Threshold : "
                f"{best_threshold}"
            )

            print(
                f"       F1        : "
                f"{best_f1:.4f}"
            )

        else:

            print(
                f"   No improvement."
                f" Best remains "
                f"Epoch {best_epoch} "
                f"(F1={best_f1:.4f})"
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("MILD AUGMENTED TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"BEST EPOCH     : "
        f"{best_epoch}"
    )

    print(
        f"BEST THRESHOLD : "
        f"{best_threshold}"
    )

    print(
        f"BEST F1        : "
        f"{best_f1:.4f}"
    )

    print()
    print(
        "NEW CHECKPOINT:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()
    print("=" * 60)
    print("PREVIOUS MODELS WERE NOT OVERWRITTEN")
    print("=" * 60)

    print()
    print(
        "Original model:"
    )

    print(
        "diagnostic_scar_binary/"
        "best_scar_model.pth"
    )

    print()
    print(
        "Old aggressive augmentation:"
    )

    print(
        "diagnostic_scar_binary/"
        "scar_augmented_best.pth"
    )

    print()
    print(
        "New mild augmentation:"
    )

    print(
        "diagnostic_scar_binary/"
        "scar_mild_augmented_best.pth"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()