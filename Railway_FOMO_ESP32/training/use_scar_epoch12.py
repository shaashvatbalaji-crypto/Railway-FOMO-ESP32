from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
)
from dataset import RailwayFOMOConverter


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = Path(
    "diagnostic_scar_binary/scar_model_epoch12.pth"
)

THRESHOLD = 0.90

BATCH_SIZE = 32

CLASS_NAME = "Scar"


# ============================================================
# MODEL
# ============================================================

class ScarBinaryCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        )

        self.head = torch.nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        x = self.features(x)

        # Convert feature map to the required FOMO grid.
        x = torch.nn.functional.interpolate(
            x,
            size=(GRID_SIZE, GRID_SIZE),
            mode="bilinear",
            align_corners=False
        )

        x = self.head(x)

        return x


# ============================================================
# DATASET
# ============================================================

class ScarValidationDataset(Dataset):

    def __init__(self, converter):

        self.converter = converter

    def __len__(self):

        return len(self.converter.samples)

    def __getitem__(self, idx):

        image_file, label_file = self.converter.samples[idx]

        image = (
            Image.open(image_file)
            .convert("RGB")
            .resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )
        )

        image_array = (
            np.asarray(image, dtype=np.float32) / 255.0
        )

        image_tensor = torch.from_numpy(
            np.transpose(image_array, (2, 0, 1))
        )

        # ----------------------------------------------------
        # Existing FOMO target
        # [GRID, GRID, NUM_CLASSES]
        # ----------------------------------------------------

        target = self.converter.encode_label(label_file)

        # Scar is class index 1 in your CLASS_MAP.
        scar_target = target[:, :, 1]

        scar_target = torch.from_numpy(
            scar_target.astype(np.float32)
        )

        return image_tensor, scar_target, image_file.name


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model):

    if not CHECKPOINT.exists():

        print()
        print("ERROR: Epoch-12 checkpoint was not found.")
        print()
        print("Expected:")
        print(CHECKPOINT)
        print()
        print(
            "Your training script may have overwritten the "
            "checkpoint."
        )
        print(
            "We must recover/save the Epoch-12 weights before "
            "using this model."
        )

        sys.exit(1)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    print()
    print("Checkpoint contents:")

    if isinstance(checkpoint, dict):

        for key in checkpoint.keys():
            print("  ", key)

    # --------------------------------------------------------
    # Normal checkpoint
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            # Sometimes the checkpoint itself is state_dict.
            try:
                model.load_state_dict(checkpoint)

            except Exception as e:

                print()
                print("ERROR loading checkpoint:")
                print(e)
                sys.exit(1)

    else:

        model.load_state_dict(checkpoint)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("=" * 60)
    print("SCAR EPOCH-12 MODEL EVALUATION")
    print("=" * 60)

    print()
    print("CUDA available :", torch.cuda.is_available())
    print("Device         :", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU            :",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version   :",
            torch.version.cuda
        )

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Threshold:", THRESHOLD)

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    converter = RailwayFOMOConverter(
        VALID_IMAGES,
        VALID_LABELS
    )

    print()
    print(
        "Validation samples loaded:",
        len(converter.samples)
    )

    dataset = ScarValidationDataset(converter)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ScarBinaryCNN().to(DEVICE)

    model = load_checkpoint(model)

    model.eval()

    print()
    print("✓ Epoch-12 Scar model loaded")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    probability_sum_positive = 0.0
    probability_sum_background = 0.0

    positive_count = 0
    background_count = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for images, targets, names in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            probabilities = probabilities.squeeze(1)

            predictions = (
                probabilities >= THRESHOLD
            )

            actual = targets >= 0.5

            TP += (
                predictions & actual
            ).sum().item()

            FP += (
                predictions & ~actual
            ).sum().item()

            FN += (
                ~predictions & actual
            ).sum().item()

            TN += (
                ~predictions & ~actual
            ).sum().item()

            # ------------------------------------------------
            # Probability statistics
            # ------------------------------------------------

            positive_probs = probabilities[actual]

            background_probs = probabilities[~actual]

            if positive_probs.numel() > 0:

                probability_sum_positive += (
                    positive_probs.sum().item()
                )

                positive_count += (
                    positive_probs.numel()
                )

            if background_probs.numel() > 0:

                probability_sum_background += (
                    background_probs.sum().item()
                )

                background_count += (
                    background_probs.numel()
                )

    # ========================================================
    # METRICS
    # ========================================================

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    positive_mean = (
        probability_sum_positive / positive_count
        if positive_count > 0
        else 0.0
    )

    background_mean = (
        probability_sum_background / background_count
        if background_count > 0
        else 0.0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("EPOCH-12 SCAR MODEL RESULTS")
    print("=" * 60)

    print()
    print("Threshold :", THRESHOLD)

    print()
    print("TP        :", TP)
    print("FP        :", FP)
    print("FN        :", FN)
    print("TN        :", TN)

    print()
    print("Precision :", f"{precision:.4f}")
    print("Recall    :", f"{recall:.4f}")
    print("F1        :", f"{f1:.4f}")

    print()
    print("Scar probability mean       :",
          f"{positive_mean:.4f}")

    print("Background probability mean:",
          f"{background_mean:.4f}")

    print()
    print("=" * 60)
    print("SCAR MODEL EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()