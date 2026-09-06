from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from tiny_overfit_test import TinyOverfitDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "diagnostic_overfit/overfit_test_model.pth"
)


def main():

    print("=" * 60)
    print("OVERFIT MODEL LOGIT INSPECTION")
    print("=" * 60)

    print()
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    converter = RailwayFOMOConverter(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    dataset = TinyOverfitDataset(
        converter,
        samples_per_class=2,
        max_samples=8
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = FOMOModel(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("✓ Model loaded")

    # --------------------------------------------------------
    # Get tiny dataset batch
    # --------------------------------------------------------

    images, targets = next(
        iter(loader)
    )

    images = images.to(DEVICE)
    targets = targets.to(DEVICE)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(images)

    print()
    print("Output shape:", tuple(outputs.shape))

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RAW LOGIT STATISTICS")
    print("=" * 60)

    print()

    for cls in range(NUM_CLASSES):

        values = outputs[:, cls, :, :]

        print(
            f"{CLASS_NAMES[cls]:12s} "
            f"min={values.min().item(): .4f} "
            f"max={values.max().item(): .4f} "
            f"mean={values.mean().item(): .4f}"
        )

    # --------------------------------------------------------
    # Target-positive vs background logits
    # --------------------------------------------------------

    target_positive = (
        targets.max(dim=-1).values > 0
    )

    target_class = (
        targets.argmax(dim=-1)
    )

    print()
    print("=" * 60)
    print("POSITIVE CELL LOGITS")
    print("=" * 60)

    positive_count = 0

    for cls in range(NUM_CLASSES):

        mask = (
            target_positive
            &
            (target_class == cls)
        )

        if mask.any():

            class_logits = outputs[
                :, cls, :, :
            ][mask]

            print()
            print(
                CLASS_NAMES[cls]
            )

            print(
                "  Positive cells:",
                int(mask.sum().item())
            )

            print(
                "  Correct-class logit mean:",
                f"{class_logits.mean().item():.4f}"
            )

            print(
                "  Correct-class logit max:",
                f"{class_logits.max().item():.4f}"
            )

            print(
                "  Correct-class logit min:",
                f"{class_logits.min().item():.4f}"
            )

            positive_count += (
                mask.sum().item()
            )

    # --------------------------------------------------------
    # Background statistics
    # --------------------------------------------------------

    background_mask = ~target_positive

    print()
    print("=" * 60)
    print("BACKGROUND LOGITS")
    print("=" * 60)

    if background_mask.any():

        background_values = outputs[
            :, :, :, :
        ].permute(
            0, 2, 3, 1
        )[background_mask]

        print()
        print(
            "Background cells:",
            int(background_mask.sum().item())
        )

        print(
            "Maximum defect logit:",
            f"{background_values.max().item():.4f}"
        )

        print(
            "Mean defect logit:",
            f"{background_values.mean().item():.4f}"
        )

        print(
            "Minimum defect logit:",
            f"{background_values.min().item():.4f}"
        )

    # --------------------------------------------------------
    # Positive cells: compare all four classes
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("POSITIVE CELL CLASS COMPETITION")
    print("=" * 60)

    for cls in range(NUM_CLASSES):

        mask = (
            target_positive
            &
            (target_class == cls)
        )

        if not mask.any():
            continue

        values = outputs.permute(
            0, 2, 3, 1
        )[mask]

        mean_logits = values.mean(
            dim=0
        )

        print()
        print(
            "Actual:",
            CLASS_NAMES[cls]
        )

        for predicted_cls in range(
            NUM_CLASSES
        ):

            print(
                f"  {CLASS_NAMES[predicted_cls]:12s}: "
                f"{mean_logits[predicted_cls].item(): .4f}"
            )

    # --------------------------------------------------------
    # Maximum class prediction
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("POSITIVE CELL PREDICTION CHECK")
    print("=" * 60)

    predicted_classes = outputs.argmax(
        dim=1
    )

    for cls in range(NUM_CLASSES):

        mask = (
            target_positive
            &
            (target_class == cls)
        )

        if not mask.any():
            continue

        predictions = predicted_classes[
            mask
        ]

        counts = torch.bincount(
            predictions,
            minlength=NUM_CLASSES
        )

        print()
        print(
            "Actual:",
            CLASS_NAMES[cls]
        )

        for predicted_cls in range(
            NUM_CLASSES
        ):

            print(
                f"  predicted "
                f"{CLASS_NAMES[predicted_cls]:12s}: "
                f"{int(counts[predicted_cls])}"
            )

    print()
    print("=" * 60)
    print("LOGIT INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()