import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS,
    CLASS_MAP,
)

from dataset import RailwayFOMOConverter


def verify_split(name, image_dir, label_dir):

    converter = RailwayFOMOConverter(
        image_dir,
        label_dir
    )

    stats = {
        "images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "cracks_cells": 0,
        "scars_cells": 0,
        "breaks_cells": 0,
        "lightbands_cells": 0,
        "cracks_images": 0,
        "scars_images": 0,
        "breaks_images": 0,
        "lightbands_images": 0,
        "max_cells": 0,
        "total_positive_cells": 0,
    }

    class_cell_counts = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    for _, label_file in converter.samples:

        target = converter.encode_label(
            label_file
        )

        stats["images"] += 1

        positive_cells = int(
            np.sum(target > 0)
        )

        stats["total_positive_cells"] += (
            positive_cells
        )

        stats["max_cells"] = max(
            stats["max_cells"],
            positive_cells
        )

        if positive_cells > 0:
            stats["positive_images"] += 1
        else:
            stats["negative_images"] += 1

        for cls in range(4):

            cells = int(
                np.sum(target[:, :, cls] > 0)
            )

            class_cell_counts[cls].append(
                cells
            )

            if cls == 0:
                stats["cracks_cells"] += cells

                if cells > 0:
                    stats["cracks_images"] += 1

            elif cls == 1:
                stats["scars_cells"] += cells

                if cells > 0:
                    stats["scars_images"] += 1

            elif cls == 2:
                stats["breaks_cells"] += cells

                if cells > 0:
                    stats["breaks_images"] += 1

            elif cls == 3:
                stats["lightbands_cells"] += cells

                if cells > 0:
                    stats["lightbands_images"] += 1

    print()
    print("=" * 60)
    print(f"{name.upper()} ADAPTIVE FOMO VERIFICATION")
    print("=" * 60)

    print()
    print("Images              :", stats["images"])
    print("Defect images       :", stats["positive_images"])
    print("No-defect images    :", stats["negative_images"])
    print("Total positive cells:", stats["total_positive_cells"])
    print("Maximum cells/image :", stats["max_cells"])

    print()
    print("Class image counts:")
    print("Cracks              :", stats["cracks_images"])
    print("Scars               :", stats["scars_images"])
    print("breaks              :", stats["breaks_images"])
    print("lightbands          :", stats["lightbands_images"])

    print()
    print("Total positive cells by class:")
    print("Cracks              :", stats["cracks_cells"])
    print("Scars               :", stats["scars_cells"])
    print("breaks              :", stats["breaks_cells"])
    print("lightbands          :", stats["lightbands_cells"])

    print()
    print("Average cells per annotation/image-class:")
    print(
        "Cracks              :",
        np.mean(class_cell_counts[0])
    )
    print(
        "Scars               :",
        np.mean(class_cell_counts[1])
    )
    print(
        "breaks              :",
        np.mean(class_cell_counts[2])
    )
    print(
        "lightbands          :",
        np.mean(class_cell_counts[3])
    )

    return converter, stats


def main():

    print("=" * 60)
    print("RAILWAY ADAPTIVE FOMO VERIFICATION")
    print("=" * 60)

    train, train_stats = verify_split(
        "train",
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    valid, valid_stats = verify_split(
        "valid",
        VALID_IMAGES,
        VALID_LABELS
    )

    print()
    print("=" * 60)
    print("FINAL CHECKS")
    print("=" * 60)

    checks = []

    # --------------------------------------------------------
    # Sample counts
    # --------------------------------------------------------

    check = train_stats["images"] == 560

    checks.append(check)

    print(
        "Train sample count (560):",
        "✓" if check else "✗"
    )

    check = valid_stats["images"] == 558

    checks.append(check)

    print(
        "Valid sample count (558):",
        "✓" if check else "✗"
    )

    # --------------------------------------------------------
    # All four classes
    # --------------------------------------------------------

    all_classes = all(
        train_stats[key] > 0
        for key in [
            "cracks_images",
            "scars_images",
            "breaks_images",
            "lightbands_images",
        ]
    )

    checks.append(all_classes)

    print(
        "All 4 FOMO classes present:",
        "✓" if all_classes else "✗"
    )

    # --------------------------------------------------------
    # Negative samples
    # --------------------------------------------------------

    negatives = (
        train_stats["negative_images"] > 0
    )

    checks.append(negatives)

    print(
        "Negative samples preserved:",
        "✓" if negatives else "✗"
    )

    # --------------------------------------------------------
    # Target shape
    # --------------------------------------------------------

    sample_target = train.encode_label(
        train.samples[0][1]
    )

    shape_ok = (
        sample_target.shape == (
            12,
            12,
            4
        )
    )

    checks.append(shape_ok)

    print(
        "Target shape (12,12,4):",
        "✓" if shape_ok else "✗"
    )

    # --------------------------------------------------------
    # No invalid values
    # --------------------------------------------------------

    values_ok = (
        np.all(
            (sample_target == 0)
            |
            (sample_target == 1)
        )
    )

    checks.append(values_ok)

    print(
        "Target values valid (0/1):",
        "✓" if values_ok else "✗"
    )

    print()
    print("=" * 60)

    if all(checks):
        print("✓ ADAPTIVE DATASET VERIFICATION PASSED")
    else:
        print("✗ ADAPTIVE DATASET VERIFICATION FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()