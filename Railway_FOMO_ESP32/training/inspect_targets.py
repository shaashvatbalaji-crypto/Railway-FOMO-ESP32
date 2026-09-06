from pathlib import Path
import sys
import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    NUM_CLASSES,
    CLASS_NAMES,
    GRID_SIZE
)

from dataset import RailwayFOMOConverter


print("=" * 60)
print("RAILWAY FOMO TARGET DISTRIBUTION")
print("=" * 60)

converter = RailwayFOMOConverter(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

print()
print("Total training samples:", len(converter.samples))
print("Grid size:", GRID_SIZE)
print("Classes:", NUM_CLASSES)

# ------------------------------------------------------------
# Use the first 16 images from the training set
# ------------------------------------------------------------
samples = converter.samples

total_cells = 0
total_object_cells = 0

class_cells = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

object_cells_per_image = []

print()
print("=" * 60)
print("TARGET STATISTICS")
print("=" * 60)

for index, (image_path, label_path) in enumerate(samples):

    target = converter.encode_label(
        label_path
    )

    # Objectness channel
    objectness = target[:, :, 0]

    object_count = int(
        np.sum(objectness > 0.5)
    )

    total_object_cells += object_count

    image_cells = (
        GRID_SIZE * GRID_SIZE
    )

    total_cells += image_cells

    object_cells_per_image.append(
        object_count
    )

    # Class channels 1..4
    for cls in range(NUM_CLASSES):

        class_count = int(
            np.sum(
                target[:, :, cls + 1]
                > 0.5
            )
        )

        class_cells[cls] += class_count

    print(
        f"{index + 1:02d}. "
        f"{image_path.name[:45]:45s} "
        f"Object cells: {object_count}"
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

background_cells = (
    total_cells
    -
    total_object_cells
)

object_percentage = (
    100.0
    *
    total_object_cells
    /
    max(total_cells, 1)
)

background_percentage = (
    100.0
    *
    background_cells
    /
    max(total_cells, 1)
)

print()
print("=" * 60)
print("FINAL DISTRIBUTION")
print("=" * 60)

print()
print(
    "Total grid cells:",
    total_cells
)

print(
    "Object cells:",
    total_object_cells
)

print(
    "Background cells:",
    background_cells
)

print(
    "Object percentage:",
    f"{object_percentage:.4f}%"
)

print(
    "Background percentage:",
    f"{background_percentage:.4f}%"
)

print()
print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

for cls in range(NUM_CLASSES):

    percentage = (
        100.0
        *
        class_cells[cls]
        /
        max(total_object_cells, 1)
    )

    print(
        f"{CLASS_NAMES[cls]:12s}"
        f": {class_cells[cls]:4d} cells"
        f" ({percentage:.2f}%)"
    )

print()
print("=" * 60)
print("AVERAGE OBJECT CELLS / IMAGE")
print("=" * 60)

print(
    "Average:",
    f"{np.mean(object_cells_per_image):.2f}"
)

print(
    "Minimum:",
    min(object_cells_per_image)
)

print(
    "Maximum:",
    max(object_cells_per_image)
)

print()
print("=" * 60)
print("LOSS WEIGHT ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Calculate a mathematically balanced positive weight
# ------------------------------------------------------------

if total_object_cells > 0:

    balanced_positive_weight = (
        background_cells
        /
        total_object_cells
    )

    print()
    print(
        "Background / Object ratio:",
        f"{balanced_positive_weight:.4f}"
    )

    print()
    print(
        "Suggested positive objectness weight:",
        f"{balanced_positive_weight:.4f}"
    )

print()
print("=" * 60)
print("TARGET INSPECTION COMPLETE")
print("=" * 60)