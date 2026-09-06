from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_IMAGES, TRAIN_LABELS, NUM_CLASSES, CLASS_NAMES
from dataset import RailwayFOMOConverter


def generate_statistics():
    print("=" * 60)
    print("FULL DATASET STATS & FORMULA GENERATOR")
    print("=" * 60)

    converter = RailwayFOMOConverter(TRAIN_IMAGES, TRAIN_LABELS)
    num_samples = len(converter.samples)
    print(f"Total training samples analyzed: {num_samples}")

    total_grid_cells = 0
    total_object_cells = 0
    total_background_cells = 0

    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    for _, label_path in converter.samples:
        target = converter.encode_label(label_path)  # [H, W, 5]
        h, w, c = target.shape
        total_grid_cells += (h * w)

        # Channel 0 is objectness
        obj_mask = target[:, :, 0] > 0
        num_objs = np.count_nonzero(obj_mask)
        total_object_cells += num_objs
        total_background_cells += ((h * w) - num_objs)

        # Count classes from channels 1-4
        active_indices = np.argwhere(obj_mask)
        for y, x in active_indices:
            cls_idx = np.argmax(target[y, x, 1:])
            class_counts[cls_idx] += 1

    # Calculate exact mathematical formulas
    bg_obj_ratio = total_background_cells / total_object_cells if total_object_cells > 0 else 1.0

    # Class inverse frequency weighting for class_weights
    max_class_count = max(class_counts.values()) if sum(class_counts.values()) > 0 else 1
    class_weights = []
    for i in range(NUM_CLASSES):
        count = class_counts[i]
        weight = max_class_count / count if count > 0 else 1.0
        class_weights.append(round(float(weight), 2))

    print("\n" + "=" * 60)
    print("EXACT COMPUTED FORMULAS FOR CONFIG / LOSS")
    print("=" * 60)
    print(f"Total Grid Cells       : {total_grid_cells}")
    print(f"Total Object Cells     : {total_object_cells}")
    print(f"Total Background Cells : {total_background_cells}")
    print(f"\n1. Exact Objectness pos_weight:")
    print(f"   pos_weight = {bg_obj_ratio:.4f}")
    
    print(f"\n2. Exact Class Weights (Inverse Frequency):")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"   - {name:10s} (Class {idx}): count = {class_counts[idx]}, weight = {class_weights[idx]}")

    print("\n" + "=" * 60)
    print("HOW TO REPLACE IN fomo_loss.py:")
    print("=" * 60)
    print(f"class_weights = {class_weights}")
    print(f"pos_weight = {bg_obj_ratio:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    generate_statistics()