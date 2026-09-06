from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__.__path__[0]) if '__path__' in locals() else str(Path(__file__).resolve().parent)))

from config import VALID_IMAGES, VALID_LABELS, IMAGE_SIZE, GRID_SIZE, CLASS_MAP
from scar_binary_v2 import DedicatedScarBinaryCNN, ScarBinaryDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("diagnostic_scar_binary/scar_model.pth")


def load_scar_validation_data():
    image_dir = Path(VALID_IMAGES)
    label_dir = Path(VALID_LABELS)
    samples = []

    for label_file in sorted(label_dir.glob("*.txt")):
        image_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img = image_dir / (label_file.stem + ext)
            if img.exists():
                image_file = img
                break
        if image_file is not None:
            samples.append((image_file, label_file))

    return samples


def evaluate_thresholds():
    print("=" * 60)
    print("SCAR BINARY VALIDATION & THRESHOLD SCAN")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}. Run scar_binary_v2.py first.")
        return

    # Load model
    model = DedicatedScarBinaryCNN().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = load_scar_validation_data()
    print(f"Validation samples loaded: {len(samples)}")

    # Preload and encode validation targets to save time
    print("Parsing validation dataset targets...")
    dataset_records = []
    total_val_scars = 0

    for img_path, lbl_path in samples:
        image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1))).unsqueeze(0)

        target = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        text = lbl_path.read_text().strip()
        if text:
            for line in text.splitlines():
                values = line.split()
                if len(values) == 5:
                    original_class = int(values[0])
                    fomo_class = CLASS_MAP.get(original_class, -1)
                    if fomo_class == 1:  # Scar class ID
                        xc = np.clip(float(values[1]), 0.0, 1.0)
                        yc = np.clip(float(values[2]), 0.0, 1.0)
                        gx = min(int(xc * GRID_SIZE), GRID_SIZE - 1)
                        gy = min(int(yc * GRID_SIZE), GRID_SIZE - 1)
                        target[gy, gx] = 1.0

        total_val_scars += int(target.sum())
        dataset_records.append((img_tensor, target))

    print(f"Total actual Scar cells in validation set: {total_val_scars}\n")

    # Threshold scan list
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    best_f1 = -1.0
    best_thresh = 0.5

    print(f"{'Threshold':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 75)

    with torch.no_grad():
        for thresh in thresholds:
            tp, fp, fn = 0, 0, 0

            for img_tensor, target_grid in dataset_records:
                outputs = model(img_tensor.to(DEVICE))
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # [12, 12]
                preds = (probs > thresh).astype(np.float32)

                tp += int(np.sum((preds == 1) & (target_grid == 1)))
                fp += int(np.sum((preds == 1) & (target_grid == 0)))
                fn += int(np.sum((preds == 0) & (target_grid == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                f1_diff = f1 - best_f1
                if f1_diff > 0:
                    best_f1 = f1
                    best_thresh = thresh

            print(f"{thresh:<10.2f} | {tp:<6} | {fp:<6} | {fn:<6} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")

    print("-" * 75)
    print(f"\n[*] Optimal Threshold: {best_thresh} with Validation F1: {best_f1:.4f}")
    
    if best_f1 > 0.3:
        print("✅ SUCCESS: Scar model shows valid generalization capabilities!")
    else:
        print("⚠️ WARNING: F1 is low. Model may have memorized the 16 training samples and needs training on a larger subset.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_thresholds()