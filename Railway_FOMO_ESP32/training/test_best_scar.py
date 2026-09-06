from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    VALID_IMAGES,
    VALID_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
    CLASS_MAP
)

from scar_binary_v2 import DedicatedScarBinaryCNN


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "diagnostic_scar_binary/best_scar_model.pth"
)

THRESHOLD = 0.90
BATCH_SIZE = 32


# ============================================================
# DATASET
# ============================================================

class ScarTestDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.samples = []

        for label_file in sorted(
            self.label_dir.glob("*.txt")
        ):

            image_file = self.find_image(
                label_file.stem
            )

            if image_file is not None:
                self.samples.append(
                    (image_file, label_file)
                )

    def find_image(self, stem):

        for ext in [
            ".jpg", ".jpeg", ".png",
            ".JPG", ".JPEG", ".PNG"
        ]:

            path = self.image_dir / (stem + ext)

            if path.exists():
                return path

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label_path = self.samples[idx]

        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )
        )

        image_array = (
            np.asarray(
                image,
                dtype=np.float32
            ) / 255.0
        )

        image_tensor = torch.from_numpy(
            np.transpose(
                image_array,
                (2, 0, 1)
            )
        )

        # ----------------------------------------------------
        # Scar target
        # ----------------------------------------------------

        target = np.zeros(
            (GRID_SIZE, GRID_SIZE),
            dtype=np.float32
        )

        text = label_path.read_text().strip()

        if text:

            for line in text.splitlines():

                values = line.split()

                if len(values) != 5:
                    continue

                original_class = int(values[0])

                fomo_class = CLASS_MAP.get(
                    original_class,
                    -1
                )

                # Scar = FOMO class 1
                if fomo_class == 1:

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

                    gx = min(
                        int(xc * GRID_SIZE),
                        GRID_SIZE - 1
                    )

                    gy = min(
                        int(yc * GRID_SIZE),
                        GRID_SIZE - 1
                    )

                    target[gy, gx] = 1.0

        return (
            image_tensor,
            torch.from_numpy(target)
        )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("BEST SCAR MODEL TEST")
    print("=" * 60)

    print()
    print("CUDA available :", torch.cuda.is_available())
    print("Device         :", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU            :",
            torch.cuda.get_device_name(0)
        )

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Threshold:", THRESHOLD)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        print()
        print("ERROR: checkpoint not found.")
        print(CHECKPOINT)

        return

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    dataset = ScarTestDataset(
        VALID_IMAGES,
        VALID_LABELS
    )

    print()
    print(
        "Validation images:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = DedicatedScarBinaryCNN().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("✓ Model loaded successfully")

    if "epoch" in checkpoint:

        print(
            "Saved epoch     :",
            checkpoint["epoch"]
        )

    if "best_threshold" in checkpoint:

        print(
            "Saved threshold :",
            checkpoint["best_threshold"]
        )

    if "best_f1" in checkpoint:

        print(
            "Saved F1        :",
            checkpoint["best_f1"]
        )

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            ).squeeze(1)

            predictions = (
                probabilities >= THRESHOLD
            )

            actual = (
                targets >= 0.5
            )

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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL SCAR MODEL TEST RESULTS")
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
    print("=" * 60)
    print("SCAR MODEL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()