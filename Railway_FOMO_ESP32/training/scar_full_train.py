from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
# GPU CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path("diagnostic_scar_binary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "best_scar_model.pth"


# ============================================================
# DATASET
# ============================================================

class FullScarDataset(Dataset):

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
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        ]:

            img = self.image_dir / (stem + ext)

            if img.exists():

                return img

        return None

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, lbl_path = self.samples[idx]

        image = (
            Image.open(img_path)
            .convert("RGB")
            .resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )
        )

        img_arr = (
            np.asarray(
                image,
                dtype=np.float32
            ) / 255.0
        )

        img_tensor = torch.from_numpy(
            np.transpose(
                img_arr,
                (2, 0, 1)
            )
        )

        # ----------------------------------------------------
        # Binary Scar target
        # ----------------------------------------------------

        target = np.zeros(
            (GRID_SIZE, GRID_SIZE, 1),
            dtype=np.float32
        )

        text = lbl_path.read_text().strip()

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

                # Scar class
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

                    target[gy, gx, 0] = 1.0

        return (
            img_tensor,
            torch.from_numpy(
                target
            ).permute(2, 0, 1)
        )


# ============================================================
# POSITIVE WEIGHT
# ============================================================

def compute_dataset_pos_weight(dataset):

    total_cells = 0
    scar_cells = 0

    for _, target in dataset:

        total_cells += (
            GRID_SIZE * GRID_SIZE
        )

        scar_cells += torch.sum(
            target > 0.5
        ).item()

    background_cells = (
        total_cells - scar_cells
    )

    if scar_cells == 0:

        return 1.0

    pos_weight = (
        background_cells / scar_cells
    )

    print(
        f"[Full Data Stats] "
        f"Total cells: {total_cells} | "
        f"Scar cells: {scar_cells} | "
        f"Background cells: {background_cells}"
    )

    print(
        f"[Full Data Stats] "
        f"Computed Scar-specific pos_weight: "
        f"{pos_weight:.4f}"
    )

    return float(pos_weight)


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(model, val_loader):

    model.eval()

    thresholds = [
        0.30,
        0.50,
        0.70,
        0.90
    ]

    results = {}

    with torch.no_grad():

        for threshold in thresholds:

            tp = 0
            fp = 0
            fn = 0

            for images, targets in val_loader:

                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(images)

                probabilities = torch.sigmoid(
                    outputs
                )

                predictions = (
                    probabilities > threshold
                )

                actual = (
                    targets > 0.5
                )

                tp += (
                    predictions & actual
                ).sum().item()

                fp += (
                    predictions & ~actual
                ).sum().item()

                fn += (
                    ~predictions & actual
                ).sum().item()

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            f1 = (
                2 * precision * recall /
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
# TRAINING
# ============================================================

def train_full_scar_model():

    print("=" * 60)
    print("FULL DATASET SCAR BINARY TRAINING")
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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("LOADING DATASET")
    print("=" * 60)

    train_dataset = FullScarDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    valid_dataset = FullScarDataset(
        VALID_IMAGES,
        VALID_LABELS
    )

    print(
        f"Loaded {len(train_dataset)} "
        f"training samples and "
        f"{len(valid_dataset)} "
        f"validation samples."
    )

    # --------------------------------------------------------
    # Positive weight
    # --------------------------------------------------------

    pos_val = compute_dataset_pos_weight(
        train_dataset
    )

    pos_weight_tensor = torch.tensor(
        [pos_val],
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DedicatedScarBinaryCNN().to(
        DEVICE
    )

    print()
    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    epochs = 15

    # ========================================================
    # IMPORTANT:
    # GLOBAL BEST VALIDATION F1
    # ========================================================

    best_f1 = -1.0
    best_epoch = 0
    best_threshold = None

    print()
    print("=" * 60)
    print("STARTING FULL SCAR TRAINING")
    print("=" * 60)

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        epoch_loss = 0.0

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

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

            outputs = model(images)

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

            epoch_loss += loss.item()

        avg_loss = (
            epoch_loss /
            len(train_loader)
        )

        scheduler.step(avg_loss)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate_model(
            model,
            valid_loader
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {avg_loss:.4f}"
        )

        for threshold in [
            0.30,
            0.50,
            0.70,
            0.90
        ]:

            metrics = val_metrics[
                threshold
            ]

            print(
                f"  -> Thresh {threshold:.2f} | "
                f"TP={metrics['tp']} "
                f"FP={metrics['fp']} "
                f"FN={metrics['fn']} | "
                f"Prec={metrics['precision']:.4f} "
                f"Rec={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

        # ====================================================
        # FIND BEST THRESHOLD FOR THIS EPOCH
        # ====================================================

        epoch_best_threshold = max(
            val_metrics,
            key=lambda t:
            val_metrics[t]["f1"]
        )

        epoch_best_f1 = val_metrics[
            epoch_best_threshold
        ]["f1"]

        # ====================================================
        # GLOBAL BEST CHECKPOINT
        # ====================================================

        if epoch_best_f1 > best_f1:

            best_f1 = epoch_best_f1
            best_epoch = epoch
            best_threshold = (
                epoch_best_threshold
            )

            best_metrics = val_metrics[
                epoch_best_threshold
            ]

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "best_f1":
                    best_f1,

                "best_threshold":
                    best_threshold,

                "precision":
                    best_metrics["precision"],

                "recall":
                    best_metrics["recall"],

                "tp":
                    best_metrics["tp"],

                "fp":
                    best_metrics["fp"],

                "fn":
                    best_metrics["fn"],

                "pos_weight":
                    pos_val
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

            print()
            print(
                "   [✓] NEW BEST MODEL SAVED"
            )

            print(
                f"       Epoch     : {best_epoch}"
            )

            print(
                f"       Threshold : "
                f"{best_threshold:.2f}"
            )

            print(
                f"       F1        : "
                f"{best_f1:.4f}"
            )

        else:

            print()
            print(
                "   No improvement. "
                f"Best remains Epoch "
                f"{best_epoch} "
                f"(F1={best_f1:.4f})"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"BEST EPOCH     : {best_epoch}"
    )

    print(
        f"BEST THRESHOLD : "
        f"{best_threshold:.2f}"
    )

    print(
        f"BEST F1        : "
        f"{best_f1:.4f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()
    print(
        "The checkpoint contains the "
        "actual best validation model."
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_full_scar_model()