from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ============================================================
# IMPORTS
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_CLASSES,
    CLASS_NAMES
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from fomo_loss import FOMOLoss


# ============================================================
# GPU
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_CLASS = 4
MAX_SAMPLES = 16

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.003

OUTPUT_DIR = Path("diagnostic_focal_tiny")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)

print("CUDA available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    print("CUDA version    :", torch.version.cuda)

print()

print("=" * 60)
print("RAILWAY FOMO FOCAL-LOSS TINY TEST")
print("=" * 60)

print()
print("Device        :", DEVICE)
print("Image size    :", IMAGE_SIZE)
print("Grid size     :", GRID_SIZE)
print("Classes       :", NUM_CLASSES)
print("Samples/class :", SAMPLES_PER_CLASS)
print("Maximum imgs  :", MAX_SAMPLES)
print("Batch size    :", BATCH_SIZE)
print("Epochs        :", EPOCHS)
print("Learning rate :", LEARNING_RATE)


# ============================================================
# DATASET
# ============================================================

class BalancedTinyDataset(Dataset):

    def __init__(
        self,
        converter,
        samples_per_class=4,
        max_samples=16
    ):

        candidates = {
            cls: []
            for cls in range(NUM_CLASSES)
        }

        print()
        print("=" * 60)
        print("SEARCHING FOR CLASS-SPECIFIC IMAGES")
        print("=" * 60)

        # ----------------------------------------------------
        # Find images containing each class
        # ----------------------------------------------------

        for image_path, label_path in converter.samples:

            target = converter.encode_label(label_path)

            if target.shape[-1] != NUM_CLASSES + 1:
                raise RuntimeError(
                    f"Expected 5-channel target, "
                    f"got {target.shape}"
                )

            # Channels 1..4 are classes
            class_part = target[:, :, 1:]

            active_classes = np.unique(
                np.argwhere(class_part > 0)[:, 2]
            )

            for cls in active_classes:

                if cls < NUM_CLASSES:
                    candidates[cls].append(
                        (image_path, label_path)
                    )

        # ----------------------------------------------------
        # Candidate counts
        # ----------------------------------------------------

        for cls in range(NUM_CLASSES):

            print(
                f"{CLASS_NAMES[cls]:12s}: "
                f"{len(candidates[cls])} candidate images"
            )

        # ====================================================
        # SELECT UNIQUE IMAGES
        # ====================================================

        selected = []
        selected_paths = set()

        print()
        print("=" * 60)
        print("SELECTING BALANCED TINY DATASET")
        print("=" * 60)

        for cls in range(NUM_CLASSES):

            count = 0

            for image_path, label_path in candidates[cls]:

                key = str(label_path)

                if key in selected_paths:
                    continue

                selected.append(
                    (image_path, label_path)
                )

                selected_paths.add(key)

                count += 1

                if count >= samples_per_class:
                    break

            print(
                f"{CLASS_NAMES[cls]:12s}: "
                f"{count}/{samples_per_class} selected"
            )

        # ----------------------------------------------------
        # Verify at least one sample for every class
        # ----------------------------------------------------

        if len(selected) == 0:
            raise RuntimeError(
                "No samples were selected."
            )

        actual_counts = {
            cls: 0
            for cls in range(NUM_CLASSES)
        }

        for image_path, label_path in selected:

            target = converter.encode_label(
                label_path
            )

            class_part = target[:, :, 1:]

            active_classes = np.unique(
                np.argwhere(class_part > 0)[:, 2]
            )

            for cls in active_classes:

                if cls < NUM_CLASSES:
                    actual_counts[cls] += 1

        print()
        print("Actual class-containing image counts:")

        for cls in range(NUM_CLASSES):

            print(
                f"  {CLASS_NAMES[cls]:12s}: "
                f"{actual_counts[cls]}"
            )

        missing = [
            CLASS_NAMES[cls]
            for cls in range(NUM_CLASSES)
            if actual_counts[cls] == 0
        ]

        if missing:

            raise RuntimeError(
                "Missing classes: "
                + ", ".join(missing)
            )

        self.samples = selected[:max_samples]

        print()
        print(
            "Final selected images:",
            len(self.samples)
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):
        return len(self.samples)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path, label_path = self.samples[index]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

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
        # TARGET
        # ----------------------------------------------------

        converter = RailwayFOMOConverter(
            image_path.parent,
            label_path.parent
        )

        encoded = converter.encode_label(
            label_path
        )

        # Expected:
        # [12,12,5]
        #
        # channel 0 = objectness
        # channel 1 = Cracks
        # channel 2 = Scars
        # channel 3 = breaks
        # channel 4 = lightbands

        expected_shape = (
            GRID_SIZE,
            GRID_SIZE,
            NUM_CLASSES + 1
        )

        if encoded.shape != expected_shape:

            raise RuntimeError(
                f"Expected target {expected_shape}, "
                f"got {encoded.shape}"
            )

        objectness = encoded[:, :, 0]

        class_one_hot = encoded[:, :, 1:]

        class_index = np.argmax(
            class_one_hot,
            axis=2
        ).astype(np.float32)

        # ----------------------------------------------------
        # Loss target:
        #
        # [H,W,2]
        #
        # channel 0 = objectness
        # channel 1 = class index
        # ----------------------------------------------------

        target = np.zeros(
            (
                GRID_SIZE,
                GRID_SIZE,
                2
            ),
            dtype=np.float32
        )

        target[:, :, 0] = objectness
        target[:, :, 1] = class_index

        return (
            image_tensor,
            torch.from_numpy(target)
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    # --------------------------------------------------------
    # OBJECTNESS
    # --------------------------------------------------------

    objectness_probability = torch.sigmoid(
        predictions[:, 0]
    )

    predicted_object = (
        objectness_probability >= 0.5
    )

    # --------------------------------------------------------
    # CLASS
    # --------------------------------------------------------

    class_logits = predictions[:, 1:, :, :]

    predicted_class = class_logits.argmax(
        dim=1
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    actual_object = (
        targets[:, :, :, 0] > 0.5
    )

    actual_class = (
        targets[:, :, :, 1].long()
    )

    results = {}

    total_background_fp = 0

    for cls in range(NUM_CLASSES):

        predicted_cls = (
            predicted_object
            &
            (predicted_class == cls)
        )

        actual_cls = (
            actual_object
            &
            (actual_class == cls)
        )

        tp = (
            predicted_cls
            &
            actual_cls
        ).sum().item()

        fp = (
            predicted_cls
            &
            ~actual_cls
        ).sum().item()

        fn = (
            actual_cls
            &
            ~predicted_cls
        ).sum().item()

        background_fp = (
            predicted_cls
            &
            ~actual_object
        ).sum().item()

        total_background_fp += background_fp

        precision = (
            tp / max(tp + fp, 1)
        )

        recall = (
            tp / max(tp + fn, 1)
        )

        if precision + recall > 0:

            f1 = (
                2.0
                * precision
                * recall
                /
                (precision + recall)
            )

        else:

            f1 = 0.0

        results[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    macro_f1 = np.mean(
        [
            results[c]["f1"]
            for c in range(NUM_CLASSES)
        ]
    )

    return (
        results,
        macro_f1,
        total_background_fp
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    converter = RailwayFOMOConverter(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = BalancedTinyDataset(
        converter,
        samples_per_class=SAMPLES_PER_CLASS,
        max_samples=MAX_SAMPLES
    )

    if len(dataset) < 4:

        raise RuntimeError(
            "Tiny dataset is too small."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = FOMOModel(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Model parameters:",
        parameter_count
    )

    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        dummy = torch.randn(
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=DEVICE
        )

        dummy_output = model(dummy)

    expected_output = (
        1,
        NUM_CLASSES + 1,
        GRID_SIZE,
        GRID_SIZE
    )

    print(
        "Model output shape:",
        tuple(dummy_output.shape)
    )

    if tuple(dummy_output.shape) != expected_output:

        raise RuntimeError(
            f"Expected {expected_output}, "
            f"got {tuple(dummy_output.shape)}"
        )

    print(
        "✓ 5-channel model verified"
    )

    # ========================================================
    # FOCAL LOSS
    # ========================================================

    criterion = FOMOLoss(
        num_classes=NUM_CLASSES,

        class_weights=[
            3.26,
            1.00,
            2.62,
            1.29
        ],

        alpha=0.75,
        gamma=2.0,

        class_loss_weight=1.0
    ).to(DEVICE)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 60)
    print("STARTING FOCAL-LOSS TINY TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for images, targets in dataloader:

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

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss
            /
            max(len(dataloader), 1)
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():

            for images, targets in dataloader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                predictions = model(
                    images
                )

                all_predictions.append(
                    predictions
                )

                all_targets.append(
                    targets.to(DEVICE)
                )

        predictions = torch.cat(
            all_predictions,
            dim=0
        )

        targets = torch.cat(
            all_targets,
            dim=0
        )

        (
            metrics,
            macro_f1,
            background_fp
        ) = calculate_metrics(
            predictions,
            targets
        )

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | Loss: {average_loss:.6f}"
            f" | Macro F1: {macro_f1:.4f}"
        )

        for cls in range(NUM_CLASSES):

            m = metrics[cls]

            print(
                f"   {CLASS_NAMES[cls]:12s}"
                f" TP={m['tp']:3d}"
                f" FP={m['fp']:4d}"
                f" FN={m['fn']:3d}"
                f" F1={m['f1']:.4f}"
            )

        print(
            "   Background FP:",
            background_fp
        )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL FOCAL-LOSS TINY TEST")
    print("=" * 60)

    model.eval()

    with torch.no_grad():

        all_predictions = []
        all_targets = []

        for images, targets in dataloader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(
                images
            )

            all_predictions.append(
                predictions
            )

            all_targets.append(
                targets.to(DEVICE)
            )

        predictions = torch.cat(
            all_predictions,
            dim=0
        )

        targets = torch.cat(
            all_targets,
            dim=0
        )

    (
        metrics,
        macro_f1,
        background_fp
    ) = calculate_metrics(
        predictions,
        targets
    )

    # --------------------------------------------------------
    # Final class metrics
    # --------------------------------------------------------

    for cls in range(NUM_CLASSES):

        m = metrics[cls]

        print()
        print(
            CLASS_NAMES[cls]
        )

        print(
            "  TP        :",
            m["tp"]
        )

        print(
            "  FP        :",
            m["fp"]
        )

        print(
            "  FN        :",
            m["fn"]
        )

        print(
            "  Precision :",
            f"{m['precision']:.4f}"
        )

        print(
            "  Recall    :",
            f"{m['recall']:.4f}"
        )

        print(
            "  F1        :",
            f"{m['f1']:.4f}"
        )

    print()
    print(
        "Macro F1:",
        f"{macro_f1:.4f}"
    )

    print(
        "Background false positives:",
        background_fp
    )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    class_f1_dict = {}

    for cls in range(NUM_CLASSES):

        class_f1_dict[
            CLASS_NAMES[cls]
        ] = float(
            metrics[cls]["f1"]
        )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "macro_f1":
            float(macro_f1),

        "background_fp":
            int(background_fp),

        "class_f1":
            class_f1_dict,

        "class_weights":
            [
                3.26,
                1.00,
                2.62,
                1.29
            ],

        "focal_alpha":
            0.75,

        "focal_gamma":
            2.0
    }

    checkpoint_path = (
        OUTPUT_DIR
        /
        "focal_tiny_test_model.pth"
    )

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print()
    print(
        "Checkpoint saved:"
    )

    print(
        checkpoint_path
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    learned_classes = sum(
        1
        for cls in range(NUM_CLASSES)
        if metrics[cls]["tp"] > 0
    )

    print()
    print("=" * 60)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 60)

    print()
    print(
        f"Classes with TP > 0: "
        f"{learned_classes}/{NUM_CLASSES}"
    )

    if learned_classes == NUM_CLASSES:

        print()
        print(
            "✓ ALL FOUR CLASSES SHOW "
            "CORRECT PREDICTIONS"
        )

    elif learned_classes > 1:

        print()
        print(
            "⚠ MULTIPLE CLASSES ARE LEARNING"
        )

    else:

        print()
        print(
            "✗ MODEL IS STILL COLLAPSING"
        )

    print()
    print(
        "Do NOT start full 560-image training "
        "until this diagnostic is satisfactory."
    )

    print()
    print("=" * 60)
    print("FOCAL-LOSS TINY TEST COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()