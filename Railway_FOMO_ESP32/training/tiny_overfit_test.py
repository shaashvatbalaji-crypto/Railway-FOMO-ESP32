from pathlib import Path
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from fomo_loss import FOMOLoss


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path(
    "diagnostic_overfit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Small dataset for memorization test
SAMPLES_PER_CLASS = 2

# Maximum number of images
MAX_SAMPLES = 8

# Use all 8 images in one batch
BATCH_SIZE = 8

# Enough for a diagnostic, but much faster
EPOCHS = 20

LEARNING_RATE = 0.005

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TINY OVERFIT DATASET
# ============================================================

class TinyOverfitDataset(Dataset):

    def __init__(
        self,
        converter,
        samples_per_class=2,
        max_samples=8
    ):

        self.samples = []

        # ----------------------------------------------------
        # Find candidate images for every class
        # ----------------------------------------------------

        candidates = {
            cls: []
            for cls in range(NUM_CLASSES)
        }

        for image_path, label_path in converter.samples:

            target = converter.encode_label(
                label_path
            )

            positive_positions = np.argwhere(
                target > 0
            )

            if len(positive_positions) == 0:
                continue

            active_classes = np.unique(
                positive_positions[:, 2]
            )

            for cls in active_classes:

                cls = int(cls)

                candidates[cls].append(
                    (
                        image_path,
                        label_path
                    )
                )

        # ----------------------------------------------------
        # Display available candidates
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("TINY DATASET SELECTION")
        print("=" * 60)

        print()

        for cls in range(NUM_CLASSES):

            print(
                f"{CLASS_NAMES[cls]:12s}: "
                f"{len(candidates[cls])} candidate images"
            )

        # ----------------------------------------------------
        # Select images
        #
        # We prioritize one/two examples from every class.
        # Duplicate images containing multiple classes are
        # allowed to satisfy multiple class requirements.
        # ----------------------------------------------------

        selected = []

        for cls in range(NUM_CLASSES):

            candidates_cls = candidates[cls].copy()

            random.shuffle(
                candidates_cls
            )

            count = 0

            for sample in candidates_cls:

                if count >= samples_per_class:
                    break

                if sample not in selected:

                    selected.append(
                        sample
                    )

                    count += 1

        # ----------------------------------------------------
        # If fewer than MAX_SAMPLES were selected, fill them
        # with additional defect images.
        # ----------------------------------------------------

        all_candidates = []

        for cls in range(NUM_CLASSES):

            all_candidates.extend(
                candidates[cls]
            )

        random.shuffle(
            all_candidates
        )

        for sample in all_candidates:

            if len(selected) >= max_samples:
                break

            if sample not in selected:

                selected.append(
                    sample
                )

        self.samples = selected[
            :max_samples
        ]

        # ----------------------------------------------------
        # Calculate actual class coverage
        # ----------------------------------------------------

        actual_counts = {
            cls: 0
            for cls in range(NUM_CLASSES)
        }

        for image_path, label_path in self.samples:

            target = converter.encode_label(
                label_path
            )

            positions = np.argwhere(
                target > 0
            )

            if len(positions) == 0:
                continue

            classes = np.unique(
                positions[:, 2]
            )

            for cls in classes:

                actual_counts[
                    int(cls)
                ] += 1

        # ----------------------------------------------------
        # Display selected dataset
        # ----------------------------------------------------

        print()
        print(
            "Selected images:",
            len(self.samples)
        )

        print()
        print(
            "Images containing each class:"
        )

        for cls in range(NUM_CLASSES):

            print(
                f"  {CLASS_NAMES[cls]:12s}: "
                f"{actual_counts[cls]}"
            )

        # ----------------------------------------------------
        # Make sure all classes exist
        # ----------------------------------------------------

        missing = [
            CLASS_NAMES[cls]
            for cls in range(NUM_CLASSES)
            if actual_counts[cls] == 0
        ]

        if missing:

            raise RuntimeError(
                "Missing classes in tiny dataset: "
                + ", ".join(missing)
            )

        print()
        print(
            "✓ All four classes represented"
        )

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(self):

        return len(
            self.samples
        )

    # --------------------------------------------------------
    # Dataset item
    # --------------------------------------------------------

    def __getitem__(self, index):

        image_path, label_path = (
            self.samples[index]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Convert to float
        # ----------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        image_tensor = torch.from_numpy(
            image_array
        )

        # ----------------------------------------------------
        # FOMO target
        # ----------------------------------------------------

        converter = RailwayFOMOConverter(
            image_path.parent,
            label_path.parent
        )

        target_array = converter.encode_label(
            label_path
        )

        target_tensor = torch.from_numpy(
            target_array
        )

        return (
            image_tensor,
            target_tensor
        )


# ============================================================
# FAST FINAL EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataloader
):

    model.eval()

    tp = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    fp = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    fn = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    background_fp = 0

    total_positive = 0

    # --------------------------------------------------------
    # No gradients during evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Add implicit background channel
            # ------------------------------------------------

            background_logits = torch.zeros(
                outputs.shape[0],
                1,
                outputs.shape[2],
                outputs.shape[3],
                device=DEVICE,
                dtype=outputs.dtype
            )

            logits = torch.cat(
                [
                    background_logits,
                    outputs
                ],
                dim=1
            )

            # ------------------------------------------------
            # Prediction
            #
            # 0 = background
            # 1..4 = FOMO classes
            # ------------------------------------------------

            predicted = logits.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            target_positive = (
                targets.max(
                    dim=-1
                ).values > 0
            )

            target_class = (
                targets.argmax(
                    dim=-1
                )
            )

            total_positive += (
                target_positive.sum().item()
            )

            # ------------------------------------------------
            # Vectorized evaluation
            # ------------------------------------------------

            for cls in range(NUM_CLASSES):

                predicted_cls = (
                    predicted
                    == cls + 1
                )

                actual_cls = (
                    target_positive
                    &
                    (target_class == cls)
                )

                tp[cls] += (
                    predicted_cls
                    &
                    actual_cls
                ).sum().item()

                fp[cls] += (
                    predicted_cls
                    &
                    ~target_positive
                ).sum().item()

                fn[cls] += (
                    actual_cls
                    &
                    ~predicted_cls
                ).sum().item()

            # ------------------------------------------------
            # Background false positives
            # ------------------------------------------------

            background_fp += (
                (predicted > 0)
                &
                ~target_positive
            ).sum().item()

    # ========================================================
    # METRICS
    # ========================================================

    precision = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    recall = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    f1 = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    for cls in range(NUM_CLASSES):

        if (
            tp[cls]
            +
            fp[cls]
        ) > 0:

            precision[cls] = (
                tp[cls]
                /
                (
                    tp[cls]
                    +
                    fp[cls]
                )
            )

        if (
            tp[cls]
            +
            fn[cls]
        ) > 0:

            recall[cls] = (
                tp[cls]
                /
                (
                    tp[cls]
                    +
                    fn[cls]
                )
            )

        if (
            precision[cls]
            +
            recall[cls]
        ) > 0:

            f1[cls] = (
                2
                *
                precision[cls]
                *
                recall[cls]
                /
                (
                    precision[cls]
                    +
                    recall[cls]
                )
            )

    macro_f1 = f1.mean()

    model.train()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "background_fp": background_fp,
        "total_positive": total_positive
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("RAILWAY FOMO TINY-SET OVERFIT TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print()
    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Grid size:",
        GRID_SIZE
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Samples per class:",
        SAMPLES_PER_CLASS
    )

    print(
        "Maximum samples:",
        MAX_SAMPLES
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    converter = RailwayFOMOConverter(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    dataset = TinyOverfitDataset(
        converter,
        samples_per_class=SAMPLES_PER_CLASS,
        max_samples=MAX_SAMPLES
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
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

    # ========================================================
    # LOSS
    # ========================================================

    criterion = FOMOLoss(
        num_classes=NUM_CLASSES
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
    print("STARTING FAST OVERFIT TEST")
    print("=" * 60)

    print()

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

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

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                targets
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        average_loss = (
            total_loss
            /
            len(dataloader)
        )

        # ----------------------------------------------------
        # Lightweight progress output
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:02d}/{EPOCHS} | "
                f"Loss: "
                f"{average_loss:.6f}"
            )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL TINY-SET EVALUATION")
    print("=" * 60)

    metrics = evaluate_model(
        model,
        dataloader
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print(
        "Total positive cells:",
        metrics["total_positive"]
    )

    print()

    for cls in range(NUM_CLASSES):

        print(
            CLASS_NAMES[cls]
        )

        print(
            "  TP        :",
            int(metrics["tp"][cls])
        )

        print(
            "  FP        :",
            int(metrics["fp"][cls])
        )

        print(
            "  FN        :",
            int(metrics["fn"][cls])
        )

        print(
            "  Precision :",
            f"{metrics['precision'][cls]:.4f}"
        )

        print(
            "  Recall    :",
            f"{metrics['recall'][cls]:.4f}"
        )

        print(
            "  F1        :",
            f"{metrics['f1'][cls]:.4f}"
        )

        print()

    print(
        "Macro F1:",
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        "Background false positives:",
        int(metrics["background_fp"])
    )

    # ========================================================
    # DIAGNOSTIC INTERPRETATION
    # ========================================================

    print()
    print("=" * 60)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 60)

    minimum_recall = metrics["recall"].min()

    if (
        metrics["macro_f1"] >= 0.80
        and minimum_recall >= 0.70
    ):

        print()
        print(
            "✓ TINY-SET MEMORIZATION SUCCESSFUL"
        )

        print()
        print(
            "The model can learn the tiny dataset."
        )

        print(
            "The core model + target + loss "
            "pipeline is capable of learning."
        )

        print()
        print(
            "The remaining problem is likely "
            "generalization, data diversity, "
            "resolution, augmentation, or "
            "hard-negative confusion."
        )

    else:

        print()
        print(
            "⚠ TINY-SET MEMORIZATION NOT YET SUCCESSFUL"
        )

        print()
        print(
            "The model has not reliably memorized "
            "all four classes."
        )

        print()
        print(
            "Investigate target encoding, "
            "loss formulation, preprocessing, "
            "class representation, or model capacity."
        )

    # ========================================================
    # SAVE DIAGNOSTIC CHECKPOINT
    # ========================================================

    checkpoint_path = (
        OUTPUT_DIR
        /
        "overfit_test_model.pth"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "macro_f1":
                float(
                    metrics["macro_f1"]
                ),

            "class_f1":
                metrics["f1"].tolist(),

            "background_fp":
                int(
                    metrics["background_fp"]
                )
        },
        checkpoint_path
    )

    print()
    print(
        "Diagnostic checkpoint:"
    )

    print(
        checkpoint_path
    )

    print()
    print("=" * 60)
    print("OVERFIT DIAGNOSTIC COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()