from pathlib import Path
import sys
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ============================================================
# FORCE GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available."
    )

DEVICE = torch.device("cuda")

print("=" * 60)
print("RAILWAY FOMO OBJECTNESS TINY TEST")
print("=" * 60)

print()
print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# IMPORTS
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from fomo_loss import FOMOLoss


# ============================================================
# SETTINGS
# ============================================================

PER_CLASS = 4
TOTAL_IMAGES = 16

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.003

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class TinyDataset(Dataset):

    def __init__(
        self,
        converter,
        per_class=4
    ):

        self.samples = []

        candidates = {
            c: []
            for c in range(NUM_CLASSES)
        }

        for image_path, label_path in converter.samples:

            target = converter.encode_label(
                label_path
            )

            # Classes are channels 1..4
            for c in range(NUM_CLASSES):

                if np.any(
                    target[:, :, c + 1] > 0
                ):

                    candidates[c].append(
                        (
                            image_path,
                            label_path
                        )
                    )

        selected = set()

        print()
        print(
            "Class candidates:"
        )

        for c in range(NUM_CLASSES):

            print(
                f"{CLASS_NAMES[c]:12s}: "
                f"{len(candidates[c])}"
            )

        print()
        print(
            "Selecting balanced images..."
        )

        for c in range(NUM_CLASSES):

            pool = candidates[c].copy()

            random.shuffle(pool)

            count = 0

            for sample in pool:

                if count >= per_class:
                    break

                image_path, label_path = sample

                key = str(label_path)

                if key in selected:
                    continue

                selected.add(key)

                self.samples.append(
                    sample
                )

                count += 1

        if len(self.samples) < TOTAL_IMAGES:

            raise RuntimeError(
                f"Only selected "
                f"{len(self.samples)} "
                f"images."
            )

        print(
            "Selected images:",
            len(self.samples)
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label_path = (
            self.samples[idx]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        image_array = (
            np.asarray(
                image,
                dtype=np.float32
            )
            /
            255.0
        )

        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        image_tensor = torch.from_numpy(
            image_array
        )

        converter = RailwayFOMOConverter(
            image_path.parent,
            label_path.parent
        )

        target = converter.encode_label(
            label_path
        )

        target_tensor = torch.from_numpy(
            target
        )

        return (
            image_tensor,
            target_tensor
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
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

    with torch.no_grad():

        for images, targets in loader:

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
            # Objectness
            # ------------------------------------------------

            objectness_prob = torch.sigmoid(
                outputs[:, 0]
            )

            predicted_object = (
                objectness_prob > 0.5
            )

            actual_object = (
                targets[:, :, :, 0] > 0.5
            )

            background_fp += (
                predicted_object
                &
                ~actual_object
            ).sum().item()

            # ------------------------------------------------
            # Class prediction
            # ------------------------------------------------

            class_logits = outputs[
                :,
                1:,
                :,
                :
            ]

            predicted_class = (
                class_logits.argmax(
                    dim=1
                )
            )

            actual_class = (
                targets[:, :, :, 1:]
                .argmax(dim=-1)
            )

            # ------------------------------------------------
            # Per class
            # ------------------------------------------------

            for c in range(NUM_CLASSES):

                pred_c = (
                    predicted_object
                    &
                    (
                        predicted_class
                        == c
                    )
                )

                actual_c = (
                    actual_object
                    &
                    (
                        actual_class
                        == c
                    )
                )

                tp[c] += (
                    pred_c
                    &
                    actual_c
                ).sum().item()

                fp[c] += (
                    pred_c
                    &
                    ~actual_c
                ).sum().item()

                fn[c] += (
                    actual_c
                    &
                    ~pred_c
                ).sum().item()

    f1 = np.zeros(
        NUM_CLASSES
    )

    for c in range(NUM_CLASSES):

        precision_den = (
            tp[c] + fp[c]
        )

        recall_den = (
            tp[c] + fn[c]
        )

        if precision_den > 0:

            precision = (
                tp[c]
                /
                precision_den
            )

        else:

            precision = 0.0

        if recall_den > 0:

            recall = (
                tp[c]
                /
                recall_den
            )

        else:

            recall = 0.0

        if precision + recall > 0:

            f1[c] = (
                2
                * precision
                * recall
                /
                (
                    precision
                    +
                    recall
                )
            )

    model.train()

    return (
        tp,
        fp,
        fn,
        f1,
        background_fp
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Output channels:",
        NUM_CLASSES + 1
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    converter = RailwayFOMOConverter(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    dataset = TinyDataset(
        converter,
        per_class=PER_CLASS
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FOMOModel(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print()
    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = FOMOLoss(
        num_classes=NUM_CLASSES,
        objectness_positive_weight=4.0,
        objectness_negative_weight=0.1
    ).to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 60)
    print("STARTING OBJECTNESS TRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        for images, targets in loader:

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

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        (
            tp,
            fp,
            fn,
            f1,
            background_fp
        ) = evaluate(
            model,
            loader
        )

        macro_f1 = f1.mean()

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | Loss: "
            f"{total_loss / len(loader):.6f}"
            f" | Macro F1: "
            f"{macro_f1:.4f}"
        )

        for c in range(NUM_CLASSES):

            print(
                f"   {CLASS_NAMES[c]:12s}"
                f" TP={tp[c]:3d}"
                f" FP={fp[c]:3d}"
                f" FN={fn[c]:3d}"
                f" F1={f1[c]:.3f}"
            )

        print(
            "   Background FP:",
            background_fp
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("OBJECTNESS TINY TEST COMPLETE")
    print("=" * 60)

    print()
    print(
        "Final Macro F1:",
        f"{macro_f1:.4f}"
    )

    print()
    print(
        "This result determines whether "
        "we proceed to full training."
    )


if __name__ == "__main__":

    main()