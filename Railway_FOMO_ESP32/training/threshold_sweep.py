from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from config import (
    VALID_IMAGES,
    VALID_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
    NUM_CLASSES,
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = Path(
    "evaluation/training/best_fomo_model.pth"
)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = FOMOModel(
    num_classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# DATASET
# ============================================================

converter = RailwayFOMOConverter(
    VALID_IMAGES,
    VALID_LABELS
)


# ============================================================
# METRIC STORAGE
# ============================================================

results = {}

for threshold in THRESHOLDS:

    results[threshold] = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image_path):

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

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    array = np.transpose(
        array,
        (2, 0, 1)
    )

    return torch.from_numpy(
        array
    ).unsqueeze(0)


# ============================================================
# MATCH PREDICTIONS TO TARGETS
# ============================================================

def evaluate_threshold(
    predicted,
    target,
    threshold
):

    # --------------------------------------------------------
    # predicted:
    # [C,H,W] defect probabilities
    #
    # target:
    # [H,W,C]
    # --------------------------------------------------------

    predicted_class = (
        predicted.argmax(
            dim=0
        )
    )

    predicted_confidence = (
        predicted.max(
            dim=0
        ).values
    )

    predicted_positive = (
        predicted_confidence
        >= threshold
    )

    target_positive = (
        target.max(
            dim=-1
        ).values > 0
    )

    # --------------------------------------------------------
    # TP / FP / FN
    #
    # We evaluate spatial occupancy first.
    # Class correctness is measured separately below.
    # --------------------------------------------------------

    tp = (
        predicted_positive
        &
        target_positive
    ).sum().item()

    fp = (
        predicted_positive
        &
        ~target_positive
    ).sum().item()

    fn = (
        ~predicted_positive
        &
        target_positive
    ).sum().item()

    return tp, fp, fn


# ============================================================
# RUN VALIDATION
# ============================================================

print("=" * 70)
print("RAILWAY FOMO THRESHOLD CALIBRATION")
print("=" * 70)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print(
    "Validation samples:",
    len(converter.samples)
)

print()
print(
    "Model:",
    MODEL_PATH
)

print()
print("=" * 70)


# ============================================================
# EVALUATE EVERY VALIDATION IMAGE
# ============================================================

with torch.no_grad():

    for index, (
        image_path,
        label_path
    ) in enumerate(
        converter.samples
    ):

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        image_tensor = preprocess(
            image_path
        ).to(DEVICE)

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        output = model(
            image_tensor
        )[0]

        # ----------------------------------------------------
        # Implicit background
        #
        # Model:
        #   4 defect channels
        #
        # Loss:
        #   adds background channel
        # ----------------------------------------------------

        background = torch.zeros(
            1,
            GRID_SIZE,
            GRID_SIZE,
            device=DEVICE,
            dtype=output.dtype
        )

        logits = torch.cat(
            [
                background,
                output
            ],
            dim=0
        )

        probabilities = torch.softmax(
            logits,
            dim=0
        )

        # ----------------------------------------------------
        # Remove background
        #
        # probabilities[1:] =
        # four defect probabilities
        # ----------------------------------------------------

        defect_probabilities = (
            probabilities[1:]
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        target = torch.from_numpy(
            converter.encode_label(
                label_path
            )
        ).to(DEVICE)

        # ----------------------------------------------------
        # Evaluate each threshold
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            tp, fp, fn = evaluate_threshold(
                defect_probabilities,
                target,
                threshold
            )

            results[threshold]["tp"] += tp
            results[threshold]["fp"] += fp
            results[threshold]["fn"] += fn

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 100 == 0
            or
            index + 1 == len(
                converter.samples
            )
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(converter.samples)}"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

best_threshold = None
best_f1 = -1.0

for threshold in THRESHOLDS:

    tp = results[threshold]["tp"]
    fp = results[threshold]["fp"]
    fn = results[threshold]["fn"]

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

    if (
        precision + recall
        > 0
    ):

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:

        f1 = 0.0

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold

    print()
    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"TP        : {tp}"
    )

    print(
        f"FP        : {fp}"
    )

    print(
        f"FN        : {fn}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

print()
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print()
print(
    f"Best threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Best F1        : "
    f"{best_f1:.4f}"
)

print()
print("=" * 70)
print("THRESHOLD CALIBRATION COMPLETE")
print("=" * 70)