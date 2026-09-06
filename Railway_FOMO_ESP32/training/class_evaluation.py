from pathlib import Path
import sys

import numpy as np
import torch
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
    "evaluation/training/moderate_best_fomo_model.pth"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Best spatial threshold found earlier
CONFIDENCE_THRESHOLD = 0.20

CLASS_NAMES = [
    "Cracks",
    "Scars",
    "breaks",
    "lightbands",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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

    return model


# ============================================================
# PREPROCESS IMAGE
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

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    return tensor


# ============================================================
# GET MODEL PROBABILITIES
# ============================================================

def get_probabilities(
    model,
    image
):

    image = image.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            image
        )[0]

    # --------------------------------------------------------
    # Add implicit background channel
    #
    # 0 = background
    # 1 = cracks
    # 2 = scars
    # 3 = breaks
    # 4 = lightbands
    # --------------------------------------------------------

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

    return probabilities


# ============================================================
# EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(
    probabilities,
    target
):

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_class = (
        probabilities.argmax(
            dim=0
        )
    )

    predicted_confidence = (
        probabilities.max(
            dim=0
        ).values
    )

    # --------------------------------------------------------
    # Ground-truth class
    # --------------------------------------------------------

    target_positive = (
        target.max(
            dim=-1
        ).values > 0
    )

    target_class = (
        target.argmax(
            dim=-1
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {

        "tp": [0] * NUM_CLASSES,
        "fp": [0] * NUM_CLASSES,
        "fn": [0] * NUM_CLASSES,

        "background_fp": 0,

        "class_confusion": [
            [0] * NUM_CLASSES
            for _ in range(NUM_CLASSES)
        ],
    }

    # --------------------------------------------------------
    # Examine every FOMO cell
    # --------------------------------------------------------

    for y in range(
        GRID_SIZE
    ):

        for x in range(
            GRID_SIZE
        ):

            predicted = (
                predicted_class[
                    y,
                    x
                ].item()
            )

            confidence = (
                predicted_confidence[
                    y,
                    x
                ].item()
            )

            actual_positive = (
                target_positive[
                    y,
                    x
                ].item()
            )

            # ------------------------------------------------
            # Background prediction
            # ------------------------------------------------

            if predicted == 0:

                # Missed real defect
                if actual_positive:

                    actual = (
                        target_class[
                            y,
                            x
                        ].item()
                    )

                    stats["fn"][actual] += 1

                continue

            # ------------------------------------------------
            # Defect prediction below threshold
            # ------------------------------------------------

            if confidence < CONFIDENCE_THRESHOLD:

                if actual_positive:

                    actual = (
                        target_class[
                            y,
                            x
                        ].item()
                    )

                    stats["fn"][actual] += 1

                continue

            # ------------------------------------------------
            # Convert model class to FOMO class
            #
            # Model:
            # 1 -> FOMO 0
            # 2 -> FOMO 1
            # 3 -> FOMO 2
            # 4 -> FOMO 3
            # ------------------------------------------------

            predicted_fomo = (
                predicted - 1
            )

            # ------------------------------------------------
            # False positive against background
            # ------------------------------------------------

            if not actual_positive:

                stats[
                    "background_fp"
                ] += 1

                stats[
                    "fp"
                ][predicted_fomo] += 1

                continue

            # ------------------------------------------------
            # Real defect
            # ------------------------------------------------

            actual_fomo = (
                target_class[
                    y,
                    x
                ].item()
            )

            # ------------------------------------------------
            # Correct class
            # ------------------------------------------------

            if predicted_fomo == actual_fomo:

                stats[
                    "tp"
                ][actual_fomo] += 1

            # ------------------------------------------------
            # Wrong class
            # ------------------------------------------------

            else:

                # Predicted class is FP
                stats[
                    "fp"
                ][predicted_fomo] += 1

                # Actual class is FN
                stats[
                    "fn"
                ][actual_fomo] += 1

                # Confusion matrix
                stats[
                    "class_confusion"
                ][actual_fomo][
                    predicted_fomo
                ] += 1

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAILWAY FOMO CLASS-AWARE EVALUATION")
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
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD
    )

    print(
        "Validation images:",
        "loading..."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    converter = RailwayFOMOConverter(
        VALID_IMAGES,
        VALID_LABELS
    )

    print(
        "Validation images:",
        len(converter.samples)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = load_model()

    print("✓ Model loaded")

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_tp = [0] * NUM_CLASSES
    total_fp = [0] * NUM_CLASSES
    total_fn = [0] * NUM_CLASSES

    total_background_fp = 0

    total_confusion = [
        [0] * NUM_CLASSES
        for _ in range(NUM_CLASSES)
    ]

    # --------------------------------------------------------
    # Evaluate validation set
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING VALIDATION DATASET")
    print("=" * 70)

    with torch.no_grad():

        for index, (
            image_path,
            label_path
        ) in enumerate(
            converter.samples
        ):

            image = preprocess(
                image_path
            )

            probabilities = (
                get_probabilities(
                    model,
                    image
                )
            )

            target_np = (
                converter.encode_label(
                    label_path
                )
            )

            target = torch.from_numpy(
                target_np
            ).to(DEVICE)

            stats = evaluate_image(
                probabilities,
                target
            )

            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            for cls in range(
                NUM_CLASSES
            ):

                total_tp[cls] += (
                    stats["tp"][cls]
                )

                total_fp[cls] += (
                    stats["fp"][cls]
                )

                total_fn[cls] += (
                    stats["fn"][cls]
                )

            total_background_fp += (
                stats["background_fp"]
            )

            for actual in range(
                NUM_CLASSES
            ):

                for predicted in range(
                    NUM_CLASSES
                ):

                    total_confusion[
                        actual
                    ][
                        predicted
                    ] += stats[
                        "class_confusion"
                    ][
                        actual
                    ][
                        predicted
                    ]

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (index + 1) % 100 == 0
                or
                index + 1
                == len(converter.samples)
            ):

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(converter.samples)}"
                )

    # ========================================================
    # CLASS METRICS
    # ========================================================

    print()
    print("=" * 70)
    print("CLASS-AWARE RESULTS")
    print("=" * 70)

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for cls in range(
        NUM_CLASSES
    ):

        tp = total_tp[cls]
        fp = total_fp[cls]
        fn = total_fn[cls]

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        if (
            precision + recall
            > 0
        ):

            f1 = (
                2
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        print()
        print(
            CLASS_NAMES[cls]
        )

        print(
            "  TP        :",
            tp
        )

        print(
            "  FP        :",
            fp
        )

        print(
            "  FN        :",
            fn
        )

        print(
            "  Precision :",
            f"{precision:.4f}"
        )

        print(
            "  Recall    :",
            f"{recall:.4f}"
        )

        print(
            "  F1        :",
            f"{f1:.4f}"
        )

    # --------------------------------------------------------
    # Macro average
    # --------------------------------------------------------

    macro_precision /= NUM_CLASSES
    macro_recall /= NUM_CLASSES
    macro_f1 /= NUM_CLASSES

    print()
    print("=" * 70)
    print("MACRO AVERAGE")
    print("=" * 70)

    print(
        "Precision :",
        f"{macro_precision:.4f}"
    )

    print(
        "Recall    :",
        f"{macro_recall:.4f}"
    )

    print(
        "F1        :",
        f"{macro_f1:.4f}"
    )

    # ========================================================
    # BACKGROUND FALSE POSITIVES
    # ========================================================

    print()
    print("=" * 70)
    print("BACKGROUND FALSE POSITIVES")
    print("=" * 70)

    print(
        "Background → defect cells:",
        total_background_fp
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("CLASS CONFUSION MATRIX")
    print("=" * 70)

    print()
    print(
        "Rows    = actual class"
    )

    print(
        "Columns = predicted class"
    )

    print()

    print(
        f"{'Actual':12s}",
        end=""
    )

    for name in CLASS_NAMES:

        print(
            f"{name:12s}",
            end=""
        )

    print()

    for actual in range(
        NUM_CLASSES
    ):

        print(
            f"{CLASS_NAMES[actual]:12s}",
            end=""
        )

        for predicted in range(
            NUM_CLASSES
        ):

            print(
                f"{total_confusion[actual][predicted]:12d}",
                end=""
            )

        print()

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("CLASS EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()