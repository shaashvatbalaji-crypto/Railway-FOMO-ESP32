from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


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
    "evaluation/training/best_fomo_model.pth"
)

OUTPUT_DIR = Path(
    "evaluation/trained_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Confidence threshold
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50

# ------------------------------------------------------------
# Number of validation images to test
# ------------------------------------------------------------

NUM_SAMPLES = 10

# ------------------------------------------------------------
# Class names
#
# These correspond to FOMO classes:
#
# 0 = Cracks
# 1 = Scars
# 2 = breaks
# 3 = lightbands
# ------------------------------------------------------------

CLASS_NAMES = [
    "Cracks",
    "Scars",
    "breaks",
    "lightbands",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    # --------------------------------------------------------
    # Load RGB image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Resize to model input
    # --------------------------------------------------------

    resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Convert:
    #
    # H,W,C
    #     ↓
    # C,H,W
    # --------------------------------------------------------

    array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    array = np.transpose(
        array,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    return resized, tensor


# ============================================================
# GROUND-TRUTH CELLS
# ============================================================

def get_ground_truth_cells(
    converter,
    label_path
):

    target = converter.encode_label(
        label_path
    )

    cells = []

    positions = np.argwhere(
        target > 0
    )

    for y, x, cls in positions:

        cells.append(
            (
                int(x),
                int(y),
                int(cls)
            )
        )

    return cells


# ============================================================
# DRAW GRID
# ============================================================

def draw_prediction(
    image,
    cells,
    title
):

    # --------------------------------------------------------
    # Enlarge for visualization
    # --------------------------------------------------------

    canvas = image.resize(
        (
            384,
            384
        ),
        Image.Resampling.NEAREST
    ).copy()

    draw = ImageDraw.Draw(
        canvas
    )

    cell_size = (
        384 // GRID_SIZE
    )

    # --------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------

    for i in range(
        GRID_SIZE + 1
    ):

        position = (
            i * cell_size
        )

        draw.line(
            (
                position,
                0,
                position,
                384
            ),
            fill="white",
            width=1
        )

        draw.line(
            (
                0,
                position,
                384,
                position
            ),
            fill="white",
            width=1
        )

    # --------------------------------------------------------
    # Draw positive cells
    # --------------------------------------------------------

    for x, y, cls in cells:

        left = (
            x * cell_size
        )

        top = (
            y * cell_size
        )

        right = (
            left + cell_size
        )

        bottom = (
            top + cell_size
        )

        # ----------------------------------------------------
        # Red bounding cell
        # ----------------------------------------------------

        draw.rectangle(
            (
                left + 2,
                top + 2,
                right - 2,
                bottom - 2
            ),
            outline="red",
            width=4
        )

        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        if (
            0 <= cls
            < len(CLASS_NAMES)
        ):

            name = CLASS_NAMES[cls]

        else:

            name = str(cls)

        draw.text(
            (
                left + 3,
                top + 3
            ),
            name,
            fill="yellow"
        )

    # --------------------------------------------------------
    # Title bar
    # --------------------------------------------------------

    draw.rectangle(
        (
            0,
            0,
            384,
            22
        ),
        fill="black"
    )

    draw.text(
        (
            5,
            5
        ),
        title,
        fill="white"
    )

    return canvas


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model():

    print()
    print("Loading trained model...")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Create same architecture
    # --------------------------------------------------------

    model = FOMOModel(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print()
    print("✓ Trained model loaded")

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "train_loss" in checkpoint:

        print(
            "Checkpoint train loss:",
            f"{checkpoint['train_loss']:.6f}"
        )

    if "validation_loss" in checkpoint:

        print(
            "Checkpoint validation loss:",
            f"{checkpoint['validation_loss']:.6f}"
        )

    return model


# ============================================================
# RUN PREDICTION
# ============================================================

def predict(
    model,
    tensor
):

    tensor = tensor.to(
        DEVICE,
        non_blocking=True
    )

    with torch.no_grad():

        output = model(
            tensor
        )

    # --------------------------------------------------------
    # Model output:
    #
    # [1, 4, 12, 12]
    #
    # Remove batch dimension:
    #
    # [4, 12, 12]
    # --------------------------------------------------------

    prediction = output[0]

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The model has 4 defect channels:
    #
    # 0 = Cracks
    # 1 = Scars
    # 2 = breaks
    # 3 = lightbands
    #
    # FOMOLoss adds an implicit background logit = 0.
    #
    # Therefore inference MUST do the same.
    # --------------------------------------------------------

    background_logit = torch.zeros(
        (
            1,
            GRID_SIZE,
            GRID_SIZE
        ),
        device=prediction.device,
        dtype=prediction.dtype
    )

    logits = torch.cat(
        [
            background_logit,
            prediction
        ],
        dim=0
    )

    # --------------------------------------------------------
    # 5-class probabilities:
    #
    # 0 = background
    # 1 = Cracks
    # 2 = Scars
    # 3 = breaks
    # 4 = lightbands
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=0
    )

    predicted_class = (
        probabilities.argmax(
            dim=0
        )
    )

    confidence = (
        probabilities.max(
            dim=0
        ).values
    )

    # --------------------------------------------------------
    # Collect positive defect cells
    # --------------------------------------------------------

    predicted_cells = []

    for y in range(
        GRID_SIZE
    ):

        for x in range(
            GRID_SIZE
        ):

            class_index = (
                predicted_class[
                    y,
                    x
                ].item()
            )

            cell_confidence = (
                confidence[
                    y,
                    x
                ].item()
            )

            # ------------------------------------------------
            # Background
            # ------------------------------------------------

            if class_index == 0:
                continue

            # ------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------

            if (
                cell_confidence
                < CONFIDENCE_THRESHOLD
            ):
                continue

            # ------------------------------------------------
            # Convert:
            #
            # model class 1 → FOMO class 0
            # model class 2 → FOMO class 1
            # model class 3 → FOMO class 2
            # model class 4 → FOMO class 3
            # ------------------------------------------------

            fomo_class = (
                class_index - 1
            )

            predicted_cells.append(
                (
                    x,
                    y,
                    fomo_class
                )
            )

    return (
        predicted_cells,
        probabilities
    )


# ============================================================
# PRINT PREDICTIONS
# ============================================================

def print_prediction_summary(
    predicted_cells,
    probabilities
):

    print()
    print(
        "Predicted positive cells:",
        len(predicted_cells)
    )

    if len(predicted_cells) == 0:

        print(
            "  No defect cells detected."
        )

        return

    # --------------------------------------------------------
    # Count classes
    # --------------------------------------------------------

    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    for x, y, cls in predicted_cells:

        counts[cls] += 1

    print()
    print("Predicted class counts:")

    for cls in range(
        NUM_CLASSES
    ):

        print(
            f"  {CLASS_NAMES[cls]:12s}: "
            f"{counts[cls]}"
        )

    # --------------------------------------------------------
    # Print cells
    # --------------------------------------------------------

    print()
    print("Cells:")

    for x, y, cls in predicted_cells:

        # Probability of this defect class
        probability = probabilities[
            cls + 1,
            y,
            x
        ].item()

        print(
            f"  grid=({x},{y}) "
            f"class={CLASS_NAMES[cls]} "
            f"confidence={probability:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINED RAILWAY FOMO VISUAL VERIFICATION")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    print()
    print(
        "Input resolution:",
        f"{IMAGE_SIZE}×{IMAGE_SIZE}"
    )

    print(
        "FOMO grid:",
        f"{GRID_SIZE}×{GRID_SIZE}"
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD
    )

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    converter = RailwayFOMOConverter(
        VALID_IMAGES,
        VALID_LABELS
    )

    print()
    print(
        "Validation samples:",
        len(converter.samples)
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_trained_model()

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    num_samples = min(
        NUM_SAMPLES,
        len(converter.samples)
    )

    print()
    print(
        f"Testing {num_samples} validation images..."
    )

    # --------------------------------------------------------
    # Test samples
    # --------------------------------------------------------

    for index in range(
        num_samples
    ):

        image_path, label_path = (
            converter.samples[index]
        )

        print()
        print("-" * 70)

        print(
            f"SAMPLE {index + 1}/{num_samples}"
        )

        print(
            "Image:",
            image_path.name
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        resized_image, tensor = (
            preprocess_image(
                image_path
            )
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truth_cells = (
            get_ground_truth_cells(
                converter,
                label_path
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_cells, probabilities = (
            predict(
                model,
                tensor
            )
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            "Ground-truth cells:",
            len(ground_truth_cells)
        )

        print_prediction_summary(
            predicted_cells,
            probabilities
        )

        # ----------------------------------------------------
        # Create visuals
        # ----------------------------------------------------

        ground_truth_visual = (
            draw_prediction(
                resized_image,
                ground_truth_cells,
                "GROUND TRUTH"
            )
        )

        trained_visual = (
            draw_prediction(
                resized_image,
                predicted_cells,
                "TRAINED MODEL"
            )
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        comparison = Image.new(
            "RGB",
            (
                768,
                384
            )
        )

        comparison.paste(
            ground_truth_visual,
            (
                0,
                0
            )
        )

        comparison.paste(
            trained_visual,
            (
                384,
                0
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR /
            f"{index + 1:02d}_prediction.jpg"
        )

        comparison.save(
            output_file,
            quality=95
        )

        print()
        print(
            "Saved:",
            output_file
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINED MODEL VISUAL VERIFICATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Model was NOT modified."
    )

    print(
        "Dataset was NOT modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()