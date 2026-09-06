from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

# ------------------------------------------------------------
# Import project files
# ------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

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


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path("evaluation/visualization")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = image.copy()

    resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

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

    return original, resized, tensor


# ============================================================
# FOMO TARGET VISUALIZATION
# ============================================================

def get_target_cells(
    converter,
    label_file
):

    target = converter.encode_label(
        label_file
    )

    cells = []

    for y in range(GRID_SIZE):

        for x in range(GRID_SIZE):

            for cls in range(NUM_CLASSES):

                if target[y, x, cls] > 0:

                    cells.append(
                        (
                            x,
                            y,
                            cls
                        )
                    )

    return cells


# ============================================================
# DRAW GRID
# ============================================================

def draw_grid(
    image,
    cells,
    title
):

    canvas = image.resize(
        (
            384,
            384
        )
    ).copy()

    draw = ImageDraw.Draw(
        canvas
    )

    cell_size = 384 // GRID_SIZE

    # Grid lines
    for i in range(GRID_SIZE + 1):

        position = i * cell_size

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

    # Highlight target/prediction cells
    for x, y, cls in cells:

        left = x * cell_size
        top = y * cell_size
        right = left + cell_size
        bottom = top + cell_size

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

        label = CLASS_NAMES[cls]

        draw.text(
            (
                left + 4,
                top + 4
            ),
            label,
            fill="yellow"
        )

    # Title
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
# MODEL PREDICTION
# ============================================================

def predict(
    model,
    tensor
):

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            tensor
        )

    # [1, C, H, W]
    prediction = output[0]

    # Find highest class for each grid cell
    class_map = prediction.argmax(
        dim=0
    )

    # Confidence-like score
    confidence_map = torch.softmax(
        prediction,
        dim=0
    )

    max_confidence = confidence_map.max(
        dim=0
    ).values

    cells = []

    for y in range(GRID_SIZE):

        for x in range(GRID_SIZE):

            cls = class_map[
                y,
                x
            ].item()

            confidence = max_confidence[
                y,
                x
            ].item()

            cells.append(
                (
                    x,
                    y,
                    cls,
                    confidence
                )
            )

    return cells


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RAILWAY FOMO VISUAL VERIFICATION")
    print("=" * 60)

    print()
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset converter
    # --------------------------------------------------------

    converter = RailwayFOMOConverter(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    print(
        "Training samples:",
        len(converter.samples)
    )

    # --------------------------------------------------------
    # Pick first sample
    # --------------------------------------------------------

    image_path, label_path = (
        converter.samples[0]
    )

    print()
    print("Image:")
    print(image_path)

    print()
    print("Label:")
    print(label_path)

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    original, resized, tensor = (
        preprocess_image(
            image_path
        )
    )

    print()
    print(
        "Model input:",
        tuple(tensor.shape)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    target_cells = get_target_cells(
        converter,
        label_path
    )

    print()
    print("Ground-truth FOMO cells:")

    if len(target_cells) == 0:

        print("  No defect cell")

    else:

        for x, y, cls in target_cells:

            print(
                f"  grid=({x},{y}) "
                f"class={cls} "
                f"{CLASS_NAMES[cls]}"
            )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = FOMOModel(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = predict(
        model,
        tensor
    )

    # Only display high-confidence predictions
    threshold = 0.40

    predicted_cells = [
        (
            x,
            y,
            cls,
            confidence
        )
        for x, y, cls, confidence
        in predictions
        if confidence >= threshold
    ]

    print()
    print(
        f"Predicted cells "
        f"(confidence >= {threshold}):"
    )

    if len(predicted_cells) == 0:

        print("  None")

    else:

        for (
            x,
            y,
            cls,
            confidence
        ) in predicted_cells:

            print(
                f"  grid=({x},{y}) "
                f"class={cls} "
                f"{CLASS_NAMES[cls]} "
                f"confidence={confidence:.3f}"
            )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    original_file = (
        OUTPUT_DIR /
        "01_original.jpg"
    )

    resized_file = (
        OUTPUT_DIR /
        "02_model_input_96x96.jpg"
    )

    target_file = (
        OUTPUT_DIR /
        "03_ground_truth_grid.jpg"
    )

    prediction_file = (
        OUTPUT_DIR /
        "04_untrained_prediction.jpg"
    )

    original.save(
        original_file
    )

    resized.save(
        resized_file
    )

    target_visual = draw_grid(
        resized,
        target_cells,
        "GROUND TRUTH"
    )

    target_visual.save(
        target_file
    )

    prediction_visual = draw_grid(
        resized,
        [
            (
                x,
                y,
                cls
            )
            for x, y, cls, confidence
            in predicted_cells
        ],
        "UNTRAINED MODEL"
    )

    prediction_visual.save(
        prediction_file
    )

    print()
    print("Saved:")
    print(original_file)
    print(resized_file)
    print(target_file)
    print(prediction_file)

    print()
    print("=" * 60)
    print("VISUAL VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()