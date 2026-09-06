from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import IMAGE_SIZE, GRID_SIZE
from scar_binary_v2 import DedicatedScarBinaryCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = Path(
    "diagnostic_scar_binary/best_scar_model.pth"
)

DEFAULT_THRESHOLD = 0.90

OUTPUT_DIR = Path(
    "diagnostic_scar_binary/image_tests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        np.transpose(
            image_array,
            (2, 0, 1)
        )
    )

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not CHECKPOINT.exists():

        print()
        print("ERROR: Model checkpoint not found.")
        print()
        print("Expected:")
        print(CHECKPOINT)
        print()

        sys.exit(1)

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model = DedicatedScarBinaryCNN().to(
        DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    saved_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    saved_f1 = checkpoint.get(
        "best_f1",
        "unknown"
    )

    saved_threshold = checkpoint.get(
        "best_threshold",
        DEFAULT_THRESHOLD
    )

    print()
    print("✓ Model loaded successfully")

    print(
        "Saved epoch     :",
        saved_epoch
    )

    print(
        "Saved F1        :",
        saved_f1
    )

    print(
        "Saved threshold :",
        saved_threshold
    )

    return model, float(saved_threshold)


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    original_image,
    detections,
    top_detection
):

    image = original_image.copy()

    draw = ImageDraw.Draw(image)

    width, height = image.size

    cell_width = width / GRID_SIZE
    cell_height = height / GRID_SIZE

    # --------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------

    for x in range(GRID_SIZE + 1):

        px = int(x * cell_width)

        draw.line(
            [(px, 0), (px, height)],
            fill=(255, 255, 0),
            width=1
        )

    for y in range(GRID_SIZE + 1):

        py = int(y * cell_height)

        draw.line(
            [(0, py), (width, py)],
            fill=(255, 255, 0),
            width=1
        )

    # --------------------------------------------------------
    # Draw Scar detections
    # --------------------------------------------------------

    for detection in detections:

        gx = detection["grid_x"]
        gy = detection["grid_y"]
        probability = detection["probability"]

        left = int(gx * cell_width)
        top = int(gy * cell_height)

        right = int(
            (gx + 1) * cell_width
        )

        bottom = int(
            (gy + 1) * cell_height
        )

        # Red detection box
        draw.rectangle(
            [left, top, right, bottom],
            outline=(255, 0, 0),
            width=5
        )

        # Center point
        center_x = int(
            (left + right) / 2
        )

        center_y = int(
            (top + bottom) / 2
        )

        radius = 8

        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius
            ],
            fill=(255, 0, 0)
        )

        # Probability label
        label = f"Scar {probability:.2f}"

        label_y = max(
            0,
            top + 2
        )

        draw.text(
            (left + 3, label_y),
            label,
            fill=(255, 0, 0)
        )

    # --------------------------------------------------------
    # Highlight strongest detection
    # --------------------------------------------------------

    if top_detection is not None:

        gx = top_detection["grid_x"]
        gy = top_detection["grid_y"]

        left = int(gx * cell_width)
        top = int(gy * cell_height)

        right = int(
            (gx + 1) * cell_width
        )

        bottom = int(
            (gy + 1) * cell_height
        )

        draw.rectangle(
            [left, top, right, bottom],
            outline=(0, 255, 0),
            width=7
        )

    return image


# ============================================================
# RUN INFERENCE
# ============================================================

def test_image(image_path):

    print()
    print("=" * 60)
    print("RAILWAY SCAR IMAGE TEST")
    print("=" * 60)

    print()
    print("Input image:")
    print(image_path)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        print()
        print("ERROR: Image not found.")
        print(image_path)

        return

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    print()
    print(
        "Original image size:",
        original_image.size
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, threshold = load_model()

    print()
    print("Device    :", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU       :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Threshold :",
        threshold
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = preprocess_image(
        original_image
    ).to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            output
        )

    # --------------------------------------------------------
    # Convert probability map
    # --------------------------------------------------------

    probability_map = (
        probabilities[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Find detections
    # --------------------------------------------------------

    detections = []

    for gy in range(GRID_SIZE):

        for gx in range(GRID_SIZE):

            probability = float(
                probability_map[gy, gx]
            )

            if probability >= threshold:

                detections.append(
                    {
                        "grid_x": gx,
                        "grid_y": gy,
                        "probability": probability
                    }
                )

    # --------------------------------------------------------
    # Sort detections by probability
    # --------------------------------------------------------

    detections.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    # --------------------------------------------------------
    # Top probability
    # --------------------------------------------------------

    flat_indices = np.argsort(
        probability_map.flatten()
    )[::-1]

    print()
    print("=" * 60)
    print("TOP 10 GRID PROBABILITIES")
    print("=" * 60)

    for rank, index in enumerate(
        flat_indices[:10],
        start=1
    ):

        gy, gx = np.unravel_index(
            index,
            probability_map.shape
        )

        probability = float(
            probability_map[gy, gx]
        )

        print(
            f"{rank:02d}. "
            f"Grid=({gx:02d},{gy:02d}) "
            f"Probability={probability:.4f}"
        )

    # --------------------------------------------------------
    # Detection result
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SCAR DETECTION RESULT")
    print("=" * 60)

    if len(detections) == 0:

        print()
        print("✗ NO SCAR DETECTED")
        print()
        print(
            "No grid cell crossed the "
            f"threshold of {threshold:.2f}."
        )

    else:

        print()
        print(
            f"✓ SCAR DETECTED"
        )

        print()
        print(
            "Detection cells:",
            len(detections)
        )

        print()

        for i, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"{i:02d}. "
                f"Grid cell = "
                f"({detection['grid_x']}, "
                f"{detection['grid_y']}) | "
                f"Probability = "
                f"{detection['probability']:.4f}"
            )

    # --------------------------------------------------------
    # Strongest detection
    # --------------------------------------------------------

    top_detection = None

    if len(detections) > 0:

        top_detection = detections[0]

    else:

        max_index = np.argmax(
            probability_map
        )

        gy, gx = np.unravel_index(
            max_index,
            probability_map.shape
        )

        top_detection = {
            "grid_x": int(gx),
            "grid_y": int(gy),
            "probability": float(
                probability_map[gy, gx]
            )
        }

    print()
    print(
        "Strongest grid cell:"
    )

    print(
        f"  Grid X        : "
        f"{top_detection['grid_x']}"
    )

    print(
        f"  Grid Y        : "
        f"{top_detection['grid_y']}"
    )

    print(
        f"  Probability   : "
        f"{top_detection['probability']:.4f}"
    )

    # --------------------------------------------------------
    # Draw annotated image
    # --------------------------------------------------------

    annotated = draw_detections(
        original_image,
        detections,
        top_detection
    )

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_scar_result.png"
    )

    annotated.save(
        output_path
    )

    print()
    print("=" * 60)
    print("ANNOTATED RESULT")
    print("=" * 60)

    print()
    print(
        "Saved to:"
    )

    print(
        output_path
    )

    print()
    print("=" * 60)
    print("IMAGE TEST COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python training/scar_image_test.py "
            "<image_path>"
        )

        print()
        print(
            "Example:"
        )

        print(
            "python training/scar_image_test.py "
            "/path/to/rail_image.jpg"
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    test_image(image_path)