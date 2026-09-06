"""
final_integration.py
Railway_FOMO_ESP32 - Final Scar Edge-AI Reference Pipeline

IMPORTANT:
- Uses the existing trained Scar checkpoint ONLY.
- Does NOT retrain or modify the model.
- Uses the exact DedicatedScarBinaryCNN architecture already used by the
  Scar training/testing pipeline.
- Hybrid target generation belongs to training; inference runs the trained
  model and reads its 12x12 FOMO probability map.
- This Python file is the PC-side reference/integration test. The final
  ESP32 DevKit V1 firmware will use a converted MCU-compatible version of
  the same trained model.
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TRAINING_DIR = PROJECT_DIR / "Railway_FOMO_ESP32" / "training"

# Allow this file to work if it is instead placed inside the inner project.
if not TRAINING_DIR.exists():
    TRAINING_DIR = PROJECT_DIR / "training"

sys.path.insert(0, str(TRAINING_DIR))


# ============================================================
# EXISTING SCAR MODEL
# ============================================================

from scar_binary_v2 import DedicatedScarBinaryCNN


MODEL_PATH = (
    PROJECT_DIR
    / "Railway_FOMO_ESP32"
    / "diagnostic_scar_binary"
    / "scar_mild_augmented_best.pth"
)

if not MODEL_PATH.exists():
    MODEL_PATH = (
        PROJECT_DIR
        / "diagnostic_scar_binary"
        / "scar_mild_augmented_best.pth"
    )


# ============================================================
# MODEL / IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 96
GRID_SIZE = 12

DEFAULT_THRESHOLD = 0.70

OUTPUT_DIR = PROJECT_DIR / "final_integration_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 8
TOP_K = 5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FONT
# ============================================================

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
    SMALL_FONT = ImageFont.truetype("DejaVuSans.ttf", 11)
except Exception:
    FONT = ImageFont.load_default()
    SMALL_FONT = ImageFont.load_default()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("=" * 72)
    print("RAILWAY FOMO - FINAL SCAR INTEGRATION")
    print("=" * 72)

    print()
    print("MODEL")
    print("-" * 72)
    print("Checkpoint :", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nScar checkpoint not found:\n{MODEL_PATH}\n"
        )

    print("Device     :", DEVICE)

    if torch.cuda.is_available():
        print("GPU        :", torch.cuda.get_device_name(0))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = DedicatedScarBinaryCNN().to(DEVICE)

    # The existing checkpoints use model_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Support a raw state_dict without changing the model.
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "epoch": (
            checkpoint.get("epoch", "unknown")
            if isinstance(checkpoint, dict)
            else "unknown"
        ),
        "f1": (
            checkpoint.get("best_f1", "unknown")
            if isinstance(checkpoint, dict)
            else "unknown"
        ),
        "threshold": (
            checkpoint.get("best_threshold", DEFAULT_THRESHOLD)
            if isinstance(checkpoint, dict)
            else DEFAULT_THRESHOLD
        ),
    }

    print("Model      : DedicatedScarBinaryCNN")
    print("Input      : 3 x 96 x 96")
    print("Output     : 1 x 12 x 12")
    print("Training   : Hybrid-target-trained Scar model")
    print("Epoch      :", metadata["epoch"])
    print("Best F1    :", metadata["f1"])
    print("Threshold  :", metadata["threshold"])

    print()
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 72)

    return model, metadata


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image):
    """
    Exact inference preprocessing used by the Scar pipeline:

        RGB
        ↓
        resize 96 x 96
        ↓
        /255
        ↓
        HWC -> CHW
        ↓
        batch dimension
    """

    image_96 = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image_96,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        np.transpose(
            array,
            (2, 0, 1)
        )
    ).unsqueeze(0)

    return image_96, tensor.to(DEVICE)


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def infer(model, tensor):
    start = time.perf_counter()

    logits = model(tensor)
    probability = torch.sigmoid(logits)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    prob_map = (
        probability
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    if prob_map.shape != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(
            f"Unexpected FOMO output shape: {prob_map.shape}. "
            f"Expected {(GRID_SIZE, GRID_SIZE)}."
        )

    return prob_map, elapsed_ms


# ============================================================
# TOP CELLS
# ============================================================

def get_top_cells(prob_map, k=TOP_K):
    flat = prob_map.reshape(-1)

    indices = np.argsort(flat)[-k:][::-1]

    cells = []

    for index in indices:
        gy, gx = np.unravel_index(
            index,
            prob_map.shape
        )

        cells.append(
            {
                "x": int(gx),
                "y": int(gy),
                "probability": float(prob_map[gy, gx])
            }
        )

    return cells


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_annotation(
    image_96,
    prob_map,
    threshold,
    status,
    max_x,
    max_y,
    max_probability,
    inference_ms
):
    """
    Creates a human-readable FOMO grid visualization.

    Detection decision:
        max_probability >= threshold -> SCAR DETECTED
        otherwise                    -> NORMAL TRACK

    We intentionally do NOT require multiple cells above threshold.
    A single strong FOMO cell is enough to represent a localized Scar.
    """

    annotated = image_96.resize(
        (
            IMAGE_SIZE * SCALE,
            IMAGE_SIZE * SCALE
        ),
        Image.Resampling.NEAREST
    )

    draw = ImageDraw.Draw(annotated)

    cell_w = IMAGE_SIZE / GRID_SIZE
    cell_h = IMAGE_SIZE / GRID_SIZE

    detected_cells = []

    # --------------------------------------------------------
    # Draw probability grid
    # --------------------------------------------------------

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):

            probability = float(
                prob_map[gy, gx]
            )

            x1 = int(gx * cell_w * SCALE)
            y1 = int(gy * cell_h * SCALE)

            x2 = int((gx + 1) * cell_w * SCALE)
            y2 = int((gy + 1) * cell_h * SCALE)

            # Grid boundary
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 255, 255),
                width=1
            )

            # Strong cell / threshold cells
            if probability >= threshold:

                detected_cells.append(
                    (
                        gx,
                        gy,
                        probability
                    )
                )

                draw.rectangle(
                    [
                        x1 + 2,
                        y1 + 2,
                        x2 - 2,
                        y2 - 2
                    ],
                    outline=(255, 0, 0),
                    width=3
                )

                draw.text(
                    (
                        x1 + 3,
                        y1 + 3
                    ),
                    f"{probability:.2f}",
                    fill=(255, 255, 255),
                    font=SMALL_FONT
                )

    # --------------------------------------------------------
    # Highlight strongest cell
    # --------------------------------------------------------

    sx1 = int(max_x * cell_w * SCALE)
    sy1 = int(max_y * cell_h * SCALE)

    sx2 = int((max_x + 1) * cell_w * SCALE)
    sy2 = int((max_y + 1) * cell_h * SCALE)

    draw.rectangle(
        [
            sx1 + 1,
            sy1 + 1,
            sx2 - 1,
            sy2 - 1
        ],
        outline=(255, 255, 0),
        width=5
    )

    draw.text(
        (
            sx1 + 3,
            sy1 + 18
        ),
        f"{max_probability:.2f}",
        fill=(255, 255, 0),
        font=FONT
    )

    # --------------------------------------------------------
    # Bottom information panel
    # --------------------------------------------------------

    panel_height = 116

    final_image = Image.new(
        "RGB",
        (
            annotated.width,
            annotated.height + panel_height
        ),
        (0, 0, 0)
    )

    final_image.paste(
        annotated,
        (0, 0)
    )

    panel = ImageDraw.Draw(final_image)

    y = annotated.height + 6

    panel.text(
        (8, y),
        status,
        fill=(255, 255, 255),
        font=FONT
    )

    panel.text(
        (8, y + 20),
        f"Max probability : {max_probability:.4f}",
        fill=(255, 255, 255),
        font=SMALL_FONT
    )

    panel.text(
        (8, y + 38),
        f"Strongest cell  : ({max_x},{max_y})",
        fill=(255, 255, 255),
        font=SMALL_FONT
    )

    panel.text(
        (8, y + 56),
        f"Threshold        : {threshold:.2f}",
        fill=(255, 255, 255),
        font=SMALL_FONT
    )

    panel.text(
        (8, y + 74),
        f"Cells >= threshold: {len(detected_cells)}",
        fill=(255, 255, 255),
        font=SMALL_FONT
    )

    panel.text(
        (8, y + 92),
        f"Inference time   : {inference_ms:.2f} ms",
        fill=(255, 255, 255),
        font=SMALL_FONT
    )

    return final_image, detected_cells


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    model,
    image_path,
    threshold
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    original = Image.open(image_path).convert("RGB")

    image_96, tensor = preprocess(original)

    prob_map, inference_ms = infer(
        model,
        tensor
    )

    max_index = int(
        np.argmax(prob_map)
    )

    max_y, max_x = np.unravel_index(
        max_index,
        prob_map.shape
    )

    max_probability = float(
        prob_map[max_y, max_x]
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if max_probability >= threshold:
        status = "SCAR DETECTED"
    else:
        status = "NORMAL TRACK"

    top_cells = get_top_cells(
        prob_map
    )

    annotated, detected_cells = create_annotation(
        image_96=image_96,
        prob_map=prob_map,
        threshold=threshold,
        status=status,
        max_x=int(max_x),
        max_y=int(max_y),
        max_probability=max_probability,
        inference_ms=inference_ms
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_final_result.png"
    )

    annotated.save(
        output_path
    )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("IMAGE INFERENCE")
    print("=" * 72)

    print("Image          :", image_path)
    print("Original size  :", original.size)
    print("Model input    :", "96 x 96")
    print("FOMO output    :", "12 x 12")
    print("Inference time :", f"{inference_ms:.2f} ms")

    print()
    print("TOP FOMO CELLS")
    print("-" * 72)

    for i, cell in enumerate(top_cells, start=1):
        print(
            f"{i:02d}. Grid=({cell['x']},{cell['y']}) "
            f"Prob={cell['probability']:.4f}"
        )

    print()
    print("FINAL DECISION")
    print("-" * 72)
    print("Status         :", status)
    print("Max probability:", f"{max_probability:.4f}")
    print("Strongest grid :", f"({int(max_x)},{int(max_y)})")
    print("Threshold      :", f"{threshold:.4f}")
    print(
        "Detected cells :",
        len(detected_cells)
    )

    print()
    print("ANNOTATED RESULT")
    print("-" * 72)
    print(output_path)

    print("=" * 72)

    return {
        "image": str(image_path),
        "status": status,
        "max_probability": max_probability,
        "grid_x": int(max_x),
        "grid_y": int(max_y),
        "threshold": threshold,
        "detected_cells": len(detected_cells),
        "inference_ms": inference_ms,
        "output": str(output_path)
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Final Scar FOMO integration test using "
            "scar_mild_augmented_best.pth"
        )
    )

    parser.add_argument(
        "image",
        help="Path to the railway image to test"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold override. "
            "By default the checkpoint's saved threshold is used."
        )
    )

    args = parser.parse_args()

    model, metadata = load_model()

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata["threshold"])
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0.0 and 1.0."
        )

    print()
    print("FINAL INFERENCE THRESHOLD:", threshold)

    process_image(
        model=model,
        image_path=args.image,
        threshold=threshold
    )


if __name__ == "__main__":
    main()
