from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PROJECT PATH
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from config import IMAGE_SIZE, GRID_SIZE
from scar_binary_v2 import DedicatedScarBinaryCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Mild augmented Scar model
CHECKPOINT = Path(
    "diagnostic_scar_binary/"
    "scar_mild_augmented_best.pth"
)

# Change this to test another image
IMAGE_PATH = Path(
    "scar_test.png"
)

# Output directory
OUTPUT_DIR = Path(
    "diagnostic_scar_binary/image_tests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# IMPORTANT
# Use the threshold saved during training.
# Current best threshold = 0.90
# ============================================================

DEFAULT_THRESHOLD = 0.90


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SCAR VISUAL / GRID ANNOTATION TEST")
    print("=" * 60)

    print()
    print("Device :", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU    :",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if not IMAGE_PATH.exists():

        print()
        print("ERROR: Image not found:")
        print(
            IMAGE_PATH.resolve()
        )

        return

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not CHECKPOINT.exists():

        print()
        print("ERROR: Model not found:")
        print(CHECKPOINT)

        return

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model = (
        DedicatedScarBinaryCNN()
        .to(DEVICE)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    saved_threshold = checkpoint.get(
        "best_threshold",
        DEFAULT_THRESHOLD
    )

    threshold = float(
        saved_threshold
    )

    saved_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    saved_f1 = checkpoint.get(
        "best_f1",
        "unknown"
    )

    print()
    print("✓ Model loaded")

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "Epoch:",
        saved_epoch
    )

    print(
        "Validation F1:",
        saved_f1
    )

    print(
        "Threshold:",
        threshold
    )

    # ========================================================
    # LOAD ORIGINAL IMAGE
    # ========================================================

    original = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print()
    print("Original image")

    print(
        "  Size:",
        original.size
    )

    # ========================================================
    # CREATE CNN INPUT
    # ========================================================

    image_96 = original.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    image_uint8 = np.asarray(
        image_96,
        dtype=np.uint8
    )

    image_float = (
        image_uint8.astype(
            np.float32
        ) / 255.0
    )

    image_tensor = torch.from_numpy(
        np.transpose(
            image_float,
            (2, 0, 1)
        )
    ).unsqueeze(0).to(DEVICE)

    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

    prob_map = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    # ========================================================
    # STRONGEST CELL
    # ========================================================

    max_index = np.argmax(
        prob_map
    )

    max_y, max_x = np.unravel_index(
        max_index,
        prob_map.shape
    )

    max_probability = float(
        prob_map[
            max_y,
            max_x
        ]
    )

    print()
    print("=" * 60)
    print("DETECTION RESULT")
    print("=" * 60)

    print()
    print(
        f"Strongest cell : "
        f"({max_x}, {max_y})"
    )

    print(
        f"Probability    : "
        f"{max_probability:.4f}"
    )

    print(
        f"Threshold      : "
        f"{threshold:.2f}"
    )

    if max_probability >= threshold:

        print()
        print(
            "✓ SCAR DETECTED"
        )

    else:

        print()
        print(
            "✗ NO SCAR DETECTED"
        )

    # ========================================================
    # CREATE ANNOTATED IMAGE
    # ========================================================
    #
    # We draw the grid on the 96x96 image first.
    # Then enlarge it so the grid is easy to see.
    #

    annotated = image_96.copy()

    draw = ImageDraw.Draw(
        annotated,
        "RGBA"
    )

    width, height = annotated.size

    cell_width = (
        width /
        GRID_SIZE
    )

    cell_height = (
        height /
        GRID_SIZE
    )

    # ========================================================
    # DRAW GRID
    # ========================================================

    for x in range(
        GRID_SIZE + 1
    ):

        px = int(
            x * cell_width
        )

        draw.line(
            [
                (px, 0),
                (px, height)
            ],
            fill=(255, 255, 255, 120),
            width=1
        )

    for y in range(
        GRID_SIZE + 1
    ):

        py = int(
            y * cell_height
        )

        draw.line(
            [
                (0, py),
                (width, py)
            ],
            fill=(255, 255, 255, 120),
            width=1
        )

    # ========================================================
    # HIGHLIGHT CELLS ABOVE THRESHOLD
    # ========================================================

    detected_cells = []

    for gy in range(
        GRID_SIZE
    ):

        for gx in range(
            GRID_SIZE
        ):

            prob = float(
                prob_map[
                    gy,
                    gx
                ]
            )

            if prob >= threshold:

                detected_cells.append(
                    (
                        gx,
                        gy,
                        prob
                    )
                )

                left = int(
                    gx * cell_width
                )

                top = int(
                    gy * cell_height
                )

                right = int(
                    (gx + 1) *
                    cell_width
                )

                bottom = int(
                    (gy + 1) *
                    cell_height
                )

                # Highlight detected cell
                draw.rectangle(
                    [
                        left,
                        top,
                        right,
                        bottom
                    ],
                    fill=(255, 0, 0, 80),
                    outline=(255, 0, 0, 255),
                    width=2
                )

    # ========================================================
    # HIGHLIGHT STRONGEST CELL
    # ========================================================

    left = int(
        max_x * cell_width
    )

    top = int(
        max_y * cell_height
    )

    right = int(
        (max_x + 1) *
        cell_width
    )

    bottom = int(
        (max_y + 1) *
        cell_height
    )

    # Strongest cell = yellow
    draw.rectangle(
        [
            left,
            top,
            right,
            bottom
        ],
        fill=(255, 255, 0, 80),
        outline=(255, 255, 0, 255),
        width=3
    )

    # ========================================================
    # RESIZE FOR EASY VIEWING
    # ========================================================

    display_size = (
        768,
        768
    )

    annotated = annotated.resize(
        display_size,
        Image.Resampling.NEAREST
    )

    # ========================================================
    # DRAW TEXT
    # ========================================================

    draw = ImageDraw.Draw(
        annotated
    )

    # Try to load a standard font
    try:

        font = ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            20
        )

        small_font = ImageFont.truetype(
            "DejaVuSans.ttf",
            14
        )

    except:

        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if max_probability >= threshold:

        title = (
            f"SCAR DETECTED | "
            f"Prob={max_probability:.3f} | "
            f"Grid=({max_x},{max_y})"
        )

    else:

        title = (
            f"NO SCAR | "
            f"Max={max_probability:.3f} | "
            f"Grid=({max_x},{max_y})"
        )

    draw.rectangle(
        [
            0,
            0,
            768,
            38
        ],
        fill=(0, 0, 0)
    )

    draw.text(
        (
            10,
            8
        ),
        title,
        fill=(255, 255, 255),
        font=font
    )

    # ========================================================
    # GRID COORDINATE LABELS
    # ========================================================

    scale = (
        display_size[0] /
        IMAGE_SIZE
    )

    for gy in range(
        GRID_SIZE
    ):

        for gx in range(
            GRID_SIZE
        ):

            prob = float(
                prob_map[
                    gy,
                    gx
                ]
            )

            # Only write probabilities for
            # reasonably active cells.
            if prob >= 0.50:

                cx = int(
                    (
                        gx + 0.5
                    )
                    *
                    cell_width
                    *
                    scale
                )

                cy = int(
                    (
                        gy + 0.5
                    )
                    *
                    cell_height
                    *
                    scale
                )

                text = (
                    f"{prob:.2f}"
                )

                draw.text(
                    (
                        cx - 15,
                        cy - 8
                    ),
                    text,
                    fill=(255, 255, 255),
                    font=small_font
                )

    # ========================================================
    # SAVE
    # ========================================================

    output_name = (
        IMAGE_PATH.stem +
        "_scar_annotated.png"
    )

    output_path = (
        OUTPUT_DIR /
        output_name
    )

    annotated.save(
        output_path
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("ANNOTATED RESULT")
    print("=" * 60)

    print()
    print(
        "Detected cells above threshold:",
        len(detected_cells)
    )

    if detected_cells:

        print()

        for gx, gy, prob in sorted(
            detected_cells,
            key=lambda x: x[2],
            reverse=True
        )[:10]:

            print(
                f"Grid=({gx},{gy}) "
                f"Probability={prob:.4f}"
            )

    print()
    print(
        "Strongest cell:"
    )

    print(
        f"  Grid        : "
        f"({max_x},{max_y})"
    )

    print(
        f"  Probability : "
        f"{max_probability:.4f}"
    )

    print()
    print(
        "Saved annotated image:"
    )

    print(
        output_path
    )

    print()
    print("=" * 60)
    print("VISUAL TEST COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()