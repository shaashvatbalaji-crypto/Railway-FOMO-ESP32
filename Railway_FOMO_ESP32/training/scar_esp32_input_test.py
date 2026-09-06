from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)


from config import (
    IMAGE_SIZE,
    GRID_SIZE
)

from scar_binary_v2 import (
    DedicatedScarBinaryCNN
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# FINAL MILD-AUGMENTED SCAR MODEL
# ------------------------------------------------------------

CHECKPOINT = Path(
    "diagnostic_scar_binary/"
    "scar_mild_augmented_best.pth"
)

# ------------------------------------------------------------
# Test image
# ------------------------------------------------------------

IMAGE_PATH = Path(
    "rail_test.png"
)

# ------------------------------------------------------------
# Best validation threshold from training
# ------------------------------------------------------------

THRESHOLD = 0.90


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MILD AUGMENTED SCAR ESP32-STYLE INPUT TEST")
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

        print()
        print(
            "Make sure rail_test.png is in the "
            "project root."
        )

        return

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not CHECKPOINT.exists():

        print()
        print("ERROR: Model not found:")

        print(
            CHECKPOINT
        )

        print()
        print(
            "Expected:"
        )

        print(
            "diagnostic_scar_binary/"
            "scar_mild_augmented_best.pth"
        )

        return

    # ========================================================
    # LOAD ORIGINAL IMAGE
    # ========================================================

    original = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print()
    print("Original image")

    print(
        "  Size :",
        original.size
    )

    print(
        "  Mode :",
        original.mode
    )

    # ========================================================
    # ESP32-STYLE PREPROCESSING
    # ========================================================

    image_96 = original.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Convert to uint8
    # --------------------------------------------------------

    image_uint8 = np.asarray(
        image_96,
        dtype=np.uint8
    )

    print()
    print("ESP32-compatible image")

    print(
        "  Size :",
        image_uint8.shape
    )

    print(
        "  Type :",
        image_uint8.dtype
    )

    print(
        "  Range:",
        int(image_uint8.min()),
        "to",
        int(image_uint8.max())
    )

    # ========================================================
    # CONVERT TO CNN INPUT
    # ========================================================

    # uint8 [0,255]
    #        ↓
    # float32 [0,1]

    image_float = (
        image_uint8.astype(
            np.float32
        ) / 255.0
    )

    # HWC
    # ↓
    # CHW

    image_tensor = torch.from_numpy(
        np.transpose(
            image_float,
            (2, 0, 1)
        )
    )

    # Add batch dimension

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    print()
    print("CNN input tensor")

    print(
        "  Shape :",
        tuple(image_tensor.shape)
    )

    print(
        "  Type  :",
        image_tensor.dtype
    )

    print(
        "  Range :",
        float(image_tensor.min()),
        "to",
        float(image_tensor.max())
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = (
        DedicatedScarBinaryCNN()
        .to(DEVICE)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

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
        THRESHOLD
    )

    print()
    print("Model loaded successfully")

    print(
        "  Checkpoint:",
        CHECKPOINT
    )

    print(
        "  Saved epoch:",
        saved_epoch
    )

    print(
        "  Saved F1:",
        saved_f1
    )

    print(
        "  Saved threshold:",
        saved_threshold
    )

    # Use the threshold selected during training
    threshold = float(
        saved_threshold
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

    # ========================================================
    # PROBABILITY MAP
    # ========================================================

    prob_map = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    # ========================================================
    # FIND STRONGEST CELL
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

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("ESP32-STYLE SCAR INFERENCE")
    print("=" * 60)

    print()

    print(
        f"Input          : "
        f"{IMAGE_SIZE} × {IMAGE_SIZE}"
    )

    print(
        f"Output grid    : "
        f"{GRID_SIZE} × {GRID_SIZE}"
    )

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

    # ========================================================
    # SCAR DECISION
    # ========================================================

    if max_probability >= threshold:

        print()
        print(
            "✓ SCAR DETECTED"
        )

        print(
            f"Confidence: "
            f"{max_probability * 100:.2f}%"
        )

    else:

        print()
        print(
            "✗ NO SCAR DETECTED"
        )

        print(
            "No grid cell crossed "
            f"the threshold of {threshold:.2f}."
        )

    # ========================================================
    # TOP 5 GRID PROBABILITIES
    # ========================================================

    flat = prob_map.flatten()

    top_indices = np.argsort(
        flat
    )[-5:][::-1]

    print()
    print(
        "TOP 5 GRID PROBABILITIES"
    )

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        y, x = np.unravel_index(
            index,
            prob_map.shape
        )

        print(
            f"{rank}. "
            f"Grid=({x},{y}) "
            f"Probability="
            f"{prob_map[y, x]:.4f}"
        )

    # ========================================================
    # FORMAT VERIFICATION
    # ========================================================

    print()
    print("=" * 60)
    print("FORMAT VERIFICATION")
    print("=" * 60)

    checks = [

        (
            "RGB image",
            image_uint8.shape
            == (
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            )
        ),

        (
            "uint8 pixels",
            image_uint8.dtype
            == np.uint8
        ),

        (
            "96x96 input",
            image_uint8.shape[:2]
            == (
                96,
                96
            )
        ),

        (
            "12x12 output",
            prob_map.shape
            == (
                GRID_SIZE,
                GRID_SIZE
            )
        )
    ]

    all_passed = True

    for name, result in checks:

        if result:

            print(
                f"✓ {name}"
            )

        else:

            print(
                f"✗ {name}"
            )

            all_passed = False

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    if all_passed:

        print()
        print(
            "✓ ESP32-STYLE INPUT FORMAT PASSED"
        )

    else:

        print()
        print(
            "✗ INPUT FORMAT VERIFICATION FAILED"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()