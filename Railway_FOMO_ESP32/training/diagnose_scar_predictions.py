from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    VALID_IMAGES,
    VALID_LABELS,
    IMAGE_SIZE,
    GRID_SIZE,
    CLASS_MAP
)

from scar_binary_v2 import DedicatedScarBinaryCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# IMPORTANT:
# FINAL MILD-AUGMENTED SCAR MODEL
# ============================================================

CHECKPOINT_PATH = Path(
    "diagnostic_scar_binary/scar_mild_augmented_best.pth"
)

DEFAULT_THRESHOLD = 0.90


# ============================================================
# FIND VALIDATION SAMPLES
# ============================================================

def load_diagnostic_samples():

    image_dir = Path(VALID_IMAGES)
    label_dir = Path(VALID_LABELS)

    scar_samples = []
    bg_samples = []

    for label_file in sorted(
        label_dir.glob("*.txt")
    ):

        text = label_file.read_text().strip()

        has_scar = False

        if text:

            for line in text.splitlines():

                values = line.split()

                if len(values) != 5:
                    continue

                original_class = int(values[0])

                fomo_class = CLASS_MAP.get(
                    original_class,
                    -1
                )

                if fomo_class == 1:
                    has_scar = True
                    break

        # ----------------------------------------------------
        # Find image
        # ----------------------------------------------------

        image_file = None

        for ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        ]:

            candidate = (
                image_dir /
                (label_file.stem + ext)
            )

            if candidate.exists():

                image_file = candidate
                break

        if image_file is None:
            continue

        # ----------------------------------------------------
        # Scar samples
        # ----------------------------------------------------

        if (
            has_scar
            and
            len(scar_samples) < 5
        ):

            scar_samples.append(
                (
                    image_file,
                    label_file,
                    True
                )
            )

        # ----------------------------------------------------
        # Background samples
        # ----------------------------------------------------

        elif (
            not has_scar
            and
            len(bg_samples) < 5
        ):

            bg_samples.append(
                (
                    image_file,
                    label_file,
                    False
                )
            )

        if (
            len(scar_samples) >= 5
            and
            len(bg_samples) >= 5
        ):

            break

    return scar_samples + bg_samples


# ============================================================
# GET ACTUAL SCAR GRID CELLS
# ============================================================

def get_actual_scar_cells(label_file):

    actual_coords = []

    text = label_file.read_text().strip()

    if not text:
        return actual_coords

    for line in text.splitlines():

        values = line.split()

        if len(values) != 5:
            continue

        original_class = int(values[0])

        fomo_class = CLASS_MAP.get(
            original_class,
            -1
        )

        if fomo_class != 1:
            continue

        xc = np.clip(
            float(values[1]),
            0.0,
            1.0
        )

        yc = np.clip(
            float(values[2]),
            0.0,
            1.0
        )

        gx = min(
            int(xc * GRID_SIZE),
            GRID_SIZE - 1
        )

        gy = min(
            int(yc * GRID_SIZE),
            GRID_SIZE - 1
        )

        actual_coords.append(
            (gx, gy)
        )

    return actual_coords


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def run_diagnostics():

    print("=" * 60)
    print("MILD AUGMENTED SCAR MODEL VISUAL / SPATIAL DIAGNOSTIC")
    print("=" * 60)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU   :",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        print()
        print(
            "ERROR: Mild augmented checkpoint not found:"
        )

        print(
            CHECKPOINT_PATH
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
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

    print()
    print(
        "✓ Mild augmented checkpoint loaded"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Saved epoch:",
        saved_epoch
    )

    print(
        "Saved F1:",
        saved_f1
    )

    print(
        "Saved threshold:",
        saved_threshold
    )

    # Use the threshold selected during training
    threshold = float(
        saved_threshold
    )

    # ========================================================
    # LOAD DIAGNOSTIC SAMPLES
    # ========================================================

    samples = load_diagnostic_samples()

    print()
    print(
        f"Testing {len(samples)} validation images:"
    )

    print(
        "5 Scar + 5 Background-only"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    scar_total = 0
    scar_hits = 0

    background_total = 0
    background_false_alarms = 0

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for i, (
            img_path,
            lbl_path,
            expected_scar
        ) in enumerate(
            samples,
            start=1
        ):

            # ------------------------------------------------
            # Prepare image exactly like training
            # ------------------------------------------------

            image = (
                Image.open(img_path)
                .convert("RGB")
                .resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    ),
                    Image.Resampling.BILINEAR
                )
            )

            img_arr = (
                np.asarray(
                    image,
                    dtype=np.float32
                ) / 255.0
            )

            img_tensor = torch.from_numpy(
                np.transpose(
                    img_arr,
                    (2, 0, 1)
                )
            ).unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # CNN inference
            # ------------------------------------------------

            outputs = model(
                img_tensor
            )

            probs = torch.sigmoid(
                outputs
            ).squeeze().cpu().numpy()

            # ------------------------------------------------
            # Strongest prediction
            # ------------------------------------------------

            max_index = np.argmax(probs)

            max_y, max_x = np.unravel_index(
                max_index,
                probs.shape
            )

            max_prob = float(
                probs[max_y, max_x]
            )

            # ------------------------------------------------
            # Actual Scar locations
            # ------------------------------------------------

            actual_coords = (
                get_actual_scar_cells(
                    lbl_path
                )
            )

            print()
            print("-" * 60)

            print(
                f"Sample {i:02d}"
            )

            print(
                "File:",
                img_path.name
            )

            print(
                "Actual Scar:",
                expected_scar
            )

            # =================================================
            # SCAR IMAGE
            # =================================================

            if expected_scar:

                scar_total += 1

                print(
                    "Actual Scar cells:",
                    actual_coords
                )

                print(
                    f"Model maximum:"
                    f" ({max_x},{max_y})"
                )

                print(
                    f"Maximum probability:"
                    f" {max_prob:.4f}"
                )

                # ------------------------------------------------
                # Spatial match
                #
                # ±1 cell tolerance
                # ------------------------------------------------

                hit = any(
                    abs(max_x - gx) <= 1
                    and
                    abs(max_y - gy) <= 1
                    for gx, gy
                    in actual_coords
                )

                if hit:

                    scar_hits += 1

                    print(
                        "Spatial match:"
                        " ✓ SUCCESS"
                    )

                else:

                    print(
                        "Spatial match:"
                        " ✗ MISMATCH"
                    )

            # =================================================
            # BACKGROUND IMAGE
            # =================================================

            else:

                background_total += 1

                print(
                    f"Maximum probability:"
                    f" {max_prob:.4f}"
                )

                if max_prob >= threshold:

                    background_false_alarms += 1

                    print(
                        "Background:"
                        " ⚠ FALSE ALARM"
                    )

                else:

                    print(
                        "Background:"
                        " ✓ No strong Scar"
                    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("MILD AUGMENTED MODEL DIAGNOSTIC SUMMARY")
    print("=" * 60)

    print()

    print(
        f"Scar images tested       : "
        f"{scar_total}"
    )

    print(
        f"Spatial matches          : "
        f"{scar_hits}"
    )

    if scar_total > 0:

        spatial_rate = (
            scar_hits /
            scar_total
        )

        print(
            f"Spatial match rate       : "
            f"{spatial_rate:.2%}"
        )

    print()

    print(
        f"Background images tested : "
        f"{background_total}"
    )

    print(
        f"False alarms @ "
        f"{threshold:.2f}      : "
        f"{background_false_alarms}"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    print()
    print("=" * 60)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("=" * 60)

    print()

    print(
        "Original model:"
    )

    print(
        "  F1              : 0.1384"
    )

    print(
        "  Spatial match   : 80.00%"
    )

    print(
        "  Background FP   : 1/5"
    )

    print()

    print(
        "Mild augmented model:"
    )

    print(
        f"  F1              : "
        f"{float(saved_f1):.4f}"
    )

    print(
        f"  Spatial match   : "
        f"{scar_hits}/{scar_total}"
    )

    print(
        f"  Background FP   : "
        f"{background_false_alarms}/"
        f"{background_total}"
    )

    print()

    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_diagnostics()