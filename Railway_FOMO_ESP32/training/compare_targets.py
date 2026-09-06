from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_IMAGES, TRAIN_LABELS, GRID_SIZE, CLASS_NAMES
from dataset import RailwayFOMOConverter


OUTPUT_DIR = Path("evaluation/target_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = GRID_SIZE


# ============================================================
# SELECT REPRESENTATIVE SAMPLES
# ============================================================

wanted_classes = {
    0: 3,   # Cracks
    2: 3,   # Scars
    3: 2,   # breaks
    4: 2,   # lightbands
}

selected = []

labels_dir = Path(TRAIN_LABELS)
images_dir = Path(TRAIN_IMAGES)

for label_file in labels_dir.glob("*.txt"):

    text = label_file.read_text().splitlines()

    for line in text:

        p = line.split()

        if len(p) != 5:
            continue

        cls = int(p[0])

        if cls not in wanted_classes:
            continue

        if wanted_classes[cls] <= 0:
            continue

        stem = label_file.stem

        image = None

        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

            candidate = images_dir / (stem + ext)

            if candidate.exists():
                image = candidate
                break

        if image is not None:

            selected.append(
                (cls, image, label_file)
            )

            wanted_classes[cls] -= 1

            break


print("=" * 60)
print("TARGET STRATEGY VISUAL COMPARISON")
print("=" * 60)

print()
print("Selected samples:", len(selected))


# ============================================================
# GET YOLO BOX
# ============================================================

def get_boxes(label_file):

    boxes = []

    for line in label_file.read_text().splitlines():

        p = line.split()

        if len(p) != 5:
            continue

        cls = int(p[0])

        if cls == 1:
            continue

        x = float(p[1])
        y = float(p[2])
        w = float(p[3])
        h = float(p[4])

        boxes.append(
            (cls, x, y, w, h)
        )

    return boxes


# ============================================================
# CENTER-ONLY TARGET
# ============================================================

def center_target(box):

    cls, x, y, w, h = box

    gx = min(
        int(x * GRID),
        GRID - 1
    )

    gy = min(
        int(y * GRID),
        GRID - 1
    )

    return {(gx, gy, cls)}


# ============================================================
# FULL-BOX TARGET
# ============================================================

def full_box_target(box):

    cls, x, y, w, h = box

    xmin = max(0.0, x - w / 2)
    xmax = min(1.0, x + w / 2)

    ymin = max(0.0, y - h / 2)
    ymax = min(1.0, y + h / 2)

    cells = set()

    for gy in range(GRID):

        for gx in range(GRID):

            cx = (gx + 0.5) / GRID
            cy = (gy + 0.5) / GRID

            if (
                xmin <= cx <= xmax
                and
                ymin <= cy <= ymax
            ):
                cells.add(
                    (gx, gy, cls)
                )

    # Always preserve the center
    cx = min(int(x * GRID), GRID - 1)
    cy = min(int(y * GRID), GRID - 1)

    cells.add(
        (cx, cy, cls)
    )

    return cells


# ============================================================
# HYBRID TARGET
# ============================================================

def hybrid_target(box):

    cls, x, y, w, h = box

    center_x = min(
        int(x * GRID),
        GRID - 1
    )

    center_y = min(
        int(y * GRID),
        GRID - 1
    )

    cells = {
        (center_x, center_y, cls)
    }

    # Estimate extent in grid cells
    grid_w = max(
        1,
        round(w * GRID)
    )

    grid_h = max(
        1,
        round(h * GRID)
    )

    # Limit expansion so very large boxes
    # don't dominate the target.
    max_cells = 7

    if grid_h > grid_w:

        # Vertical defect
        radius = min(
            grid_h // 2,
            3
        )

        for dy in range(
            -radius,
            radius + 1
        ):

            gy = center_y + dy

            if 0 <= gy < GRID:

                cells.add(
                    (
                        center_x,
                        gy,
                        cls
                    )
                )

    else:

        # Horizontal / compact defect
        radius_x = min(
            grid_w // 2,
            2
        )

        radius_y = min(
            grid_h // 2,
            2
        )

        for dy in range(
            -radius_y,
            radius_y + 1
        ):

            for dx in range(
                -radius_x,
                radius_x + 1
            ):

                gx = center_x + dx
                gy = center_y + dy

                if (
                    0 <= gx < GRID
                    and
                    0 <= gy < GRID
                ):

                    cells.add(
                        (
                            gx,
                            gy,
                            cls
                        )
                    )

    # Hard limit
    if len(cells) > max_cells:

        ordered = sorted(
            cells,
            key=lambda c:
            abs(c[0] - center_x)
            +
            abs(c[1] - center_y)
        )

        cells = set(
            ordered[:max_cells]
        )

    return cells


# ============================================================
# DRAW TARGET
# ============================================================

def draw_target(
    image,
    cells,
    title
):

    canvas = image.resize(
        (384, 384)
    ).copy()

    draw = ImageDraw.Draw(canvas)

    cell_size = 384 // GRID

    # Grid
    for i in range(GRID + 1):

        p = i * cell_size

        draw.line(
            (p, 0, p, 384),
            fill="white"
        )

        draw.line(
            (0, p, 384, p),
            fill="white"
        )

    # Positive cells
    for gx, gy, cls in cells:

        left = gx * cell_size
        top = gy * cell_size
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

        draw.text(
            (
                left + 3,
                top + 3
            ),
            CLASS_NAMES[
                0 if cls == 0 else
                1 if cls == 2 else
                2 if cls == 3 else
                3
            ],
            fill="yellow"
        )

    draw.rectangle(
        (0, 0, 384, 22),
        fill="black"
    )

    draw.text(
        (5, 5),
        title,
        fill="white"
    )

    return canvas


# ============================================================
# CREATE COMPARISONS
# ============================================================

for index, (cls, image_path, label_path) in enumerate(selected):

    image = Image.open(
        image_path
    ).convert("RGB")

    boxes = get_boxes(
        label_path
    )

    # Use the first defect box for this sample
    box = next(
        b for b in boxes
        if b[0] == cls
    )

    center = center_target(box)
    full = full_box_target(box)
    hybrid = hybrid_target(box)

    print()
    print(
        f"{index + 1}. "
        f"{CLASS_NAMES[0] if cls == 0 else 'class ' + str(cls)}"
    )

    print(
        "Center cells :",
        len(center)
    )

    print(
        "Full cells   :",
        len(full)
    )

    print(
        "Hybrid cells :",
        len(hybrid)
    )

    # --------------------------------------------------------
    # Create side-by-side comparison
    # --------------------------------------------------------

    a = draw_target(
        image,
        center,
        "CENTER"
    )

    b = draw_target(
        image,
        full,
        "FULL BOX"
    )

    c = draw_target(
        image,
        hybrid,
        "HYBRID"
    )

    comparison = Image.new(
        "RGB",
        (
            384 * 3,
            384
        ),
        "black"
    )

    comparison.paste(
        a,
        (0, 0)
    )

    comparison.paste(
        b,
        (384, 0)
    )

    comparison.paste(
        c,
        (768, 0)
    )

    output = (
        OUTPUT_DIR /
        f"{index + 1:02d}_{image_path.stem}.jpg"
    )

    comparison.save(
        output,
        quality=95
    )

    print(
        "Saved:",
        output
    )


print()
print("=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)