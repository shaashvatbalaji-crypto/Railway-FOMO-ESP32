from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_IMAGES, TRAIN_LABELS, GRID_SIZE


GRID = GRID_SIZE

CLASS_NAMES = {
    0: "Cracks",
    2: "Scars",
    3: "breaks",
    4: "lightbands",
}

OUTPUT_DIR = Path("evaluation/adaptive_targets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND SAMPLES
# ============================================================

wanted = {
    0: 3,
    2: 3,
    3: 2,
    4: 2,
}

images_dir = Path(TRAIN_IMAGES)
labels_dir = Path(TRAIN_LABELS)

samples = []

for label_file in labels_dir.glob("*.txt"):

    for line in label_file.read_text().splitlines():

        p = line.split()

        if len(p) != 5:
            continue

        cls = int(p[0])

        if cls not in wanted or wanted[cls] <= 0:
            continue

        image_path = None

        for ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        ]:

            candidate = images_dir / (
                label_file.stem + ext
            )

            if candidate.exists():
                image_path = candidate
                break

        if image_path:

            samples.append(
                (cls, image_path, label_file)
            )

            wanted[cls] -= 1

            break


# ============================================================
# ADAPTIVE TARGET
# ============================================================

def adaptive_target(box):

    cls, x, y, w, h = box

    # --------------------------------------------------------
    # Center cell
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convert box dimensions to grid dimensions
    # --------------------------------------------------------

    grid_w = max(
        1,
        round(w * GRID)
    )

    grid_h = max(
        1,
        round(h * GRID)
    )

    # --------------------------------------------------------
    # LONG VERTICAL DEFECT
    # --------------------------------------------------------

    if grid_h >= grid_w * 1.5:

        # Number of cells proportional to height
        count = min(
            max(3, round(grid_h * 0.65)),
            8
        )

        # Spread cells across the vertical box
        top = y - h / 2
        bottom = y + h / 2

        for i in range(count):

            if count == 1:
                fraction = 0.5
            else:
                fraction = i / (count - 1)

            cy_norm = (
                top +
                fraction * (bottom - top)
            )

            gy = min(
                max(
                    int(cy_norm * GRID),
                    0
                ),
                GRID - 1
            )

            cells.add(
                (
                    center_x,
                    gy,
                    cls
                )
            )

    # --------------------------------------------------------
    # WIDE HORIZONTAL DEFECT
    # --------------------------------------------------------

    elif grid_w >= grid_h * 1.5:

        count = min(
            max(3, round(grid_w * 0.65)),
            6
        )

        left = x - w / 2
        right = x + w / 2

        for i in range(count):

            if count == 1:
                fraction = 0.5
            else:
                fraction = i / (count - 1)

            cx_norm = (
                left +
                fraction * (right - left)
            )

            gx = min(
                max(
                    int(cx_norm * GRID),
                    0
                ),
                GRID - 1
            )

            cells.add(
                (
                    gx,
                    center_y,
                    cls
                )
            )

    # --------------------------------------------------------
    # COMPACT DEFECT
    # --------------------------------------------------------

    else:

        radius_x = 1 if grid_w >= 2 else 0
        radius_y = 1 if grid_h >= 2 else 0

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

    return cells


# ============================================================
# READ BOXES
# ============================================================

def get_box(label_file, wanted_class):

    for line in label_file.read_text().splitlines():

        p = line.split()

        if len(p) != 5:
            continue

        cls = int(p[0])

        if cls == wanted_class:

            return (
                cls,
                float(p[1]),
                float(p[2]),
                float(p[3]),
                float(p[4])
            )

    return None


# ============================================================
# DRAW TARGET
# ============================================================

def draw_target(image, cells, title):

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

    # Cells
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
            CLASS_NAMES[cls],
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
# GENERATE
# ============================================================

print("=" * 60)
print("ADAPTIVE FOMO TARGET VERIFICATION")
print("=" * 60)

print()
print("Samples:", len(samples))

for index, (
    cls,
    image_path,
    label_file
) in enumerate(samples):

    image = Image.open(
        image_path
    ).convert("RGB")

    box = get_box(
        label_file,
        cls
    )

    cells = adaptive_target(box)

    print()
    print(
        f"{index + 1:02d} "
        f"{CLASS_NAMES[cls]}"
    )

    print(
        "Grid cells:",
        len(cells)
    )

    visual = draw_target(
        image,
        cells,
        f"ADAPTIVE - {CLASS_NAMES[cls]}"
    )

    output = (
        OUTPUT_DIR /
        f"{index + 1:02d}_{image_path.stem}.jpg"
    )

    visual.save(
        output,
        quality=95
    )

    print(
        "Saved:",
        output
    )


print()
print("=" * 60)
print("ADAPTIVE TARGET TEST COMPLETE")
print("=" * 60)