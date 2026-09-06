from pathlib import Path
import math
import statistics


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
    / "railway_track_crack_v9_clean"
)


# ============================================================
# SETTINGS
# ============================================================

SPLITS = [
    "train",
    "valid",
    "test"
]

# Boxes smaller than this are suspicious
MIN_AREA = 0.002

# Boxes larger than this are suspicious
MAX_AREA = 0.80

# Aspect ratio limits
MIN_ASPECT = 0.05
MAX_ASPECT = 20.0

# How far from the dataset median we consider suspicious
OUTLIER_FACTOR = 4.0


# ============================================================
# READ YOLO LABEL
# ============================================================

def read_label_file(label_path):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                continue

            try:

                class_id = int(parts[0])

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                continue

            boxes.append(
                {
                    "class": class_id,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                    "line": line_number
                }
            )

    return boxes


# ============================================================
# CHECK SINGLE BOX
# ============================================================

def check_box(
    box
):

    problems = []

    xc = box["xc"]
    yc = box["yc"]
    w = box["w"]
    h = box["h"]

    # --------------------------------------------------------
    # Coordinate range
    # --------------------------------------------------------

    if not (
        0.0 <= xc <= 1.0
    ):

        problems.append(
            "x-center outside [0,1]"
        )

    if not (
        0.0 <= yc <= 1.0
    ):

        problems.append(
            "y-center outside [0,1]"
        )

    if not (
        0.0 < w <= 1.0
    ):

        problems.append(
            "width outside (0,1]"
        )

    if not (
        0.0 < h <= 1.0
    ):

        problems.append(
            "height outside (0,1]"
        )

    # --------------------------------------------------------
    # Bounding box boundaries
    # --------------------------------------------------------

    x1 = xc - w / 2
    x2 = xc + w / 2

    y1 = yc - h / 2
    y2 = yc + h / 2

    tolerance = 1e-6

    if x1 < -tolerance:

        problems.append(
            "box extends beyond left edge"
        )

    if x2 > 1.0 + tolerance:

        problems.append(
            "box extends beyond right edge"
        )

    if y1 < -tolerance:

        problems.append(
            "box extends beyond top edge"
        )

    if y2 > 1.0 + tolerance:

        problems.append(
            "box extends beyond bottom edge"
        )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area = w * h

    if area < MIN_AREA:

        problems.append(
            f"very small area={area:.6f}"
        )

    if area > MAX_AREA:

        problems.append(
            f"very large area={area:.6f}"
        )

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect = (
        w / h
        if h > 0
        else float("inf")
    )

    if aspect < MIN_ASPECT:

        problems.append(
            f"extreme aspect ratio={aspect:.2f}"
        )

    if aspect > MAX_ASPECT:

        problems.append(
            f"extreme aspect ratio={aspect:.2f}"
        )

    return problems


# ============================================================
# COLLECT DATA
# ============================================================

def collect_split(
    split
):

    image_dir = (
        DATASET_DIR
        / split
        / "images"
    )

    label_dir = (
        DATASET_DIR
        / split
        / "labels"
    )

    all_boxes = []

    image_count = 0
    labeled_images = 0
    background_images = 0

    suspicious = []

    for image_path in sorted(
        image_dir.iterdir()
    ):

        if not image_path.is_file():
            continue

        image_count += 1

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = read_label_file(
            label_path
        )

        if len(boxes) == 0:

            background_images += 1

            continue

        labeled_images += 1

        for box_index, box in enumerate(
            boxes,
            start=1
        ):

            problems = check_box(
                box
            )

            area = (
                box["w"]
                * box["h"]
            )

            aspect = (
                box["w"]
                / box["h"]
            )

            all_boxes.append(
                {
                    "file": image_path.name,
                    "box_index": box_index,
                    "xc": box["xc"],
                    "yc": box["yc"],
                    "w": box["w"],
                    "h": box["h"],
                    "area": area,
                    "aspect": aspect
                }
            )

            if problems:

                suspicious.append(
                    {
                        "file": image_path.name,
                        "box_index": box_index,
                        "problems": problems,
                        "box": box
                    }
                )

    return {
        "images": image_count,
        "labeled": labeled_images,
        "background": background_images,
        "boxes": all_boxes,
        "suspicious": suspicious
    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    split,
    result
):

    boxes = result["boxes"]

    print()
    print(
        "=" * 70
    )

    print(
        f"{split.upper()} ANNOTATION VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Images             : "
        f"{result['images']}"
    )

    print(
        f"Images with cracks  : "
        f"{result['labeled']}"
    )

    print(
        f"Background images   : "
        f"{result['background']}"
    )

    print(
        f"Crack boxes         : "
        f"{len(boxes)}"
    )

    if not boxes:

        return

    widths = [
        b["w"]
        for b in boxes
    ]

    heights = [
        b["h"]
        for b in boxes
    ]

    areas = [
        b["area"]
        for b in boxes
    ]

    aspects = [
        b["aspect"]
        for b in boxes
    ]

    x_centers = [
        b["xc"]
        for b in boxes
    ]

    y_centers = [
        b["yc"]
        for b in boxes
    ]

    print()

    print(
        "BOX SIZE"
    )

    print(
        f"  Width  min : "
        f"{min(widths):.4f}"
    )

    print(
        f"  Width  max : "
        f"{max(widths):.4f}"
    )

    print(
        f"  Width  avg : "
        f"{statistics.mean(widths):.4f}"
    )

    print(
        f"  Height min : "
        f"{min(heights):.4f}"
    )

    print(
        f"  Height max : "
        f"{max(heights):.4f}"
    )

    print(
        f"  Height avg : "
        f"{statistics.mean(heights):.4f}"
    )

    print()

    print(
        "BOX AREA"
    )

    print(
        f"  Min : "
        f"{min(areas):.6f}"
    )

    print(
        f"  Max : "
        f"{max(areas):.6f}"
    )

    print(
        f"  Avg : "
        f"{statistics.mean(areas):.6f}"
    )

    print(
        f"  Median : "
        f"{statistics.median(areas):.6f}"
    )

    print()

    print(
        "ASPECT RATIO"
    )

    print(
        f"  Min : "
        f"{min(aspects):.3f}"
    )

    print(
        f"  Max : "
        f"{max(aspects):.3f}"
    )

    print(
        f"  Median : "
        f"{statistics.median(aspects):.3f}"
    )

    print()

    print(
        "CRACK CENTER"
    )

    print(
        f"  X min : "
        f"{min(x_centers):.4f}"
    )

    print(
        f"  X max : "
        f"{max(x_centers):.4f}"
    )

    print(
        f"  X avg : "
        f"{statistics.mean(x_centers):.4f}"
    )

    print(
        f"  Y min : "
        f"{min(y_centers):.4f}"
    )

    print(
        f"  Y max : "
        f"{max(y_centers):.4f}"
    )

    print(
        f"  Y avg : "
        f"{statistics.mean(y_centers):.4f}"
    )

    print()

    if len(result["suspicious"]) == 0:

        print(
            "✓ No obviously suspicious boxes detected."
        )

    else:

        print(
            f"⚠ Suspicious boxes: "
            f"{len(result['suspicious'])}"
        )


# ============================================================
# PRINT SUSPICIOUS BOXES
# ============================================================

def print_suspicious(
    split,
    result,
    maximum=30
):

    suspicious = result[
        "suspicious"
    ]

    if not suspicious:

        return

    print()
    print(
        "-" * 70
    )

    print(
        f"SUSPICIOUS {split.upper()} BOXES"
    )

    print(
        "-" * 70
    )

    for item in suspicious[
        :maximum
    ]:

        print()

        print(
            item["file"]
        )

        print(
            f"Box #{item['box_index']}"
        )

        print(
            "Problems:"
        )

        for problem in item[
            "problems"
        ]:

            print(
                f"  - {problem}"
            )

        box = item["box"]

        print(
            "Values:"
        )

        print(
            f"  xc={box['xc']:.4f} "
            f"yc={box['yc']:.4f} "
            f"w={box['w']:.4f} "
            f"h={box['h']:.4f}"
        )


# ============================================================
# CROSS-SPLIT CHECK
# ============================================================

def cross_split_check(
    results
):

    print()
    print(
        "=" * 70
    )

    print(
        "CROSS-SPLIT ANNOTATION CHECK"
    )

    print(
        "=" * 70
    )

    total_boxes = 0

    for split in SPLITS:

        count = len(
            results[split]["boxes"]
        )

        total_boxes += count

        print(
            f"{split:>5}: "
            f"{count} crack boxes"
        )

    print()

    print(
        f"TOTAL: "
        f"{total_boxes} crack boxes"
    )


# ============================================================
# FINAL VERDICT
# ============================================================

def final_verdict(
    results
):

    total_suspicious = sum(
        len(
            results[split][
                "suspicious"
            ]
        )
        for split in SPLITS
    )

    total_boxes = sum(
        len(
            results[split]["boxes"]
        )
        for split in SPLITS
    )

    print()
    print(
        "=" * 70
    )

    print(
        "V9 ANNOTATION VERIFICATION RESULT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Total crack boxes : "
        f"{total_boxes}"
    )

    print(
        f"Suspicious boxes  : "
        f"{total_suspicious}"
    )

    print()

    if total_suspicious == 0:

        print(
            "✓ STRUCTURAL LABEL CHECK PASSED"
        )

        print()

        print(
            "The V9 annotations are numerically "
            "consistent and suitable for training."
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "This verifies annotation geometry, "
            "not whether every box semantically "
            "matches a visible crack."
        )

        print()

        print(
            "Next step:"
        )

        print(
            "Train V9."
        )

    else:

        print(
            "⚠ REVIEW SUSPICIOUS BOXES BEFORE TRAINING"
        )

        print()

        print(
            "The dataset is not necessarily broken."
        )

        print(
            "The suspicious boxes should be inspected "
            "visually before training."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "CRACK V9 ANNOTATION VERIFICATION"
    )

    print(
        "AUTOMATIC STRUCTURAL CHECK"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Dataset:"
    )

    print(
        DATASET_DIR
    )

    if not DATASET_DIR.exists():

        print()

        print(
            "ERROR: V9 dataset does not exist."
        )

        return

    results = {}

    for split in SPLITS:

        result = collect_split(
            split
        )

        results[split] = result

        print_statistics(
            split,
            result
        )

        print_suspicious(
            split,
            result
        )

    cross_split_check(
        results
    )

    final_verdict(
        results
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ANNOTATION VERIFICATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()