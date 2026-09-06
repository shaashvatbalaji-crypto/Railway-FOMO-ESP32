from pathlib import Path
import numpy as np

from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS,
    GRID_SIZE,
    NUM_CLASSES,
    CLASS_MAP,
)


class RailwayFOMOConverter:

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.samples = []

        for label_file in sorted(self.label_dir.glob("*.txt")):
            image_file = self.find_image(label_file.stem)

            if image_file is not None:
                self.samples.append((image_file, label_file))

    def find_image(self, stem):
        extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        ]

        for ext in extensions:
            image = self.image_dir / (stem + ext)

            if image.exists():
                return image

        return None

    def encode_label(self, label_file):

        # FOMO target:
        #
        # height = GRID_SIZE
        # width  = GRID_SIZE
        # channels = NUM_CLASSES
        #
        # Example:
        # (12, 12, 4)

        target = np.zeros(
            (GRID_SIZE, GRID_SIZE, NUM_CLASSES),
            dtype=np.float32
        )

        text = label_file.read_text().strip()

        if not text:
            return target

        for line in text.splitlines():

            values = line.split()

            if len(values) != 5:
                continue

            try:
                original_class = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])

            except ValueError:
                continue

            # ------------------------------------------------
            # Ignore Rails
            # ------------------------------------------------

            if original_class not in CLASS_MAP:
                continue

            fomo_class = CLASS_MAP[original_class]

            # ------------------------------------------------
            # Safety limits
            # ------------------------------------------------

            x_center = np.clip(
                x_center,
                0.0,
                1.0
            )

            y_center = np.clip(
                y_center,
                0.0,
                1.0
            )

            # ------------------------------------------------
            # Normalized coordinate → FOMO grid
            # ------------------------------------------------

            grid_x = int(
                x_center * GRID_SIZE
            )

            grid_y = int(
                y_center * GRID_SIZE
            )

            grid_x = min(
                max(grid_x, 0),
                GRID_SIZE - 1
            )

            grid_y = min(
                max(grid_y, 0),
                GRID_SIZE - 1
            )

            # ------------------------------------------------
            # Mark object center
            # ------------------------------------------------

            target[
                grid_y,
                grid_x,
                fomo_class
            ] = 1.0

        return target

    def inspect_sample(self, index=0):

        image_file, label_file = self.samples[index]

        target = self.encode_label(label_file)

        print()
        print("=" * 60)
        print("SAMPLE INSPECTION")
        print("=" * 60)

        print("Image:")
        print(image_file)

        print()
        print("YOLO label:")
        print(label_file)

        print()
        print("Original annotation:")

        print(label_file.read_text())

        print()
        print("FOMO target shape:")
        print(target.shape)

        print()
        print("FOMO positive cells:")

        positions = np.argwhere(target > 0)

        if len(positions) == 0:

            print("No defect target.")
            print("(Rails-only image or empty annotation)")

        else:

            for y, x, cls in positions:

                print(
                    f"grid=({x},{y}) "
                    f"class={cls}"
                )

        print()
        print("Target values:")

        print(target[target > 0])


def check_split(name, image_dir, label_dir):

    converter = RailwayFOMOConverter(
        image_dir,
        label_dir
    )

    print()
    print("=" * 60)
    print(f"{name.upper()} DATASET CHECK")
    print("=" * 60)

    print("Images directory:", image_dir)
    print("Labels directory:", label_dir)

    print()
    print("Valid image/label pairs:", len(converter.samples))

    return converter


def main():

    print("=" * 60)
    print("RAILWAY FOMO DATASET CONVERTER")
    print("=" * 60)

    print()
    print("FOMO classes:")

    for original, fomo in CLASS_MAP.items():

        print(
            f"Roboflow class {original} "
            f"-> FOMO class {fomo}"
        )

    print("Roboflow class 1 (Rails) -> IGNORED")

    # --------------------------------------------------------
    # Check training dataset
    # --------------------------------------------------------

    train = check_split(
        "train",
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    # --------------------------------------------------------
    # Check validation dataset
    # --------------------------------------------------------

    valid = check_split(
        "valid",
        VALID_IMAGES,
        VALID_LABELS
    )

    # --------------------------------------------------------
    # Inspect first training sample
    # --------------------------------------------------------

    if len(train.samples) > 0:
        train.inspect_sample(0)

    print()
    print("=" * 60)
    print("DATASET CONVERSION CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()