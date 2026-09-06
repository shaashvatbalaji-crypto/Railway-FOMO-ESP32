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


# ============================================================
# RAILWAY FOMO DATASET CONVERTER
#
# Target format:
#
# [GRID_SIZE, GRID_SIZE, NUM_CLASSES + 1]
#
# Channel 0 -> Objectness
# Channel 1 -> Cracks
# Channel 2 -> Scars
# Channel 3 -> breaks
# Channel 4 -> lightbands
#
# IMPORTANT:
# Only the CENTER CELL of each bounding box is marked
# positive during the initial diagnostic stage.
# ============================================================


class RailwayFOMOConverter:

    def __init__(
        self,
        image_dir,
        label_dir
    ):

        self.image_dir = Path(
            image_dir
        )

        self.label_dir = Path(
            label_dir
        )

        self.samples = []

        for label_file in sorted(
            self.label_dir.glob("*.txt")
        ):

            image_file = self.find_image(
                label_file.stem
            )

            if image_file is not None:

                self.samples.append(
                    (
                        image_file,
                        label_file
                    )
                )

    # ========================================================
    # FIND IMAGE
    # ========================================================

    def find_image(
        self,
        stem
    ):

        extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        ]

        for ext in extensions:

            image = (
                self.image_dir
                /
                (stem + ext)
            )

            if image.exists():

                return image

        return None

    # ========================================================
    # ENCODE YOLO LABEL
    # ========================================================

    def encode_label(
        self,
        label_file
    ):

        # ----------------------------------------------------
        # +1 channel for objectness
        #
        # 0 = objectness
        # 1 = Cracks
        # 2 = Scars
        # 3 = breaks
        # 4 = lightbands
        # ----------------------------------------------------

        target = np.zeros(
            (
                GRID_SIZE,
                GRID_SIZE,
                NUM_CLASSES + 1
            ),
            dtype=np.float32
        )

        text = label_file.read_text().strip()

        if not text:

            return target

        # ====================================================
        # PROCESS EVERY YOLO ANNOTATION
        # ====================================================

        for line in text.splitlines():

            values = line.split()

            if len(values) != 5:

                continue

            try:

                original_class = int(
                    values[0]
                )

                x_center = float(
                    values[1]
                )

                y_center = float(
                    values[2]
                )

                width = float(
                    values[3]
                )

                height = float(
                    values[4]
                )

            except ValueError:

                continue

            # ------------------------------------------------
            # Check class mapping
            # ------------------------------------------------

            if original_class not in CLASS_MAP:

                continue

            fomo_class = CLASS_MAP[
                original_class
            ]

            if not (
                0 <= fomo_class < NUM_CLASSES
            ):

                continue

            # ------------------------------------------------
            # Clamp coordinates
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

            width = np.clip(
                width,
                0.0,
                1.0
            )

            height = np.clip(
                height,
                0.0,
                1.0
            )

            # =================================================
            # CONVERT IMAGE COORDINATE → FOMO GRID
            # =================================================

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

            # =================================================
            # OBJECTNESS
            # =================================================

            target[
                grid_y,
                grid_x,
                0
            ] = 1.0

            # =================================================
            # CLASS
            #
            # fomo_class:
            #
            # 0 -> Cracks
            # 1 -> Scars
            # 2 -> breaks
            # 3 -> lightbands
            #
            # Target channel = fomo_class + 1
            # =================================================

            target[
                grid_y,
                grid_x,
                fomo_class + 1
            ] = 1.0

        return target

    # ========================================================
    # INSPECT SAMPLE
    # ========================================================

    def inspect_sample(
        self,
        index=0
    ):

        if len(self.samples) == 0:

            print(
                "No samples found."
            )

            return

        image_file, label_file = (
            self.samples[index]
        )

        target = self.encode_label(
            label_file
        )

        object_cells = np.count_nonzero(
            target[:, :, 0]
        )

        print(
            "Inspected sample:",
            image_file.name
        )

        print(
            "Object cells:",
            object_cells
        )

        for cls in range(NUM_CLASSES):

            count = np.count_nonzero(
                target[:, :, cls + 1]
            )

            print(
                f"{cls}: {count}"
            )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAILWAY FOMO DATASET TEST")
    print("=" * 60)

    converter = RailwayFOMOConverter(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    print()
    print(
        "Training samples:",
        len(converter.samples)
    )

    if len(converter.samples) > 0:

        target = converter.encode_label(
            converter.samples[0][1]
        )

        print(
            "Target shape:",
            target.shape
        )

        print(
            "Expected:",
            (
                GRID_SIZE,
                GRID_SIZE,
                NUM_CLASSES + 1
            )
        )

        print(
            "Objectness cells:",
            int(
                target[:, :, 0].sum()
            )
        )

        print(
            "Class cells:",
            int(
                target[:, :, 1:].sum()
            )
        )

        if target.shape == (
            GRID_SIZE,
            GRID_SIZE,
            NUM_CLASSES + 1
        ):

            print(
                "✓ TARGET SHAPE CORRECT"
            )

        else:

            print(
                "✗ TARGET SHAPE INCORRECT"
            )

    print()
    print("=" * 60)
    print("DATASET TEST COMPLETE")
    print("=" * 60)