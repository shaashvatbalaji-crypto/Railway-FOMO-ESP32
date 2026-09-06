from pathlib import Path
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "railway_crack_v12_balanced"

print("=" * 70)
print("PIPELINE IMAGE DIAGNOSTIC")
print("=" * 70)

for split in ["train", "valid", "test"]:

    image_dir = DATASET / split / "images"

    print()
    print(f"===== {split.upper()} =====")

    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png",
            ".bmp", ".webp"
        }
    ])

    for path in images[:5]:

        image = Image.open(path).convert("RGB")

        arr = np.asarray(
            image,
            dtype=np.float32
        )

        print(
            f"{path.name:55s} "
            f"size={image.size} "
            f"mean={arr.mean(axis=(0,1))} "
            f"min={arr.min():.0f} "
            f"max={arr.max():.0f}"
        )

print()
print("=" * 70)
print("EXTERNAL IMAGES")
print("=" * 70)

external_dir = BASE_DIR / "external_test"

for path in sorted(external_dir.iterdir()):

    if path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".webp"
    }:
        continue

    image = Image.open(path).convert("RGB")

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    print(
        f"{path.name:55s} "
        f"size={image.size} "
        f"mean={arr.mean(axis=(0,1))} "
        f"min={arr.min():.0f} "
        f"max={arr.max():.0f}"
    )

print()
print("=" * 70)
