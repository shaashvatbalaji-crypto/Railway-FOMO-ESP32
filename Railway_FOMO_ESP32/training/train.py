import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ------------------------------------------------------------
# Allow imports from training/
# ------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)


from config import (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    IMAGE_SIZE,
    NUM_CLASSES,
)

from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from fomo_loss import FOMOLoss


# ============================================================
# SINGLE-BATCH INTEGRATION TEST
# ============================================================

class FOMOBatchDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, label_dir):

        self.converter = RailwayFOMOConverter(
            image_dir,
            label_dir
        )

    def __len__(self):

        return len(self.converter.samples)

    def __getitem__(self, index):

        image_file, label_file = self.converter.samples[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        from PIL import Image
        import numpy as np

        image = Image.open(
            image_file
        ).convert("RGB")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        image = np.asarray(
            image,
            dtype=np.float32
        )

        # 0-255 → 0-1

        image = image / 255.0

        # HWC → CHW

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # FOMO target
        # ----------------------------------------------------

        target = self.converter.encode_label(
            label_file
        )

        image = torch.from_numpy(
            image
        )

        target = torch.from_numpy(
            target
        )

        return image, target


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RAILWAY FOMO SINGLE-BATCH TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = FOMOBatchDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    print()
    print("Training samples:", len(dataset))

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FOMOModel(
        num_classes=NUM_CLASSES
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = FOMOLoss(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------------
    # Get ONE batch
    # --------------------------------------------------------

    images, targets = next(
        iter(loader)
    )

    print()
    print("Batch images :", tuple(images.shape))
    print("Batch targets:", tuple(targets.shape))

    # --------------------------------------------------------
    # Move to GPU
    # --------------------------------------------------------

    images = images.to(device)
    targets = targets.to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    optimizer.zero_grad()

    predictions = model(
        images
    )

    print(
        "Predictions  :",
        tuple(predictions.shape)
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = criterion(
        predictions,
        targets
    )

    print(
        "Loss         :",
        loss.item()
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # GPU synchronization
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize()

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    print()

    print(
        "Loss finite  :",
        torch.isfinite(loss).item()
    )

    gradient_ok = True

    for parameter in model.parameters():

        if parameter.grad is not None:

            if not torch.isfinite(
                parameter.grad
            ).all():

                gradient_ok = False
                break

    print(
        "Gradients OK :",
        gradient_ok
    )

    print()

    if (
        torch.isfinite(loss).item()
        and gradient_ok
        and predictions.shape == (
            8,
            NUM_CLASSES,
            12,
            12
        )
    ):

        print("✓ SINGLE-BATCH TEST PASSED")

    else:

        print("✗ SINGLE-BATCH TEST FAILED")

    print()
    print("=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()