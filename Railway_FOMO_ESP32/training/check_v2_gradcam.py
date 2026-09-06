# ============================================================
# V2 GRAD-CAM DIAGNOSTIC
# Railway Track Crack Detection
#
# PURPOSE:
#   Check whether the existing V2 CNN is actually focusing
#   on useful image regions when predicting a crack.
#
# IMPORTANT:
#   - NO TRAINING
#   - NO WEIGHT CHANGES
#   - NO V2 modification
#   - Uses the EXISTING V2 checkpoint
#   - Does NOT require torchvision
#
# INPUT:
#   One external/test image
#
# OUTPUT:
#   Original image
#   Grad-CAM heatmap
#   Overlay
#   Prediction probability
# ============================================================

import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    BASE_DIR
    / "diagnostic_crack_binary"
    / "crack_scar_method_v2"
    / "crack_scar_method_v2_best.pth"
)

OUTPUT_DIR = BASE_DIR / "diagnostic_v2_gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

IMAGE_SIZE = 96
GRID_SIZE = 12

ROI_X1 = 0.05
ROI_Y1 = 0.15
ROI_X2 = 0.95
ROI_Y2 = 0.85


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("V2 GRAD-CAM DIAGNOSTIC")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Checkpoint:", CHECKPOINT)


# ============================================================
# V2 MODEL
#
# THIS MUST MATCH THE TRAINING MODEL EXACTLY
# ============================================================

class CrackScarMethodCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.head(x)

        x = F.interpolate(
            x,
            size=(GRID_SIZE, GRID_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# LOAD MODEL
# ============================================================

if not CHECKPOINT.exists():

    print()
    print("ERROR: V2 checkpoint not found:")
    print(CHECKPOINT)
    sys.exit(1)


model = CrackScarMethodCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


# Handle common checkpoint formats

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove possible DataParallel prefix

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()

print()
print("V2 checkpoint loaded successfully.")
print()


# ============================================================
# GRAD-CAM TARGET LAYER
#
# LAST CONVOLUTIONAL FEATURE LAYER
#
# features:
#   [0] Conv
#   ...
#   [24] Conv2d(64,64)
#
# We use the LAST CONV before the head.
# ============================================================

TARGET_LAYER = model.features[12]

print("Grad-CAM target layer:")
print(TARGET_LAYER)
print()


# ============================================================
# HOOK STORAGE
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


forward_handle = TARGET_LAYER.register_forward_hook(
    forward_hook
)

backward_handle = TARGET_LAYER.register_full_backward_hook(
    backward_hook
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    original = image.copy()

    width, height = image.size

    # Fixed railway ROI used by V2

    x1 = int(width * ROI_X1)
    y1 = int(height * ROI_Y1)

    x2 = int(width * ROI_X2)
    y2 = int(height * ROI_Y2)

    roi = image.crop(
        (x1, y1, x2, y2)
    )

    roi = roi.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        roi,
        dtype=np.float32
    ) / 255.0

    # Same basic [0,1] input convention
    # used by the V2 training pipeline.

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    ).float()

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    return original, roi, tensor


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(input_tensor):

    global activations
    global gradients

    activations = None
    gradients = None

    model.zero_grad(set_to_none=True)

    output = model(input_tensor)

    # Convert logits to probabilities

    probabilities = torch.sigmoid(output)

    # Find strongest grid location

    max_probability = probabilities.max()

    max_index = torch.argmax(
        probabilities
    )

    # Backpropagate from strongest prediction

    max_probability.backward()

    if activations is None:
        raise RuntimeError(
            "Grad-CAM activation hook did not receive data."
        )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradient hook did not receive data."
        )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    # activations:
    # [1, C, H, W]

    # gradients:
    # [1, C, H, W]

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(cam)

    # Resize to 96x96

    cam = F.interpolate(
        cam,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0].detach().cpu().numpy()

    # Normalize

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min > 1e-8:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    else:

        cam = np.zeros_like(cam)

    return (
        output.detach(),
        probabilities.detach(),
        cam
    )


# ============================================================
# SAVE DIAGNOSTIC IMAGE
# ============================================================

def save_result(
    original,
    roi,
    cam,
    probability,
    output
):

    roi_array = np.asarray(
        roi
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(15, 5)
    )

    ax1 = fig.add_subplot(1, 3, 1)

    ax1.imshow(roi_array)

    ax1.set_title(
        "V2 Input ROI"
    )

    ax1.axis("off")


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    ax2 = fig.add_subplot(1, 3, 2)

    ax2.imshow(
        roi_array
    )

    ax2.imshow(
        cam,
        cmap="jet",
        alpha=0.55
    )

    ax2.set_title(
        "V2 Grad-CAM"
    )

    ax2.axis("off")


    # --------------------------------------------------------
    # Heatmap only
    # --------------------------------------------------------

    ax3 = fig.add_subplot(1, 3, 3)

    im = ax3.imshow(
        cam,
        cmap="jet"
    )

    ax3.set_title(
        "Activation Heatmap"
    )

    ax3.axis("off")

    fig.colorbar(
        im,
        ax=ax3,
        fraction=0.046,
        pad=0.04
    )


    fig.suptitle(
        f"V2 Grad-CAM | Max Probability: {probability:.4f}"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "v2_gradcam_result.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Max Probability : {probability:.6f}"
    )

    print(
        f"Max Logit       : {output.max().item():.6f}"
    )

    print(
        f"CAM Min         : {cam.min():.6f}"
    )

    print(
        f"CAM Max         : {cam.max():.6f}"
    )

    print(
        f"CAM Mean        : {cam.mean():.6f}"
    )

    print(
        f"CAM Std         : {cam.std():.6f}"
    )

    print()
    print(
        "Saved:"
    )

    print(output_path)

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print()
        print(
            "python training/check_v2_gradcam.py IMAGE_PATH"
        )

        print()
        print("Example:")
        print(
            "python training/check_v2_gradcam.py "
            "test_images/crack.jpg"
        )

        sys.exit(1)


    image_path = Path(
        sys.argv[1]
    )


    if not image_path.exists():

        print()
        print(
            "ERROR: Image not found:"
        )

        print(
            image_path
        )

        sys.exit(1)


    print(
        "Testing image:"
    )

    print(
        image_path
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original, roi, input_tensor = load_image(
        image_path
    )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    output, probabilities, cam = generate_gradcam(
        input_tensor
    )


    max_probability = probabilities.max().item()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_result(
        original,
        roi,
        cam,
        max_probability,
        output
    )


    # --------------------------------------------------------
    # Cleanup hooks
    # --------------------------------------------------------

    forward_handle.remove()
    backward_handle.remove()


if __name__ == "__main__":
    main()