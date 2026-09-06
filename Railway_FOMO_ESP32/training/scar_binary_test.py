from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_IMAGES, TRAIN_LABELS, IMAGE_SIZE, GRID_SIZE, CLASS_MAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("diagnostic_scar_binary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DedicatedScarBinaryCNN(nn.Module):
    """
    A lightweight, dedicated binary CNN matching the FOMO philosophy.
    Input:  [B, 3, 96, 96]
    Output: [B, 1, 12, 12] (Scar probability grid)
    """
    def __init__(self):
        super(DedicatedScarBinaryCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 96x96 -> 48x48
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2: 48x48 -> 24x24
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 24x24 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final 1x1 Conv to map directly to 1 output channel at 12x12 resolution
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.features(x)


class ScarBinaryDataset(Dataset):
    def __init__(self, image_dir, label_dir, max_samples=16):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.samples = []

        for label_file in sorted(self.label_dir.glob("*.txt")):
            image_file = self.find_image(label_file.stem)
            if image_file is not None:
                self.samples.append((image_file, label_file))

        # Restrict to a tiny subset (e.g., 16 images) for proof-of-concept overfitting
        self.samples = self.samples[:max_samples]

    def find_image(self, stem):
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img = self.image_dir / (stem + ext)
            if img.exists():
                return img
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1)))

        # Build binary target map: [12, 12, 1] where Scar = 1, Background = 0
        target = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
        
        text = lbl_path.read_text().strip()
        if text:
            for line in text.splitlines():
                values = line.split()
                if len(values) == 5:
                    original_class = int(values[0])
                    # Assuming class mapping for Scars (adjust ID if needed, typically Scars maps to index 1 or specific class ID)
                    # Here we target class if it matches Scar definition
                    fomo_class = CLASS_MAP.get(original_class, -1)
                    
                    # Let's target Scar specifically (e.g., class index 1 or matching specific class name)
                    # Adjust fomo_class index based on your CLASS_MAP configuration
                    if fomo_class == 1:  # 1 corresponds to Scars in your previous logs
                        xc = np.clip(float(values[1]), 0.0, 1.0)
                        yc = np.clip(float(values[2]), 0.0, 1.0)
                        gx = min(int(xc * GRID_SIZE), GRID_SIZE - 1)
                        gy = min(int(yc * GRID_SIZE), GRID_SIZE - 1)
                        target[gy, gx, 0] = 1.0

        return img_tensor, torch.from_numpy(target).permute(2, 0, 1) # [1, 12, 12]


def compute_scar_pos_weight(dataset):
    total_cells = 0
    scar_cells = 0
    for _, target in dataset:
        total_cells += (GRID_SIZE * GRID_SIZE)
        scar_cells += torch.sum(target > 0.5).item()

    bg_cells = total_cells - scar_cells
    if scar_cells == 0:
        return 1.0
    
    pos_weight = bg_cells / scar_cells
    print(f"[Data Stats] Total cells: {total_cells} | Scar cells: {scar_cells} | Background cells: {bg_cells}")
    print(f"[Data Stats] Computed Scar-specific pos_weight: {pos_weight:.4f}")
    return float(pos_weight)


def run_scar_experiment():
    print("=" * 60)
    print("SCAR BINARY ISOLATION EXPERIMENT (v2)")
    print("=" * 60)

    dataset = ScarBinaryDataset(TRAIN_IMAGES, TRAIN_LABELS, max_samples=16)
    print(f"Loaded tiny subset of {len(dataset)} images for Scar isolation check.")

    pos_val = compute_scar_pos_weight(dataset)
    pos_weight_tensor = torch.tensor([pos_val], dtype=torch.float32).to(DEVICE)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DedicatedScarBinaryCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for images, targets in dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)  # [B, 1, 12, 12]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Evaluate overfit metrics on the subset
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                tp += torch.sum((preds == 1) & (targets == 1)).item()
                fp += torch.sum((preds == 1) & (targets == 0)).item()
                fn += torch.sum((preds == 0) & (targets == 1)).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | TP={tp} FP={fp} FN={fn} | Recall: {recall:.2f} | F1: {f1:.2f}")

    # Save artifact if successful
    torch.save({
        "model_state_dict": model.state_dict(),
        "pos_weight": pos_val
    }, OUTPUT_DIR / "scar_model.pth")
    print("\n[✓] Saved isolated scar model checkpoint to diagnostic_scar_binary/scar_model.pth")


if __name__ == "__main__":
    run_scar_experiment()