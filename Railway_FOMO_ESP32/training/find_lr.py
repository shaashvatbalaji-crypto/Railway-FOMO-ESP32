from pathlib import Path
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_IMAGES, TRAIN_LABELS, IMAGE_SIZE, NUM_CLASSES
from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel
from fomo_loss import FOMOLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_optimal_lr():
    print("=" * 60)
    print("RUNNING LEARNING RATE RANGE TEST")
    print("=" * 60)

    converter = RailwayFOMOConverter(TRAIN_IMAGES, TRAIN_LABELS)
    
    # Load a small subset of data for fast scanning
    samples = converter.samples[:64]
    dataset = []
    for img_path, lbl_path in samples:
        image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_arr = np.asarray(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1)))
        target_arr = converter.encode_label(lbl_path)
        dataset.append((img_tensor, torch.from_numpy(target_arr)))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = FOMOModel(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FOMOLoss(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Setup optimizer with an initial tiny learning rate
    start_lr = 1e-7
    end_lr = 10.0
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    num_batches = len(dataloader)
    if num_batches == 0:
        print("No batches available for LR find.")
        return

    mult = (end_lr / start_lr) ** (1.0 / num_batches)
    lr = start_lr
    
    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    for images, targets in dataloader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Stop if loss explodes
        if loss.item() > 4 * best_loss and len(losses) > 5:
            print("Loss exploded, stopping early.")
            break
            
        if loss.item() < best_loss:
            best_loss = loss.item()

        losses.append(loss.item())
        lrs.append(lr)

        loss.backward()
        optimizer.step()

        # Exponentially increase learning rate for next batch
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Find the lr with the steepest negative gradient (sharpest descent)
    losses = np.array(losses)
    lrs = np.array(lrs)
    
    if len(losses) > 2:
        gradients = np.gradient(losses)
        best_idx = np.argmin(gradients)
        optimal_lr = lrs[best_idx]
        
        print("\n" + "=" * 60)
        print(f"RECOMMENDED OPTIMAL LEARNING RATE: {optimal_lr:.5f}")
        print("=" * 60)
    else:
        print("Not enough iterations recorded. Try increasing dataset subset size.")

if __name__ == "__main__":
    find_optimal_lr()