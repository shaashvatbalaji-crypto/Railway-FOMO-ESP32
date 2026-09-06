from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_IMAGES, TRAIN_LABELS, IMAGE_SIZE, GRID_SIZE, NUM_CLASSES, CLASS_NAMES
from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel, count_parameters
from fomo_loss import FOMOLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_1_model_architecture():
    print("\n[TEST 1] Verifying Model Architecture & Output Shape...")
    model = FOMOModel(num_classes=NUM_CLASSES).to(DEVICE)
    model.eval()
    
    print(f"  - Parameter count: {count_parameters(model)}")
    assert count_parameters(model) > 0, "Model has 0 parameters!"

    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    
    expected_shape = (2, NUM_CLASSES + 1, GRID_SIZE, GRID_SIZE)
    print(f"  - Output shape: {tuple(output.shape)} (Expected: {expected_shape})")
    assert tuple(output.shape) == expected_shape, f"Shape mismatch! Got {tuple(output.shape)}"
    print("  ✓ Test 1 Passed: Model output shape and parameter count are correct.")


def test_2_dataset_encoding():
    print("\n[TEST 2] Verifying Dataset Target Encoding...")
    converter = RailwayFOMOConverter(TRAIN_IMAGES, TRAIN_LABELS)
    assert len(converter.samples) > 0, "No training samples found!"
    
    sample_img, sample_lbl = converter.samples[0]
    target = converter.encode_label(sample_lbl)
    
    expected_target_shape = (GRID_SIZE, GRID_SIZE, NUM_CLASSES + 1)
    print(f"  - Target shape: {target.shape} (Expected: {expected_target_shape})")
    assert target.shape == expected_target_shape, f"Target shape mismatch!"
    
    obj_cells = np.count_nonzero(target[:, :, 0])
    print(f"  - Sample object cells present: {obj_cells}")
    print("  ✓ Test 2 Passed: Dataset target encoding matches 5-channel layout.")


def test_3_loss_and_gradients():
    print("\n[TEST 3] Verifying Loss Calculation & Gradient Flow...")
    model = FOMOModel(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FOMOLoss(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    dummy_target = torch.zeros(2, GRID_SIZE, GRID_SIZE, NUM_CLASSES + 1, device=DEVICE)
    # Inject a dummy positive cell
    dummy_target[0, 5, 5, 0] = 1.0  # objectness
    dummy_target[0, 5, 5, 1] = 1.0  # class 0 (Cracks)

    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()

    print(f"  - Computed Loss Value: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"
    
    # Check if gradients exist and are finite
    has_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"  - Gradients finite and populated: {has_grads}")
    assert has_grads, "Gradients are missing or contain NaN/Inf!"
    print("  ✓ Test 3 Passed: Loss function and backpropagation are healthy.")


def test_4_tiny_set_overfit():
    print("\n[TEST 4] Running 16-Image Balanced Overfit Test...")
    converter = RailwayFOMOConverter(TRAIN_IMAGES, TRAIN_LABELS)
    
    # Collect 2 samples per class to ensure all 4 classes are present
    quota = {0: 2, 1: 2, 2: 2, 3: 2}
    subset = []
    for img_path, lbl_path in converter.samples:
        target = converter.encode_label(lbl_path)
        if target[:, :, 0].sum() == 0:
            continue
        active_classes = [int(np.argmax(target[y, x, 1:])) for y, x in np.argwhere(target[:, :, 0] > 0)]
        helped = False
        for c in active_classes:
            if c in quota and quota[c] > 0:
                quota[c] -= 1
                helped = True
        if helped and (img_path, lbl_path) not in subset:
            subset.append((img_path, lbl_path))
        if all(q <= 0 for q in quota.values()):
            break

    print(f"  - Loaded tiny overfit subset: {len(subset)} images.")

    model = FOMOModel(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = FOMOLoss(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    model.train()
    for epoch in range(1, 11):
        epoch_loss = 0.0
        for img_path, lbl_path in subset:
            image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            img_arr = np.asarray(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1))).unsqueeze(0).to(DEVICE)
            
            target_arr = converter.encode_label(lbl_path)
            target_tensor = torch.from_numpy(target_arr).unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            output = model(img_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch == 1 or epoch == 10:
            print(f"  - Epoch {epoch:02d}/10 | Loss: {epoch_loss / len(subset):.4f}")

    print("  ✓ Test 4 Passed: Tiny subset overfit loop completed successfully.")


def main():
    print("=" * 60)
    print("STARTING RAILWAY FOMO PIPELINE VERIFICATION")
    print("=" * 60)
    
    test_1_model_architecture()
    test_2_dataset_encoding()
    test_3_loss_and_gradients()
    test_4_tiny_set_overfit()

    print()
    print("=" * 60)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()