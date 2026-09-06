from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import VALID_IMAGES, VALID_LABELS, IMAGE_SIZE, GRID_SIZE, NUM_CLASSES, CLASS_NAMES
from dataset import RailwayFOMOConverter
from fomo_model import FOMOModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("diagnostic_balanced_tiny/balanced_5ch_model.pth")


def verify_evaluation_math():
    print("=" * 60)
    print("MATHEMATICAL EVALUATION VERIFICATION DIAGNOSTIC")
    print("=" * 60)

    model = FOMOModel(num_classes=NUM_CLASSES).to(DEVICE)
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    converter = RailwayFOMOConverter(VALID_IMAGES, VALID_LABELS)
    num_images = len(converter.samples)
    print(f"Validation samples loaded: {num_images}")

    total_cells = num_images * GRID_SIZE * GRID_SIZE

    actual_object_cells = 0
    actual_background_cells = 0
    predicted_object_cells = 0
    predicted_background_cells = 0

    actual_class_counts = {i: 0 for i in range(NUM_CLASSES)}
    predicted_class_counts = {i: 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for img_path, lbl_path in converter.samples:
            image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            img_arr = np.asarray(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1))).unsqueeze(0).to(DEVICE)

            target = converter.encode_label(lbl_path)  # [12, 12, 5]
            output = model(img_tensor)[0]              # [5, 12, 12]

            obj_prob = torch.sigmoid(output[0])        # [12, 12]
            cls_prob = torch.softmax(output[1:], dim=0)  # [4, 12, 12]

            pred_obj = obj_prob > 0.5
            pred_cls = cls_prob.argmax(dim=0)

            t_obj = target[:, :, 0] > 0.5
            t_cls = target[:, :, 1:].argmax(axis=-1)

            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    # Actual Cell Status
                    if t_obj[y, x]:
                        actual_object_cells += 1
                        actual_class_counts[t_cls[y, x]] += 1
                    else:
                        actual_background_cells += 1

                    # Predicted Cell Status
                    if pred_obj[y, x].item():
                        predicted_object_cells += 1
                        predicted_class_counts[pred_cls[y, x].item()] += 1
                    else:
                        predicted_background_cells += 1

    print("\n" + "=" * 60)
    print("CELL COUNT STATISTICS")
    print("=" * 60)
    print(f"Total grid cells           : {total_cells}")
    print(f"Actual object cells        : {actual_object_cells}")
    print(f"Actual background cells    : {actual_background_cells}")
    print(f"Predicted object cells     : {predicted_object_cells}")
    print(f"Predicted background cells : {predicted_background_cells}")

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION BREAKDOWN")
    print("=" * 60)
    print("Actual:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  - {name:10s}: {actual_class_counts[c]}")

    print("\nPredicted:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  - {name:10s}: {predicted_class_counts[c]}")

    print("\n" + "=" * 60)
    print("MATHEMATICAL CONSERVATION EQUATION CHECKS")
    print("=" * 60)

    eq1_left = actual_object_cells + actual_background_cells
    eq1_valid = (eq1_left == total_cells)
    print(f"Equation 1 (Actual check)     : {eq1_left} == {total_cells} -> Valid: {eq1_valid}")

    eq2_left = predicted_object_cells + predicted_background_cells
    eq2_valid = (eq2_left == total_cells)
    print(f"Equation 2 (Predicted check)  : {eq2_left} == {total_cells} -> Valid: {eq2_valid}")

    sum_actual_classes = sum(actual_class_counts.values())
    eq3_valid = (sum_actual_classes == actual_object_cells)
    print(f"Equation 3 (Class sum check)  : {sum_actual_classes} == {actual_object_cells} -> Valid: {eq3_valid}")

    print("=" * 60)
    if eq1_valid and eq2_valid and eq3_valid:
        print("✓ ALL EVALUATION MATHEMATICAL CHECKS PASSED")
    else:
        print("✗ MATHEMATICAL MISMATCH DETECTED IN EVALUATOR")
    print("=" * 60)


if __name__ == "__main__":
    verify_evaluation_math()