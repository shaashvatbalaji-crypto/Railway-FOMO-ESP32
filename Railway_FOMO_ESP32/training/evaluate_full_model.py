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
CHECKPOINT_PATH = Path("diagnostic_balanced_tiny/balanced_5ch_model.pth")  # Or your full training checkpoint


def evaluate():
    print("=" * 60)
    print("FULL 5-CHANNEL MODEL EVALUATION")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found at {CHECKPOINT_PATH}. Train or save a model first.")
        return

    model = FOMOModel(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    converter = RailwayFOMOConverter(VALID_IMAGES, VALID_LABELS)
    print(f"Validation samples loaded: {len(converter.samples)}")

    # Metrics storage per class: [TP, FP, FN]
    class_metrics = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(NUM_CLASSES)}
    background_fps = 0

    with torch.no_grad():
        for img_path, lbl_path in converter.samples:
            image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            img_arr = np.asarray(image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img_arr, (2, 0, 1))).unsqueeze(0).to(DEVICE)

            target = converter.encode_label(lbl_path)  # [12, 12, 5] (NumPy array)
            
            output = model(img_tensor)[0]  # [5, 12, 12]
            obj_prob = torch.sigmoid(output[0])  # [12, 12]
            cls_prob = torch.softmax(output[1:], dim=0)  # [4, 12, 12]

            pred_obj = obj_prob > 0.5
            pred_cls = cls_prob.argmax(dim=0)

            t_obj = target[:, :, 0] > 0.5
            t_cls = target[:, :, 1:].argmax(axis=-1)  # Fixed to use NumPy axis=-1

            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    p_obj = pred_obj[y, x].item()
                    p_c = pred_cls[y, x].item()
                    
                    real_obj = t_obj[y, x]
                    real_c = t_cls[y, x]

                    if p_obj and not real_obj:
                        background_fps += 1
                        class_metrics[p_c]["fp"] += 1
                    elif not p_obj and real_obj:
                        class_metrics[real_c]["fn"] += 1
                    elif p_obj and real_obj:
                        if p_c == real_c:
                            class_metrics[real_c]["tp"] += 1
                        else:
                            class_metrics[real_c]["fn"] += 1
                            class_metrics[p_c]["fp"] += 1

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Background False Positives: {background_fps}\n")

    macro_f1 = 0.0
    for c, name in enumerate(CLASS_NAMES):
        tp = class_metrics[c]["tp"]
        fp = class_metrics[c]["fp"]
        fn = class_metrics[c]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        macro_f1 += f1

        print(f"Class: {name}")
        print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")

    print(f"Macro F1 Score: {macro_f1 / NUM_CLASSES:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()