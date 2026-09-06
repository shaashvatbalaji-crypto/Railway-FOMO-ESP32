# 🚆 Railway FOMO ESP32

**AI-based Railway Rail Scar Detection using Edge Computing**

A lightweight railway rail-surface inspection prototype using an **ESP32-CAM**, **ESP32 DevKit V1**, and a trained **FOMO-style spatial detection model**.

The current model accepts a **96 × 96 RGB image** and produces a **12 × 12 spatial probability grid** to identify regions that may contain a rail scar.

> **Current focus:** Scar detection only. Crack and other defect-class experiments are not part of the final project.

![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![Platform](https://img.shields.io/badge/Platform-ESP32-blue)
![AI](https://img.shields.io/badge/AI-FOMO-green)
![Language](https://img.shields.io/badge/Language-Python-yellow)

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Objective](#-objective)
- [Model Approach](#-model-approach)
- [Hybrid Target Strategy](#-hybrid-target-strategy)
- [Selected Model](#-selected-model)
- [External Validation](#-external-validation)
- [Hardware](#-hardware)
- [Software](#-software)
- [Project Structure](#-project-structure)
- [PC Reference Inference](#-pc-reference-inference)
- [Real VS Code / Terminal Results](#️-real-vs-code--terminal-results)
- [Why FOMO?](#-why-fomo)
- [Target Edge Deployment](#-target-edge-deployment)
- [Current Status](#-current-status)
- [Limitations](#️-limitations)
- [Model and Dataset Notes](#-model-and-dataset-notes)
- [Future Improvements](#-future-improvements)
- [Project](#-project)
- [Disclaimer](#-disclaimer)

---

## ⭐ Key Features

- 🚆 Railway rail-surface scar detection
- 📷 ESP32-CAM based image capture
- 🧠 Lightweight FOMO-style spatial detection
- ⚡ Compact **96 × 96** model input
- 🗺️ **12 × 12** spatial probability grid
- 📍 Spatial localization of high-response regions
- 🔌 Designed for ESP32 edge deployment
- 💻 PC reference inference for model validation
- 📊 Annotated inference results for visual inspection

---

## 📌 Project Overview

Railway rails are exposed to mechanical loads, friction, weather, and environmental conditions. Surface abnormalities can develop over time and require regular inspection.

This project explores a compact computer-vision approach for railway rail inspection. An ESP32-CAM is intended to capture a close-up image of the rail surface, while an ESP32 DevKit V1 is planned as the edge-processing controller.

The current PC reference pipeline demonstrates the model inference and decision logic before final MCU deployment.

---

## 🏗️ System Architecture

### Overall concept

```text
Railway Rail
     ↓
ESP32-CAM
     ↓
Image Capture
     ↓
ESP32 DevKit V1
     ↓
96 × 96 Preprocessing
     ↓
Trained Scar FOMO Model
     ↓
12 × 12 Probability Grid
     ↓
Maximum Probability
     ↓
Threshold = 0.90
     ↓
┌───────────────────┬───────────────────┐
│ SCAR DETECTED     │ NORMAL TRACK      │
└───────────────────┴───────────────────┘
```

### Runtime concept

```text
ESP32-CAM
    │
    │ Captured image
    ▼
ESP32 DevKit V1
    │
    ▼
96 × 96 preprocessing
    │
    ▼
Scar FOMO inference
    │
    ▼
12 × 12 probability grid
    │
    ▼
Maximum probability
    │
    ▼
Decision
```

The **target system** is designed to perform inference locally on the ESP32 DevKit V1 rather than depending on a continuous internet connection.

---

## 🎯 Objective

The main objective is to develop a lightweight railway inspection prototype that can:

- Capture railway rail images using an ESP32-CAM.
- Process the image at 96 × 96 resolution.
- Use a trained Scar detection model.
- Produce a 12 × 12 spatial probability map.
- Identify the strongest response region.
- Classify the input as **SCAR DETECTED** or **NORMAL TRACK**.
- Provide a foundation for future real-time railway inspection and alert systems.

---

## 🧠 Model Approach

The project uses a lightweight **FOMO-style spatial detection architecture**.

### Input

```text
RGB image
96 × 96 pixels
3 channels
```

### Output

```text
1 × 12 × 12 probability grid
```

Each grid cell represents the model's confidence that a Scar-related feature is present in that spatial region.

The current PC reference decision rule is:

```text
max_probability ≥ 0.90
        ↓
SCAR DETECTED

max_probability < 0.90
        ↓
NORMAL TRACK
```

The strongest grid cell is also reported to provide a spatial indication of where the model's highest response occurs.

---

## 🔬 Hybrid Target Strategy

During training, a **hybrid target strategy** was used to teach the model the spatial neighborhood of a Scar.

The training target gives the central Scar location the strongest target value and assigns lower target values to neighboring cells. This encourages the model to learn a spatial response rather than treating the task only as whole-image classification.

The distinction between training and deployment is:

```text
TRAINING
Hybrid target strategy
        ↓
Trained Scar FOMO model
        ↓

DEPLOYMENT / INFERENCE
Trained Scar FOMO model
        ↓
12 × 12 probability grid
        ↓
Maximum probability
        ↓
Scar / Normal decision
```

> **Important:** The ESP32 does not generate hybrid training targets during inference. The hybrid target strategy is part of model training; the trained model is what performs inference.

---

## 📦 Selected Model

The current selected Scar checkpoint is:

```text
diagnostic_scar_binary/scar_mild_augmented_best.pth
```

Reference model information:

| Parameter | Value |
|---|---|
| Model | `DedicatedScarBinaryCNN` |
| Input | `3 × 96 × 96` |
| Output | `1 × 12 × 12` |
| Selected checkpoint | `scar_mild_augmented_best.pth` |
| Training epoch | 17 |
| Reference threshold | `0.90` |
| Training approach | Hybrid-target-trained Scar model |

The `.pth` checkpoint is currently used for **PC-side reference inference**. Final MCU deployment requires a model representation and inference implementation suitable for the target ESP32 hardware.

---

## 🧪 External Validation

The selected checkpoint was tested using external railway images **without retraining and without changing the checkpoint**.

### Test summary

| Test | Image Type | Max Probability | Detected Cells | Result |
|---|---|---:|---:|---|
| 1 | External Scar | **0.9983** | 26 | 🔴 **SCAR DETECTED** |
| 2 | Normal Full Track | **0.9956** | 11 | ⚠️ **False Positive** |
| 3 | Normal Close-Up Rail | **0.8558** | 0 | 🟢 **NORMAL TRACK** |

### Key observation

The tests show that **camera viewpoint and image composition are important** for the current model.

The wide full-track image produced a false positive, while the close-up normal rail image remained below the `0.90` threshold.

Therefore, the planned camera setup should focus on a **close-up inspection view of the rail surface** rather than a wide railway scene.

These tests are reference validation only and are not sufficient to establish real-world railway safety performance.

---

## 🔧 Hardware

### Planned hardware

- **ESP32-CAM**
  - Captures railway rail images.
- **ESP32 DevKit V1**
  - Target edge-processing controller.
- Railway rail inspection setup
  - Used to maintain an appropriate camera position and inspection distance.

### Planned communication

```text
ESP32-CAM
    │
    │ Image
    ▼
ESP32 DevKit V1
    │
    ▼
Scar FOMO inference
    │
    ▼
Decision / Alert
```

---

## 💻 Software

### Development tools

- Python
- PyTorch
- CUDA
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

### Development environment used for reference inference

```text
GPU      : NVIDIA GeForce RTX 3050 6GB Laptop GPU
PyTorch  : 2.11.0+cu130
CUDA     : Available
```

The PC environment is used for model development, validation, visualization, and reference inference before edge deployment.

---

## 📁 Project Structure

```text
Railway_FOMO_ESP32/
│
├── diagnostic_scar_binary/
│   ├── best_scar_model.pth
│   ├── scar_augmented_best.pth
│   ├── scar_mild_augmented_best.pth
│   └── scar_model.pth
│
├── training/
│   ├── fomo_model.py
│   ├── fomo_loss.py
│   ├── dataset.py
│   ├── compare_targets.py
│   ├── scar_augmented_train.py
│   ├── scar_binary_test.py
│   ├── scar_binary_v2.py
│   ├── scar_esp32_input_test.py
│   ├── scar_full_train.py
│   ├── scar_image_test.py
│   ├── scar_visual_test.py
│   ├── test_best_scar.py
│   ├── test_scar_validation.py
│   └── scar_mild_augmented_train.py
│
├── esp32/
│   ├── model_data.h
│   └── railway_fomo.ino
│
├── evaluation/
│
├── final_integration.py
│
└── README.md
```

> The exact contents may change as the ESP32 integration is completed.

---

## ▶️ PC Reference Inference

Activate the Python environment:

```bash
cd Railway_FOMO_ESP32
source Railway_FOMO_ESP32/.venv/bin/activate
```

Run inference:

```bash
python final_integration.py image.png
```

Example:

```bash
python final_integration.py rail_real_test.png
```

The reference script:

1. Loads the selected Scar checkpoint.
2. Reads the input image.
3. Resizes it to 96 × 96.
4. Runs Scar FOMO inference.
5. Produces the 12 × 12 probability grid.
6. Finds the strongest grid cell.
7. Applies the 0.90 threshold.
8. Generates an annotated result image.

Results are saved under:

```text
final_integration_results/
```

---

## 🖥️ Real VS Code / Terminal Results

The following are **real outputs from the VS Code/WSL terminal**, obtained using the selected checkpoint.

### Test 1 — External Scar Image

Command:

```bash
python final_integration.py rail_real_test.png
```

Relevant actual output:

```text
MODEL
------------------------------------------------------------------------
Checkpoint : .../diagnostic_scar_binary/scar_mild_augmented_best.pth
Device     : cuda
GPU        : NVIDIA GeForce RTX 3050 6GB Laptop GPU
Model      : DedicatedScarBinaryCNN
Input      : 3 x 96 x 96
Output     : 1 x 12 x 12
Training   : Hybrid-target-trained Scar model
Epoch      : 17
Best F1    : 0.15181518151815185
Threshold  : 0.9

MODEL LOADED SUCCESSFULLY

IMAGE INFERENCE
------------------------------------------------------------------------
Image          : rail_real_test.png
Original size  : (500, 374)
Model input    : 96 x 96
FOMO output    : 12 x 12

TOP FOMO CELLS
------------------------------------------------------------------------
01. Grid=(3,7) Prob=0.9983
02. Grid=(3,8) Prob=0.9982
03. Grid=(3,6) Prob=0.9979
04. Grid=(3,9) Prob=0.9968
05. Grid=(3,5) Prob=0.9953

FINAL DECISION
------------------------------------------------------------------------
Status         : SCAR DETECTED
Max probability: 0.9983
Strongest grid : (3,7)
Threshold      : 0.9000
Detected cells : 26
```

### Test 2 — Normal Full-Track Image

Command:

```bash
python final_integration.py normal_track.png
```

Relevant actual output:

```text
IMAGE INFERENCE
------------------------------------------------------------------------
Image          : normal_track.png
Original size  : (600, 900)
Model input    : 96 x 96
FOMO output    : 12 x 12

TOP FOMO CELLS
------------------------------------------------------------------------
01. Grid=(10,1) Prob=0.9956
02. Grid=(6,7) Prob=0.9926
03. Grid=(7,1) Prob=0.9881
04. Grid=(6,8) Prob=0.9878
05. Grid=(6,9) Prob=0.9711

FINAL DECISION
------------------------------------------------------------------------
Status         : SCAR DETECTED
Max probability: 0.9956
Strongest grid : (10,1)
Threshold      : 0.9000
Detected cells : 11
```

This test produced a false positive on a wide railway scene.

### Test 3 — Normal Close-Up Rail Image

Command:

```bash
python final_integration.py normal_closeup.png
```

Relevant actual output:

```text
IMAGE INFERENCE
------------------------------------------------------------------------
Image          : normal_closeup.png
Original size  : (1500, 1101)
Model input    : 96 x 96
FOMO output    : 12 x 12

TOP FOMO CELLS
------------------------------------------------------------------------
01. Grid=(9,2) Prob=0.8558
02. Grid=(3,5) Prob=0.6265
03. Grid=(3,2) Prob=0.3354
04. Grid=(8,2) Prob=0.3337
05. Grid=(5,3) Prob=0.3248

FINAL DECISION
------------------------------------------------------------------------
Status         : NORMAL TRACK
Max probability: 0.8558
Strongest grid : (9,2)
Threshold      : 0.9000
Detected cells : 0
```

### Interpretation

These real tests demonstrate that:

- The selected checkpoint can produce a strong response on the external Scar image.
- A wide normal railway scene can produce a false positive.
- A close-up normal rail image produced a result below the reference threshold.
- The camera viewpoint and inspection framing are therefore important parts of the overall system design.

---

## 🔬 Why FOMO?

Traditional object-detection models can be relatively heavy for small microcontrollers.

FOMO is attractive for this project because it represents spatial object presence using a compact grid.

Instead of:

```text
Large image
    ↓
Heavy object detector
    ↓
Bounding boxes
```

the project uses:

```text
96 × 96 image
    ↓
Lightweight FOMO-style model
    ↓
12 × 12 spatial map
    ↓
Strongest response region
```

This makes the approach suitable for exploring edge-AI deployment on resource-constrained hardware.

---

## ⚙️ Target Edge Deployment

The target architecture is:

```text
                ┌─────────────────┐
                │   ESP32-CAM     │
                │ Image Capture   │
                └────────┬────────┘
                         │
                         │ Image
                         ▼
                ┌─────────────────┐
                │ ESP32 DevKit V1 │
                │ Preprocessing   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Scar FOMO Model │
                │    96 × 96      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   12 × 12 Grid  │
                └────────┬────────┘
                         │
                         ▼
                  Max Probability
                         │
                    ┌────┴────┐
                    │ 0.90 ?  │
                    └────┬────┘
                       /   \
                     YES    NO
                      │      │
                      ▼      ▼
                   SCAR    NORMAL
```

> **Current status:** The PC-side reference inference is demonstrated. The complete ESP32-CAM → ESP32 DevKit → on-device inference pipeline is still under development.

---

## 🚧 Current Status

### Completed

- [x] Scar-focused model development
- [x] Hybrid-target training approach
- [x] Scar checkpoint selection
- [x] 96 × 96 input pipeline
- [x] 12 × 12 FOMO output
- [x] PC-side reference inference
- [x] External Scar image testing
- [x] External normal-image testing
- [x] Annotated FOMO result generation
- [x] Initial threshold-based decision logic
- [x] ESP32 firmware/model integration files started

### In Progress

- [ ] Final ESP32-CAM image capture integration
- [ ] ESP32-CAM → ESP32 DevKit image transfer
- [ ] MCU-compatible model conversion
- [ ] On-device inference validation
- [ ] Final alert/output integration
- [ ] Real railway-track field testing

---

## ⚠️ Limitations

This is an **educational/research prototype** and should not be treated as a certified railway safety system.

Current limitations include:

- Model behavior depends on camera viewpoint and image composition.
- A wide railway scene can produce false positive responses.
- Close-up inspection views are more appropriate for the current model.
- External-image testing is limited compared with a large independent validation dataset.
- The current PC results do not establish real-world railway reliability.
- The final ESP32 implementation has not yet completed on-device validation.
- Real railway deployment would require extensive validation, safety analysis, and field testing.

---

## 🔒 Model and Dataset Notes

Only files required for the final Scar project should be committed to the repository.

Avoid committing:

- Virtual environments
- Python cache files
- Temporary logs
- Generated inference results
- Large temporary datasets
- Unrelated experiments

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
*.log
final_integration_results/
*.tmp
```

If model checkpoints become too large for normal Git hosting, Git LFS or another model-storage solution can be considered.

---

## 🚀 Future Improvements

Possible future improvements include:

- Better controlled rail-surface image capture.
- Fixed camera mounting and inspection distance.
- Real-time ESP32 inference.
- Local LED/buzzer warning.
- Display-based status indication.
- Image logging for detected abnormalities.
- Larger and more diverse independent validation.
- Optimized/quantized MCU model.
- Monitoring dashboard or remote reporting.
- Real railway-track field testing.

---

## 👥 Project

### Railway FOMO ESP32

A student prototype exploring lightweight computer vision and edge AI for railway rail-surface inspection.

### Core Technologies

```text
Computer Vision
       +
FOMO
       +
PyTorch
       +
ESP32-CAM
       +
ESP32 DevKit V1
       ↓
Edge Railway Inspection Prototype
```

---

## 📜 Disclaimer

This project is developed as an educational and research prototype. It is **not intended to replace certified railway inspection procedures, professional inspection equipment, or railway safety systems**.
