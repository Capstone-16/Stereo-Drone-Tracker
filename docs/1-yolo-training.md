# YOLO Training — Drone Detection Model

This document covers the full process of building the drone detection model used in the stereo tracking pipeline: dataset construction, model selection, training iterations, and the final configuration that was deployed on the Jetson.

The complete training code can be found in [src/yolo_drone_detection_training.ipynb](../src/yolo_drone_detection_training.ipynb). The code be uploaded directly to Google Colab which gives access to different GPUs, including the A100 GPU that was used for training in this project.

---

## 1. Dataset

### Overview

The dataset consists of **3,854 images** across a single class: `drone`. It was assembled from two sources:

- **~1,500 images collected in the field** — footage captured ourselves across varied outdoor conditions, altitudes, and drone types. Frames were extracted from video recordings and manually annotated.
- **~2,354 images from open-source datasets** — sourced from Roboflow Universe to supplement coverage of edge cases, backgrounds, and drone orientations not well represented in our own footage.

The dataset is publicly available on Roboflow Universe:
> 📦 **[Capstone Drone Detection Dataset — Roboflow Universe](https://universe.roboflow.com/senior-projectdrone-dataset/capstone-drone-detection-dataset)**

### Split

The dataset was split into **80% training / 20% validation**. No separate test split was used; evaluation was performed on the validation set.

| Split | Images |
|---|---|
| Train | ~3,083 |
| Validation | ~771 |
| **Total** | **3,854** |

### Annotation

All images are annotated in YOLO format (normalised `[class cx cy w h]` per line). Labels were created and managed via Roboflow, which also handled export versioning.

---

## 2. Model Selection and Training Iterations

### Motivation

We trained over **15 model variants** before settling on the final configuration. The goal was to find the best balance between detection accuracy (especially for small, distant drones) and inference speed fast enough to sustain ~30 FPS on the Jetson Orin Nano.

Three variables were the main focus across iterations:

- **Model size** — YOLOv11 nano, small, and medium were tested
- **Input image size** — 640, 800, and 1280 pixels were tested
- **P2 detection head** — a high-resolution head added to the network to improve sensitivity to small objects (see Section 3)

### Jetson Deployment Comparison

Not all configurations survived the full deployment pipeline. TensorRT export (required for real-time inference on the Jetson) failed or produced unusable engines for some combinations, typically due to memory constraints at larger image sizes with larger models.

The table below summarises the key configurations tested on YOLOv11s, their TensorRT export status, and measured inference latency on the Jetson:

| Model | Image Size | P2 Head | TensorRT Export | Jetson Latency (ms) | FPS |
|---|---|---|---|---|---|
| YOLOv11s | 640 | No | Success | 22.9 | 29.5 |
| YOLOv11s | 640 | Yes | Success | 28.5 | 26.1 |
| YOLOv11s | 800 | No | Success | 23.1 | 30.4 |
| YOLOv11s | 800 | Yes | Success | 29.4 | 25.2 |
| YOLOv11s | 1280 | No | Fail | - | - |

> Latency measured as the YOLO batch inference time (batch=2, both camera frames in one forward pass) on the Jetson Orin Nano 8GB.
> 
> FPS is the number of frames processed per second by the system while in operation and actively detecting at least one drone.

### Key Findings

- **Nano models** were fast but missed small drones at range, producing too many false negatives to be usable for the stereo pipeline.
- **The P2 head** meaningfully improved detection of small and distant drones without a proportional cost in latency, making it the most efficient accuracy gain across all iterations.
- **YOLOv11s at 800px with a P2 head** hit the best accuracy-latency tradeoff and was selected as the final configuration.

---

## 3. The P2 Detection Head

Standard YOLO architectures detect objects at three scales (P3, P4, P5), which are optimised for medium-to-large objects. Drones — particularly at the ranges this system targets (up to 80 m) — are often only a handful of pixels wide. To address this, we injected a **P2 detection head** into the YOLOv11s architecture.

The P2 head extracts features at a shallower layer (stride 4 rather than the standard stride 8 minimum), giving the model access to higher spatial resolution feature maps for small object localisation. The result is a four-scale detector (P2, P3, P4, P5) instead of three.

The custom architecture is defined in `src/yolo11s-p2.yaml` and loaded before training:

```python
model = YOLO('yolo11s-p2.yaml').load('yolo11s.pt')
```

This uses the custom P2 layout but initialises all standard layers from the pretrained `yolo11s.pt` weights, so training benefits from transfer learning while the new P2 head learns from scratch.

---

## 4. Final Training Configuration

The final model was trained for **150 epochs** on the full dataset using the configuration below. All parameters not listed here were left at Ultralytics defaults.

### Model and Data

| Parameter | Value |
|---|---|
| Architecture | YOLOv11s + P2 head (`yolo11s-p2.yaml`) |
| Pretrained weights | `yolo11s.pt` |
| Dataset | 3,854 images, 80/20 train/val split |
| Image size | 800 × 800 |
| Epochs | 150 |
| Batch size | 32 |

### Optimizer

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Initial LR (`lr0`) | 0.001 |
| Final LR (`lrf`) | 0.01 |
| Warmup epochs | 3 |
| Weight decay | 0.0005 |
| Dropout | 0.1 |

### Augmentation

| Parameter | Value | Notes |
|---|---|---|
| Mosaic | 1.0 | Disabled for final 10 epochs (`close_mosaic=10`) |
| Copy-paste | 0.3 | Pastes extra drone instances onto backgrounds — effective for single-class aerial detectors |
| Mixup | 0.1 | Kept light; heavier values hurt single-class localisation |
| Flip LR | 0.5 | |
| Flip UD | 0.0 | Drones do not fly inverted |
| Rotation | 15° | Raised from default — drones appear at varied angles from a fixed stereo rig |
| Scale | 0.6 | Aggressive scale-down mimics drones at long range |
| Translate | 0.1 | |
| HSV hue | 0.015 | |
| HSV saturation | 0.7 | |
| HSV brightness | 0.4 | Lowered from 0.5 — extreme shifts can make small drones invisible |

### Hardware

| Parameter | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| Mixed precision | BF16 (AMP enabled) |
| Workers | 8 |
| Training time | 1h 23m 54s |

### Confidence threshold

Training validation was run at `conf=0.45` to match `CONF_THRESH` in the stereo tracking pipeline, ensuring mAP is evaluated at the same operating point used at inference.

---

## 5. Running the Training Notebook

The complete training code is available as a self-contained Colab notebook:

> 📓 **[`src/yolo_drone_detection_training.ipynb`](../src/yolo_drone_detection_training.ipynb)**

Upload it to [Google Colab](https://colab.research.google.com), switch the runtime to an **A100 GPU** or another preferred GPU (`Runtime → Change runtime type`), and run the cells top to bottom. The notebook covers environment setup, dataset download from Roboflow, dataset verification, training, evaluation, and model export — with instructions inline at each step.


## 6. Results

### Validation Metrics
Metrics for the final model (150 epochs, YOLOv11s + P2, 800px) evaluated at `conf=0.45`:

| Metric | Value |
|---|---|
| mAP50 | 0.9737 |
| mAP50-95 | 0.6563 |
| Precision | 0.9861 |
| Recall | 0.9730 |

### Training Curves
<img width="100%" alt="training_curves" src="https://github.com/user-attachments/assets/05397994-fef2-4623-bcb2-4bd614400641" />

### Trained Weights
The trained weights from the final run are available directly in the repository:

> 📥 **[`best.pt`](../best.pt)** — YOLOv11s + P2 head, trained for 150 epochs on 3,854 images

To use them, either fine-tune further or export directly to TensorRT for Jetson deployment (see Section 7).

---

## 7. Export to TensorRT

Once training is complete, the `.pt` weights must be exported to a TensorRT engine for deployment on the Jetson. The pipeline sends both camera frames as a single batched forward pass, so the engine must be exported with `batch=2`:

```bash
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine', imgsz=800, half=True, dynamic=True, batch=2, device=0)
```

This produces `best.engine`. Place it at `/workspace/best.engine`, or update `MODEL_PATH` in the tuning block at the top of `src/stereo_tracking_code.py`.

> **Note:** TensorRT engines are hardware-specific. An engine exported on a desktop GPU will not run on the Jetson — the export command must be run on the Jetson itself.
