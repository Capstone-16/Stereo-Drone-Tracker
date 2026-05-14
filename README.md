# Stereo Drone Tracker

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) 
![Platform: NVIDIA Jetson](https://img.shields.io/badge/Platform-NVIDIA%20Jetson-76B900)

> **Real-time 3D detection, tracking, and localisation of drones using a dual IMX477 CSI stereo camera rig, YOLO inference, and Kalman-filtered track management — running entirely on edge hardware.**

<p align="center">
  <img width="1920" height="912" alt="output_perfect" src="https://github.com/user-attachments/assets/fb6bb437-4e10-4442-a535-b1d0cb36c7c6" />
  <br>
</p>

---

## Abstract
The wide availability of consumer UAVs poses an increasing threat to public safety and private areas. Traditional detection methods for such aircraft, such as radar systems, are impractical for civilian deployment due to their extremely high cost and regulatory restrictions. This project presents a self-contained and autonomous system for detecting and tracking multiple drones using a network of synchronized cameras integrated with a Jetson Orin Nano. Camera feeds are processed in real time by a YOLO-based detection and tracking pipeline, followed by stereo triangulation and a Kalman filter for 3D position and speed estimation, and a Hungarian algorithm for consistent multi-drone ID assignment. The system successfully detects multiple drones simultaneously at various ranges. Additionally, it achieves position accuracy within a 5% error while maintaining a stable ID labeling framework with a mean confidence score of 80%, even after deliberate temporary occlusions. Compared with sophisticated drone tracking systems priced at over $95,000, this work demonstrates a cost-effective and legally compliant alternative. It enables civilian drone monitoring using existing camera infrastructure with a budget of ~$550, without the need for specialized radio-frequency equipment.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Performance](#-performance)
- [The Development Process](#-the-development-process)
- [Hardware Prototype](#-hardware-prototype)
- [Demo](#-demo)
- [Launch](#-launch)
- [Repository Structure](#-repository-structure)
- [Documentation](#-documentation)
- [Team](#-team)
- [License](#-license)

---

## 🚁 Overview

Tracking small, fast-moving drones in 3D space is hard — especially without dedicated radar or motion-capture infrastructure. This project tackles that problem using nothing but two cameras and an edge GPU.

A stereo pair of IMX477 sensors feeds live video into a pipeline that detects drones with a TensorRT-accelerated YOLO model, pairs detections across both cameras using epipolar geometry, triangulates their positions in 3D space, and maintains persistent drone identities across frames with a Kalman-filter track manager. The entire pipeline runs on an NVIDIA Jetson at ~27 fps, logging positions and velocities to CSV in real time.


## ✨ Key Features

- **Precise 3D localisation:** Not just "there's a drone in frame" — the system outputs a calibrated (x, y, z) position in metres, updated every frame, by triangulating detections across both cameras using epipolar geometry.

- **Multi-drone tracking without confusion:** Each drone gets a persistent ID the moment it's confirmed. The tracker resolves ambiguity even when drones cross paths, overlap, or briefly share the same region of frame.

- **Zombie re-identification:** If a drone disappears behind an obstacle and re-enters the scene, the system recognises it by extrapolating its last known position and velocity forward in time — restoring the original ID, Kalman state, and tracking history rather than spawning a new one.

- **Kalman-filtered position and velocity:** A per-drone Kalman filter runs a 6-state constant-velocity model, keeping position and velocity estimates alive through detection gaps. Velocity is what drives the zombie extrapolation and the live speed readout shown in each drone's info card.

- **Self-calibrating physical size:** Over its first ~2 seconds, each track accumulates depth-normalised bounding-box measurements and locks a physical size estimate for the drone. This becomes an additional matching cost — helping the tracker distinguish two drones at different distances even when they appear at similar image-plane locations.

- **Fully on-device, no cloud required:** Detection, stereo geometry, track management, and display all run on an NVIDIA Jetson at ~27 fps. No network connection, no inference server — the system is fully self-contained.

## 🛠 Tech Stack

- **Hardware:** Jetson Orin Nano 8GB, 2× Sony IMX477 CSI cameras
- **Software:** Python 3, GStreamer (+ Python GI bindings), OpenCV, NumPy, SciPy
- **AI / ML:** YOLOv11s (TensorRT `.engine`), Ultralytics, PyTorch
- **Calibration:** Custom stereo calibration script (`src/step2_calibrate_stereo.py`) producing an `.npz` with intrinsics, distortion, rectification matrices, and a metre-unit baseline vector

## 📊 Performance

| Metric | Result |
|---|---|
| Pipeline FPS | **27 FPS** |
| YOLO batch inference (batch=2, TensorRT) | **30 ms** |
| End-to-end latency (capture → display) | **60 ms** |
| Detection confidence — nominal | **~80%** |
| 3D position error @ 20 m (50 cm baseline) | **±5%** |
| ID recovery after full occlusion | **≤ 2 frames** |


## 🧠 The Development Process

**The concept** grew from a specific gap: existing drone detection systems either rely on radar — expensive, regulated, and actively emitting — or require cloud connectivity that fails the moment you leave a network. We wanted something passive, off-grid, and deployable anywhere with nothing but power. That constraint pointed toward a stereo camera rig and on-device AI: two IMX477 CSI cameras feeding an NVIDIA Jetson, producing real 3D coordinates with no radiated signals and no internet required.

**The foundation was the detector.** We built a dataset of around 3,800 images — roughly half collected ourselves in the field, the rest sourced from open datasets — and trained multiple iterations of YOLOv11, experimenting with model sizes from nano to medium, different image resolutions, augmentation strategies, and architectural tweaks like adding a P2 detection head for small objects. Getting a model that was both accurate enough to trust and fast enough to run in real time on the Jetson narrowed the options considerably. The final model is a YOLOv11s with a P2 head, exported to a TensorRT engine and running both camera frames as a single batched forward pass to minimise inference latency.

**The hardest part, it turned out, was making the two cameras agree on where things are.** Stereo vision is extremely sensitive to calibration quality, and a single misconfigured parameter in the rectification pipeline was silently causing every stereo pair to fail. The tracker would run, look perfectly normal on screen, and produce no 3D data whatsoever — every drone appeared as two independent single-camera detections with no depth. Diagnosing an invisible failure like that, where nothing crashes and everything looks fine, taught us more about stereo geometry than any tutorial did. We eventually added a startup sanity check that validates the calibration file before the pipeline begins — if it fails, stereo matching is disabled entirely rather than running silently wrong.

**Keeping identities stable** turned out to be its own problem. Drones move fast, disappear behind obstacles, and re-enter the frame from unexpected directions. Our first tracker would re-assign IDs constantly, making trajectory data useless for any downstream analysis. We solved this with a zombie re-identification system: confirmed tracks that go missing are held in a side list, their positions extrapolated forward using last-known velocity, and matched against new detections before a new ID is ever spawned. The result is trajectory data that stays coherent even through deliberate occlusions.


## 🔧 Hardware Prototype

<p align="center">
  <img width="1280" height="682" alt="Hardware_prototype" src="https://github.com/user-attachments/assets/f2ae8646-2171-45f0-a465-a18c073557ee" />
  <br>
  <em>The system includes two fixed camera units (adjustable) connected to NVIDIA Jetson Nano. The Jetson is connected to I/O peripherals: screen, keyboard, and mouse to monitor the run.</em>
</p>


## 🎬 Demo

> Tracking two drones simultaneously outdoors — full stereo 3D localisation, live speed readout, and ID recovery.

<p align="center">
  <video src="https://github.com/user-attachments/assets/5095782d-31fd-41e6-ba26-82f839e8186d" width="900" controls></video>
</p>

**About the system's dashboard:** The interface is a single window with two side-by-side camera panels showing live feeds from both cameras. A thin status bar runs across the middle displaying current FPS, active track count, stereo pair count, and detection counts. A bottom panel displays a stat card for each tracked drone, listing its ID, state, 3D position, depth, distance, speed, detection confidence, and which camera last saw it. Bounding boxes on each feed are colour-coded by tracking state:

- **🟢 Green — Full 3D Stereo:** Both cameras have a detection; the drone has a live triangulated position.
- **🟡 Amber — Single Camera:** Only one camera sees the drone; no depth or 3D position is available.
- **⚪ Grey — Coasting:** The drone was recently lost; the system is predicting its position from the last known trajectory while waiting for it to reappear.

---

## 🚀 Launch

> All steps below are run **on the Jetson** unless noted otherwise.

### Step 1 — Install dependencies

Install GStreamer Python bindings (not available via pip):
```bash
sudo apt install python3-gi python3-gst-1.0 gstreamer1.0-tools
```

Clone the repo and install Python packages:
```bash
git clone https://github.com/Capstone-16/Stereo-Drone-Tracker.git
cd Stereo-Drone-Tracker
pip install -r requirements.txt
```

PyTorch must be installed from NVIDIA's Jetson wheel — **do not use the generic PyPI build**. Follow the official [Jetson PyTorch install guide](https://developer.nvidia.com/embedded/downloads) for your JetPack version.



### Step 2 — Export your YOLO model to TensorRT

The trained weights used in this project are included in the repository as `best.pt`. Copy them to the Jetson's workspace, then run the pre-configured export script:

```bash
cp best.pt /workspace/best.pt
python3 src/export_TensorRT.py
```

This produces `/workspace/best.engine`, which is what the tracker loads. All required export parameters (`batch=2`, `half=True`, etc.) are already set in the script.

> **Want to train your own model instead?** The full training pipeline — dataset preparation, augmentation, model configuration, and export — is in `src/yolo_drone_detection_training.ipynb`. See [docs/1-yolo-training.md](docs/1-yolo-training.md) for the written walkthrough.



### Step 3 — Capture stereo calibration images

Mount both IMX477 cameras in their final positions — **do not move them after this step**. Then run the capture script with a physical checkerboard in view:

```bash
python3 src/step1_capture_calib_images.py
```

Capture at least 20–30 image pairs with the checkerboard at varied distances and angles. Images are saved to `calib/left/` and `calib/right/`.

> For tips on board placement, what makes a good pair, and how to read the quality output, see [docs/2-stereo-calibration.md](docs/2-stereo-calibration.md).



### Step 4 — Run stereo calibration

Open `src/step2_calibrate_stereo.py` and set `CHECKERBOARD` and `SQUARE_SIZE` to match your physical board (count **inner** corners only):

```python
CHECKERBOARD = (9, 6)   # inner corners (cols-1, rows-1) of your board
SQUARE_SIZE  = 0.020    # square side length in metres
```

Then run:

```bash
python3 src/step2_calibrate_stereo.py
```

Check the printed output before proceeding:
- **Stereo RMS should be < 1.0 px.** Above that, recapture with sharper images and more varied board poses.
- **Rectified focal ratios** (`P[0,0]/K[0,0]`) should both be within `[0.5, 2.0]`. Values outside this range mean the calibration will not be usable for stereo matching — recapture.

A valid run writes `stereo_calib.npz` to the project directory. Place it at `/workspace/stereo_calib.npz`, or update `CALIB_NPZ` in the tuning block of `src/stereo_tracking_code.py`.

> For a full explanation of the quality checks and what to do when they fail, see [docs/2-stereo-calibration.md](docs/2-stereo-calibration.md).



### Step 5 — Run the tracker

```bash
python3 src/stereo_tracking_code.py
```

Press **Q** to quit. Each run creates a timestamped folder at `/workspace/run_YYYY-MM-DD_HH-MM/` containing `tracking_log.csv` with one row per confirmed track per frame.

> For a full description of all tuning parameters, pipeline stages, and the CSV log schema, see [docs/3-stereo-tracking-pipeline.md](docs/3-stereo-tracking-pipeline.md).

---

## 📂 Repository Structure

*   `src/`
    *   `stereo_tracking_code.py` — Full self-contained pipeline (detection → stereo matching → tracking → display → logging)
    *   `yolo_drone_detection_training.ipynb` — Full YOLO training notebook (dataset prep, training, export)
    *   `step1_capture_calib_images.py` — Captures synchronised stereo image pairs for calibration
    *   `step2_calibrate_stereo.py` — Runs stereo calibration and produces `stereo_calib.npz`
    *   `export_TensorRT.py` — Exports `best.pt` to a TensorRT engine with all parameters pre-configured
*   `docs/` — Project documentation and guides
*   `assets/` — Images and media for documentation
*   `best.pt` — Trained YOLOv11s weights used in this project (~19 MB)
*   `dataset.txt` - Dataset used for the training
*   `requirements.txt` — Python dependencies


## 📚 Documentation

| Topic | Doc |
|---|---|
| YOLO training and deployment| [docs/1-yolo-training.md](docs/1-yolo-training.md) |
| Stereo calibration | [docs/2-stereo-calibration.md](docs/2-stereo-calibration.md) |
| Pipeline & tuning reference | [docs/3-stereo-tracking-pipeline.md](docs/3-stereo-tracking-pipeline.md) |

---

## 👥 Team

- **[Khaled Ghanem](https://github.com/khaledghanem0)**
- **[Eralp Erol](https://github.com/EralpErol)**
- **[Halit Özkaya](https://github.com/halitozkkaya)**
- **[Mustafa Ecevit](https://github.com/me422-arch)**

## 📄 License

This project is licensed under the [MIT License](LICENSE).
