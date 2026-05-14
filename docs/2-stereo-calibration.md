# Stereo Camera Calibration Guide

This guide walks through generating the `stereo_calib.npz` file required by the tracker. The process has two steps: capturing image pairs with both cameras, then running the calibration script on those images.

---

## What You Need

A **physical checkerboard calibration target.** The board must be rigid and flat — a printout glued to a hard backing works well. 

*A printable copy of the checkerboard pattern can be downloaded 
from [here](../assets/calibration_checkerboard_letter.pdf) or found at* `assets/calibration_checkerboard_letter.pdf`.

<img width="1275" height="1205" alt="check4" src="https://github.com/user-attachments/assets/24ff9d7d-68fc-47fd-a182-45ab789570d3" />

*A standard checkerboard target. Count the inner corners (not the squares) to set CHECKERBOARD in Step 2.*

> **Important:** Mount both cameras in their final physical positions **before** capturing. Moving the rig after calibration invalidates the calibration file.

---

## Step 1 — Capture Image Pairs

Run the capture script on the Jetson:

```bash
python3 src/step1_capture_calib_images.py
```

The script opens both cameras and streams a live side-by-side preview to your browser — no monitor needed on the Jetson. Open the following URL on any device on the same network:

```
http://<jetson-ip>:8081
```

**Controls** (type in the terminal, then press Enter):

| Key | Action |
|-----|--------|
| `s` | Save the current stereo pair |
| `d` | Delete the last saved pair |
| `q` | Quit |

Images are saved to `/workspace/calib/left/` and `/workspace/calib/right/`.

### Tips for a good calibration

- **Hold the board completely still** before pressing `s` — motion blur corrupts corner detection and will cause those pairs to be skipped.
- **Cover the full field of view:** capture pairs with the board in the corners, edges, and centre of the frame, not just the middle.
- **Vary the distance and tilt:** move the board closer and farther, and tilt it ±30° in different directions. Variety in board pose is what gives the calibration its accuracy.
- **Keep the board fully visible in both cameras simultaneously.** Pairs where the board is partially out of frame in either camera are automatically skipped.
- Aim for at least **20 pairs**. 30–40 is better. The script will warn you if you quit with fewer than 20.

---

## Step 2 — Run the Calibration

Open `src/step2_calibrate_stereo.py` and set the two values at the top to match your physical board:

```python
CHECKERBOARD = (9, 6)   # inner corners — count (columns-1, rows-1), NOT the squares
SQUARE_SIZE  = 0.020    # side length of one square in metres (e.g. 0.020 = 2 cm)
```

Then run:

```bash
python3 src/step2_calibrate_stereo.py
```

The script prints quality metrics as it runs. Check these before proceeding:

| Check | Target | What to do if it fails |
|-------|--------|------------------------|
| Stereo RMS reprojection error | **< 1.0 px** | Recapture — use a flatter board, better lighting, hold it stiller |
| Rectified focal ratios `P[0,0]/K[0,0]` | **Both within [0.5, 2.0]** | Recapture with more varied board poses and angles |
| Inter-camera rotation | **< 10°** | Check that the cameras are mounted level and parallel |

A successful run writes:

```
stereo_calib.npz  →  same directory as the script
```

Move or copy this file to `/workspace/stereo_calib.npz` (or update `CALIB_NPZ` in the tracker's tuning block to point to its location).

---

## Next Step

Run the tracker:

```bash
python3 src/stereo_tracking_code.py
```

If the tracker prints a rectification warning at startup, the calibration file is not usable — go back to Step 1 and recapture.
