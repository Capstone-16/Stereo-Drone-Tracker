# Stereo Drone Tracker — Code Reference

**File:** `src/stereo_tracking_code.py`
**Last updated:** 2026-05-13
**Platform:** NVIDIA Jetson (GStreamer + TensorRT)

---

## Purpose

Real-time detection, tracking, and 3D localisation of drones using two IMX477 CSI cameras in a stereo rig. The system uses YOLO for detection, stereo geometry for 3D reconstruction, and a Hungarian-algorithm track manager with Kalman-filter smoothing for identity continuity across frames.

---

## How to Run

```bash
python3 src/stereo_tracking_code.py
```

Press **Q** to quit. Each run creates a timestamped output folder at `/workspace/run_YYYY-MM-DD_HH-MM/`. The CSV log (`tracking_log.csv`) is written there and flushed every frame so a hard kill does not lose the tail.

---

## Code Structure (top-to-bottom)

```
TUNING PARAMETERS          Lines   ~47–295    All knobs in one place
Stage 1: CameraGrabber     Lines  ~298–343    GStreamer threaded frame grabber
Stage 2: StereoCalibration Lines  ~347–568    Calibration load, rectification, triangulation
         SharedGrid        Lines  ~571–755    Shared normalised [0,1] coordinate system
Stage 3: get_all_drones    Lines  ~758–859    YOLO result → detection dicts
         match_stereo_...  Lines  ~862–1049   Cam0 ↔ Cam1 detection pairing (Hungarian)
Stage 4: DroneKalmanFilter Lines ~1055–1100   6-state constant-velocity KF wrapper
Stage 5: Track             Lines ~1106–1372   Single-drone track object
Stage 6: TrackManager      Lines ~1376–2000   Multi-track lifecycle management
UI                         Lines ~2007–2285   Annotation, info bar, card panel, compositor
run()                      Lines ~2289–2500   Main loop
```

---

## Tuning Parameters

**Everything safe to change lives in the block at lines 47–300.**
The rest of the file references these module-level names; no other site needs editing for routine tuning.

### Paths & model
| Name | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/workspace/best.engine` | TensorRT YOLO engine |
| `CALIB_NPZ` | `/workspace/stereo_calib.npz` | Output of `step2_calibrate_stereo.py` |
| `LOG_PATH` | `/workspace/run_<timestamp>/tracking_log.csv` | Per-track CSV log. Written to a timestamped run folder created automatically on startup. |

### Camera hardware
| Name | Default | Meaning |
|---|---|---|
| `FPS` | `30` | Target capture / display rate |
| `FPS_CAP` | `30` | Hard upper limit on pipeline iterations per second. The loop sleeps spare time so the GPU is not fully pegged when the pipeline runs faster than this. Set to `0` to disable. |
| `CAM_W` / `CAM_H` | `800 / 450` | Display-resolution per camera panel (px) |
| `FLIP_METHOD` | `0` | GStreamer nvvidconv flip (0 = none, 2 = 180°) |

### Detection
| Name | Default | Meaning |
|---|---|---|
| `DRONE_CLASS` | `0` | YOLO class index for "drone" |
| `CONF_THRESH` | `0.45` | Min confidence to accept a YOLO box |

### Depth range
| Name | Default | Meaning |
|---|---|---|
| `MIN_Z_M` | `0.3 m` | Triangulated depth below this is discarded |
| `MAX_Z_M` | `80.0 m` | Triangulated depth above this is discarded |

### Stereo matching (cam0 ↔ cam1 pairing)
| Name | Default | Meaning |
|---|---|---|
| `STEREO_EPIPOLAR_TOL` | `0.05` | Max `|grid_y_L − grid_y_R|` to accept a pair. Hard gate — candidates above it get cost `1e6`. Tighten to `0.025` with a clean calibration. |
| `STEREO_DISPARITY_WEIGHT` | `0.30` | Weight on horizontal disparity inside the epipolar band |
| `STEREO_SIZE_WEIGHT` | `0.15` | Weight on bounding-box width mismatch |
| `STEREO_CONF_WEIGHT` | `0.10` | Weight penalising low-confidence pairs |
| `STEREO_STICKY_BONUS` | `0.03` | Cost discount when a candidate re-uses the same pair from the previous frame (hysteresis) |
| `STEREO_STICKY_TOL_PX` | `40.0` | Pixel radius for sticky-pair matching |

### Track lifecycle
| Name | Default | Meaning |
|---|---|---|
| `STABILITY_FRAMES` | `10` | Hits required before a track is "confirmed" and shown in the UI. Prevents YOLO phantom tracks. |
| `STABILITY_MAX_GAPS` | `3` | Consecutive missed frames tolerated without resetting the pre-confirmation hit count |
| `MAX_MISSED` | `15` | After this many fully-unmatched frames (≈ 0.5 s), a confirmed track becomes a zombie |
| `MAX_MATCH_DIST` | `0.08` | Grid-space distance gate for the Hungarian live track matcher |
| `MIN_SPEED_DIR` | `0.3 m/s` | Speed below which the direction-consistency cost is zeroed (hovering drone has no meaningful heading) |

### Zombie re-identification
| Name | Default | Meaning |
|---|---|---|
| `ZOMBIE_TTL_SEC` | `2.0 s` | Max time a zombie is kept before permanent deletion |
| `ZOMBIE_MAX_MATCH_DIST_M` | `2.5 m` | 3D Euclidean gate for stereo re-entry match |
| `ZOMBIE_MAX_MATCH_GRID` | `0.08` | Grid-space gate for single-cam re-entry match |

### Kalman filter
| Name | Default | Meaning |
|---|---|---|
| `KF_MAX_PREDICT` | `15` | Consecutive predict() calls before velocity state is zeroed (prevents unbounded drift during long single-cam stretches) |
| `KF_PROCESS_NOISE` | `5e-3` | Diagonal of `processNoiseCov` — larger = more reactive but jitterier |
| `KF_MEASUREMENT_NOISE` | `5e-2` | Diagonal of `measurementNoiseCov` — larger = smoother but laggier |

### Physical size calibration
| Name | Default | Meaning |
|---|---|---|
| `SIZE_CALIB_FRAMES` | `60` | Samples to collect before locking size (≈ 2 s at 30 fps) |
| `SIZE_SIMILARITY_THR` | `0.05 m` | Below this, size has no discriminating power |
| `SIZE_OUTLIER_STD` | `1.5` | Std-deviation gate for outlier removal before averaging |

### Cost-matrix weights (live track ↔ detection matching)
```python
WEIGHTS_WITH_SIZE = {'pos': 0.45, 'dir': 0.20, 'spd': 0.15, 'size': 0.20}
WEIGHTS_NO_SIZE   = {'pos': 0.55, 'dir': 0.25, 'spd': 0.20, 'size': 0.00}
```
`WEIGHTS_WITH_SIZE` activates once a track's physical size is locked (`SIZE_LOCKED`).

### UI layout
| Name | Default | Meaning |
|---|---|---|
| `UI_MAX_CARDS` | `2` | Drone card slots in the bottom panel. Increase to support more drones on screen simultaneously. |
| `UI_PANEL_H` | `280` | Bottom panel height (px) |
| `UI_BAR_H` | `36` | Info bar height (px) |
| `UI_CARD_PADDING` | `12` | Inner padding inside each drone card (px) |
| `UI_CARD_RADIUS` | `6` | Card border corner radius (px) |
| `UI_ROW_H` | `30` | Height of each data row inside a card (px) |
| `UI_LABEL_SCALE` | `0.38` | Font scale for row labels |
| `UI_VALUE_SCALE` | `0.65` | Font scale for row values |
| `UI_TITLE_SCALE` | `0.52` | Font scale for card title (DRONE ID N) |
| `UI_STATE_SCALE` | `0.38` | Font scale for state label inside card |

---

## Pipeline (one frame)

```
CameraGrabber (thread)     Raw BGR frames at CAM_W × CAM_H
        │
        ▼
YOLO batched inference     Both frames in one forward pass
        │
        ▼
get_all_drones()           Per-camera YOLO boxes → detection dicts
                           Each dict carries raw display-pixel coords,
                           normalised coords, shared grid coords (raw),
                           and — when calibration is trustworthy —
                           rectified grid coords (grid_x_rect / grid_y_rect)
        │
        ▼
match_stereo_detections()  Pairs cam0 ↔ cam1 detections using Hungarian
                           algorithm. Produces: stereo_pairs, cam0_only,
                           cam1_only
        │
        ▼
TrackManager.update()      Associates detections with live tracks (Hungarian).
                           Runs zombie re-ID before spawning new IDs.
                           Returns confirmed_tracks.
        │
        ├── annotate_frames()      Draw boxes/IDs on raw frames
        ├── build_canvas()         Composite display (cam row + bar + cards)
        ├── cv2.imshow()
        └── CSV log flush
```

---

## Stage Details

### Stage 1 — CameraGrabber

`nvarguscamerasrc` captures at native 1920×1080, then `nvvidconv` scales to `CAM_W × CAM_H` inside the GPU pipeline. No CPU resize is done. Each camera runs a background thread that pulls the latest frame into a lock-protected buffer; `read()` returns a copy.

---

### Stage 2 — StereoCalibration

Loads a `.npz` calibration file produced by `step2_calibrate_stereo.py`.

**Fields used downstream:**
- `K_L`, `K_R` — intrinsic matrices (raw focal length, principal point)
- `D_L`, `D_R` — distortion coefficients
- `R1`, `R2`, `P1`, `P2` — rectification rotation and projection matrices
- `T` — baseline vector **in metres**
- `image_size` — resolution the calibration was captured at (typically 1920×1080)

**Sanity check on load:** If `P1[0,0] / K_L[0,0]` or `P2[0,0] / K_R[0,0]` falls outside `[0.5, 2.0]`, the calibration is flagged `_rectification_trustworthy = False`. In this case:
- `use_point_rectification()` is silently refused.
- Detection dicts are not augmented with rectified coordinates.
- The pipeline falls back to raw-coordinate stereo matching (less accurate for wide baselines, but safe).
- A warning is printed at startup. **Fix by re-running `step2_calibrate_stereo.py`.**

**Point-level rectification** (active mode in `run()`): `use_point_rectification()` sets `_rectified = True` without building `cv2.remap` maps. Display frames stay raw (no visible zoom/warp). `rectify_points()` uses `cv2.undistortPoints(pts, K, D, R, P)` to correct each detection center — the same transform as `cv2.remap` but applied only to the handful of YOLO center points per frame. `triangulate()` then uses the fast path (inputs already rectified, no per-call `undistortPoints`).

**Full-frame rectification** (available but not used in `run()`): `build_rectification_maps()` + `rectify_pair()` applies `cv2.remap` to every pixel before YOLO. This can cause visible zoom/warp when `alpha=0` was used in `cv2.stereoRectify`. Prefer point-level rectification.

**`triangulate(pt_L, pt_R)`**: Both inputs must be in calibration-resolution coordinates (scale from display pixels using `scale_x = calib_W / CAM_W`). Returns a 3D `(x, y, z)` numpy array in metres.

---

### Stage 2b — SharedGrid

Maps each camera's normalized frame coordinates `[0,1]` to a common shared grid `[0,1] × [0,1]` that spans the *combined* horizontal FOV of both cameras.

**How it is constructed:**
1. Focal length from `K_L[0,0]` (raw, not rectified — robust against bad P1 values) is converted to display pixels.
2. `single_fov_rad = 2 * arctan(CAM_W / (2 * focal_px))` gives each camera's angular FOV.
3. `angle_offset_rad = arctan(baseline_m / 10.0)` gives the angular separation at a 10 m reference depth.
4. `offset_fraction = angle_offset_rad / single_fov_rad` — how much of one camera-width the baseline shifts cam1 relative to cam0.
5. Cam0 occupies `[0, 1]`; cam1 occupies `[offset_fraction, 1 + offset_fraction]`. Both are then normalised by `1 + offset_fraction` so the total span is `[0, 1]`.

**Y axis**: normalised Y is shared directly — after rectification, epipolar lines are horizontal so `norm_y` in either camera maps 1:1 to `grid_y`.

**Key properties:**
- `overlap_x_start / overlap_x_end` — region where both cameras see simultaneously; 3D triangulation is only possible here.
- `to_grid(norm_x, norm_y, cam_id)` — convert camera-normalised → grid.
- `to_norm(grid_x, grid_y, cam_id)` — inverse (returns `None` if outside FOV).
- `in_fov(grid_x, grid_y, cam_id)` — quick FOV check.
- `in_overlap(grid_x)` — quick overlap check.

**Important:** `T` is stored in metres in the `.npz` file. Do **not** divide `norm(T)` by 1000 — that bug collapsed the baseline to zero and caused cam0 / cam1 to overlap completely in the grid.

---

### Stage 3 — Detection

**`get_all_drones(result, ..., calib, scale_x, scale_y)`**

Converts a YOLO result for one camera into a list of detection dicts. Each dict:

```python
{
    'px', 'py'         : float  — center in display pixels (raw, used for display)
    'box'              : (x1,y1,x2,y2) in display pixels
    'conf'             : float  — YOLO confidence
    'norm_x', 'norm_y' : float  — normalised [0,1] in camera frame (raw)
    'grid_x', 'grid_y' : float  — shared grid coordinates (raw)
    'cam_id'           : int    — 0 = left, 1 = right

    # Only present when calibration is trustworthy:
    'px_rect_cal'      : float  — rectified x in calibration pixels
    'py_rect_cal'      : float  — rectified y in calibration pixels
    'grid_x_rect'      : float  — shared grid x of rectified center
    'grid_y_rect'      : float  — shared grid y of rectified center
}
```

Sorted by confidence descending (highest-confidence detections matched first).

---

**`match_stereo_detections(dets_L, dets_R, grid, ...)`**

Pairs cam0 and cam1 detections of the same physical drone.

**Step 1 — FOV pre-filter:** A cam0 detection outside cam1's FOV (or vice versa) cannot physically pair; it goes directly to `cam0_only` / `cam1_only`.

**Step 2 — Cost matrix** over remaining candidates (one row per cam0 detection, one column per cam1 detection). Cost components, all in roughly `[0, 1]`:

| Component | Key | How computed |
|---|---|---|
| Epipolar (Y) | hard gate | `|grid_y_rect_L − grid_y_rect_R| > STEREO_EPIPOLAR_TOL` → sentinel cost `1e6` |
| Disparity (X) | `STEREO_DISPARITY_WEIGHT` | `|grid_x_rect_L − grid_x_rect_R|` normalised |
| Box-size match | `STEREO_SIZE_WEIGHT` | `1 − min(wL,wR)/max(wL,wR)` — same drone has nearly equal box widths |
| Confidence | `STEREO_CONF_WEIGHT` | `1 − 0.5*(confL + confR)` — penalises weak detections |
| Sticky bonus | `−STEREO_STICKY_BONUS` | applied when this candidate re-uses the same pair from the previous frame (within `STEREO_STICKY_TOL_PX` pixels) |

Rectified coordinates (`grid_x_rect`, `grid_y_rect`) are used for the epipolar check and disparity cost when available; otherwise raw `grid_x`, `grid_y` are used.

**Step 3 — Hungarian solve** via `scipy.optimize.linear_sum_assignment`. Only pairs with cost < `1e6` are accepted.

Returns `(stereo_pairs, cam0_only, cam1_only)`.

---

### Stage 4 — DroneKalmanFilter

6-state constant-velocity model: state = `[x, y, z, vx, vy, vz]`, measurements = `[x, y, z]`.

- `correct(xyz)` — predict-then-correct with a fresh triangulated position. Resets `_predict_count = 0`. Returns `(pos, vel, speed)`.
- `predict()` — predict only (no measurement). Increments `_predict_count`. Once `_predict_count > KF_MAX_PREDICT` the velocity state is zeroed to stop unbounded drift. Returns `(pos, vel, speed)`.
- `dt` is computed from wall-clock time at every call, so the filter adapts to real frame timing rather than assuming a fixed 30 fps.

Noise tuning: `KF_PROCESS_NOISE` (model noise) and `KF_MEASUREMENT_NOISE` (measurement noise) are at the top of the file. Higher process noise = filter follows measurements more aggressively. Higher measurement noise = filter smooths more but lags.

---

### Stage 5 — Track

One instance per drone. Owns its own `DroneKalmanFilter`.

**Key fields:**
- `id` — unique integer, assigned once at `__init__`, never changed (including after zombie resurrection)
- `state` — one of `TRACKING`, `SINGLE_CAM`, `COASTING`
- `has_3d` — `True` once a valid triangulation has been applied
- `pos`, `vel`, `speed` — most recent KF output (only valid when `has_3d`)
- `grid_x`, `grid_y` — most recent shared grid position (always valid)
- `det_L`, `det_R` — detection dicts from **this frame only**; cleared on every `update()` call so stale boxes are never rendered
- `hit_streak` — consecutive frames with a detection
- `missed` — consecutive frames with no detection
- `_confirmed` — latching flag: once `True`, stays `True` for the track's lifetime
- `size_state` — one of `SIZE_UNCALIBRATED`, `SIZE_CALIBRATING`, `SIZE_LOCKED`
- `physical_size_m` — locked average physical width of the drone in metres

**Lifecycle counters per-frame (exactly one of these happens per track):**

| Outcome | Method called | KF op | `hit_streak` | `missed` |
|---|---|---|---|---|
| Stereo pair matched | `update(dL, dR, xyz)` | `kf.correct()` | `+1` | `= 0` |
| Single-cam matched (has_3d) | `update(det_L/R)` | `kf.predict()` | `+1` | `= 0` |
| Single-cam matched (no 3D) | `update(det_L/R)` | none | `+1` | `= 0` |
| No detection | `predict()` | `kf.predict()` | `−1` (≥ 0) or reset | `+1` |
| Just spawned | `__init__` | `kf.correct()` (init) | `= 1` | `= 0` |

**Confirmation latch (`is_confirmed` property):** The first time `hit_streak >= STABILITY_FRAMES`, `_confirmed` flips to `True` and stays there. Confirmed tracks remain visible through brief occlusions and do not flicker in/out of the display. Only `MAX_MISSED` can retire them to the zombie list.

**Pre-confirmation gap tolerance:** While `_confirmed` is `False`, up to `STABILITY_MAX_GAPS` consecutive misses are absorbed without resetting `hit_streak`. This allows a brand-new track to survive occasional single-frame YOLO drops during the stability window.

**Physical size calibration:** On every frame where `xyz is not None` and `det_L is not None` and `size_state != SIZE_LOCKED`, the track accumulates `(box_width_px * depth_m) / focal_px` into `size_buffer`. After `SIZE_CALIB_FRAMES` samples, outliers beyond `SIZE_OUTLIER_STD` standard deviations are discarded, the remainder is averaged, and `physical_size_m` is locked. Once locked, `WEIGHTS_WITH_SIZE` activates for this track's matching costs.

---

### Stage 6 — TrackManager

Manages all live `Track` objects plus the zombie list.

**`update(stereo_pairs, cam0_only, cam1_only)` — one call per frame:**

```
Step 1  Select weight set (WEIGHTS_WITH_SIZE or WEIGHTS_NO_SIZE)
Step 2  Stereo Hungarian: match live tracks to stereo detections
          - Failed triangulations (depth outside [MIN_Z_M, MAX_Z_M]) split
            into fallback_cam0 / fallback_cam1 for Step 3
          - Zombie re-ID (stereo): unmatched stereo dets checked against
            zombie predicted positions before spawning new IDs
          - Unmatched stereo dets spawn new tracks
Step 3  Single-cam Hungarian: match remaining live tracks to single-cam dets
          (includes fallback_cam0 / fallback_cam1 from Step 2)
          - Zombie re-ID (single-cam): unmatched single-cam dets checked
            against zombie last-known grid positions
          - Unmatched single-cam dets spawn new tracks
Step 4  predict() on all tracks not matched in Steps 2 or 3
Step 5  Retire: tracks with missed > MAX_MISSED →
            confirmed → zombie list
            unconfirmed → discarded
        Remove resurrected zombies from zombie list
Step 6  Return confirmed_tracks (tracks where is_confirmed == True)
```

**Hungarian cost function** (`_build_cost_matrix`): for each track-detection pair,

```
cost = pos_w  * pos_cost
     + dir_w  * dir_cost
     + spd_w  * spd_cost
     + size_w * size_cost
```

- `pos_cost` — grid distance / `MAX_MATCH_DIST`, clamped at `1.0`. Pairs beyond `MAX_MATCH_DIST` get sentinel `1e6`.
- `dir_cost` — angle between the track's velocity vector and the implied motion to the detection. Zeroed when `speed < MIN_SPEED_DIR`.
- `spd_cost` — ratio of (implied speed this frame) to (track's current speed), penalising implausible accelerations.
- `size_cost` — normalised difference between the detection's implied physical size and the track's locked `physical_size_m`. Zero if size not locked.

**Zombie re-identification:**

When a confirmed track exceeds `MAX_MISSED`, it is placed in `self.zombies` as `(Track, died_at_wallclock)` instead of being deleted. Two re-ID paths run before any new IDs are spawned:

- **Stereo path** (`_revive_via_stereo`): requires `zombie.has_3d`. Extrapolates the zombie's last known position forward using `pos + vel * dt_since_death`. Gate: 3D Euclidean distance ≤ `ZOMBIE_MAX_MATCH_DIST_M`. Hungarian solve over zombie-vs-detection distance matrix.

- **Single-cam path** (`_revive_via_single`): any zombie is eligible. Compares the zombie's last `grid_x, grid_y` to the detection's `grid_x, grid_y` (no velocity extrapolation — no fresh 3D available). Gate: grid distance ≤ `ZOMBIE_MAX_MATCH_GRID`. Hungarian solve.

A resurrected zombie is moved back to `self.tracks` with its original ID, Kalman state, size calibration, and history intact. `missed` is reset to 0. Zombies older than `ZOMBIE_TTL_SEC` are pruned at the start of each `update()` call by `_prune_zombies()`.

---

## CSV Log (`tracking_log.csv`)

One row per **confirmed** track per frame. Written to the run's output folder and flushed every frame. A new file is created for each run.

| Column | Content |
|---|---|
| `timestamp` | Wall-clock time of frame start (seconds, 4 decimal places) |
| `track_id` | Integer ID — stable across zombie resurrections |
| `state` | `TRACKING`, `SINGLE_CAM`, or `COASTING` |
| `cam_id` | Most recent camera that saw the track (0 or 1) |
| `conf` | YOLO confidence this frame (0.0 if fully unmatched / coasting) |
| `grid_x`, `grid_y` | Shared grid coordinates [0,1] |
| `x`, `y`, `z` | Kalman-smoothed 3D position in metres (blank if no 3D) |
| `distance` | Euclidean distance from origin `sqrt(x²+y²+z²)` in metres |
| `speed` | Kalman velocity magnitude in m/s (0.0 if no 3D) |

---

## Display Window

<img width="7200" height="3447" alt="UI_preview" src="https://github.com/user-attachments/assets/1c0feede-1e09-4826-8af2-b522566e326d" />


Box / card colors by state:
- **Green** — `TRACKING` (stereo, full 3D, Kalman active)
- **Amber** — `SINGLE_CAM` (one camera only, no 3D)
- **Grey** — `COASTING` (had 3D; now single-cam or no detection; Kalman predicting)
- **Yellow** — unconfirmed brand-new track in the pre-confirmation window (not shown — it never reaches `annotate_frames` until `is_confirmed` is True)

Bounding boxes are drawn **only when that camera produced a detection this frame**. Stale boxes from previous frames are never displayed.

---

## Calibration Notes

The system expects `stereo_calib.npz` produced by `step2_calibrate_stereo.py` using:
```python
cv2.stereoRectify(..., alpha=1, flags=cv2.CALIB_ZERO_DISPARITY)
```

`alpha=0` produces a P1 matrix with an extremely large focal length when cameras are physically toed-in. This passes the load step silently but causes:
- Point-level rectification to project detection centers wildly off their true epipolar lines.
- All stereo pairs to fail the epipolar gate → every drone appears as two single-cam-only detections.

The sanity check in `StereoCalibration.__init__` will warn at startup and disable rectification if `P1[0,0] / K_L[0,0]` or `P2[0,0] / K_R[0,0]` is outside `[0.5, 2.0]`. If you see the warning, re-calibrate.

**Baseline unit:** `T` is in **metres** in the `.npz` file. The `SharedGrid` and `StereoCalibration` baseline printout both account for this correctly. Do not divide `norm(T)` by 1000.

---

## Known Limitations

- `Track._id_counter` is a class attribute. Re-instantiating `TrackManager` in the same Python process continues numbering from the previous run. Restart the process to reset IDs.
- Physical size calibration assumes both cameras see the same drone; size estimates are meaningless on partially-occluded or overlapping detections during the `SIZE_CALIBRATING` window.
- The zombie single-cam re-ID path uses the zombie's last grid position without velocity extrapolation. For fast drones and long occlusions, raising `ZOMBIE_MAX_MATCH_GRID` (or switching to a 3D-only re-ID requirement) may be necessary to avoid false resurrections.
