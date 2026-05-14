"""
Pipeline:
  1. Two IMX477 CSI cameras stream via GStreamer Python bindings
  2. YOLO11s-TRT detection (batched, dynamic engine batch=2)
  3. Shared normalized grid — common coordinate system across both cameras
  4. Stereo matching — epipolar + disparity constraints
  5. Triangulation -> 3D position (overlap zone only)
  6. Per-drone Kalman filter (6-state) -> smoothed position + velocity
  7. Hungarian algorithm track manager — multi-drone ID assignment
  8. OpenCV composite display on Jetson screen
  9. CSV logging — one row per confirmed track per frame

Layout (single window):
  Top    : Camera 0 (left) and Camera 1 (right) side by side
  Middle : Info bar — FPS, active tracks, stereo pairs, single-cam count
  Bottom : One card per drone slot (UI_MAX_CARDS = 2)
           Green  = TRACKING   (stereo, full 3D)
           Amber  = SINGLE_CAM (one camera only, no 3D)
           Grey   = COASTING   (Kalman predicting, temporarily lost)

Controls:
  Q -- quit

Log file: /workspace/tracking_log.csv
"""

import os
import csv
import time
import threading
import concurrent.futures
import collections

os.environ['NO_AT_BRIDGE'] = '1'   # suppress GTK accessibility warning

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==============================================================================
# TUNING PARAMETERS
# ------------------------------------------------------------------------------
# Everything in this section is safe to edit and is the single source of truth
# for the pipeline's behaviour. Constants defined here are consumed by classes
# and functions further down the file; those sites reference the module-level
# names rather than hard-coding values, so you should never need to modify
# anything below this block for routine tuning.
#
# Groups:
#   1. Paths & model
#   2. Camera hardware
#   3. Detection
#   4. Depth range
#   5. Stereo matching (cam0 <-> cam1 detection pairing)
#   6. Track lifecycle (confirmation, missed-frame tolerance)
#   7. Zombie re-identification (recovering IDs after brief full occlusion)
#   8. Kalman filter
#   9. Physical size calibration
#  10. Cost-matrix weight sets (track <-> detection matching)
#  11. UI layout
#  12. Colour palette + per-state colour map
#  13. State label strings (do NOT edit — used as dict keys in logic)
# ==============================================================================

# -- 1. Paths & model ----------------------------------------------------------

MODEL_PATH = '/workspace/best.engine'        # TensorRT-exported YOLO engine
CALIB_NPZ  = '/workspace/stereo_calib.npz'   # Output of step2_calibrate_stereo.py
LOG_PATH   = '/workspace/tracking_log.csv'   # CSV log written during run


# -- 2. Camera hardware --------------------------------------------------------

FPS         = 30          # Target capture/display rate. IMX477 supports 30 or 60.
FPS_CAP     = 30          # Set to 0 to disable.
                          # Hard upper limit on pipeline iterations per second.
                          # At the end of each frame the loop sleeps any spare
                          # time so the GPU/CPU are not fully pegged when the
                          # pipeline runs faster than this.  Set to 0 to disable.

CAM_W       = 800         # Display-resolution width per camera panel (px)
CAM_H       = 450         # Display-resolution height per camera panel (px)
FLIP_METHOD = 0           # GStreamer nvvidconv flip-method (0=none, 2=180°)


# -- 3. Detection --------------------------------------------------------------

DRONE_CLASS = 0           # YOLO class index that corresponds to "drone"
CONF_THRESH = 0.45        # Minimum YOLO confidence for a detection to be kept.
                          # Raise to suppress false positives; lower to catch
                          # distant / low-contrast drones at cost of more noise.


# -- 4. Depth range ------------------------------------------------------------

MIN_Z_M = 0.3             # Minimum triangulated depth (m) to accept. Below this
                          # the stereo baseline geometry becomes unreliable.
MAX_Z_M = 80.0            # Maximum triangulated depth (m) to accept. Above this
                          # the triangulation error scales > 1 m per px of disparity
                          # noise and the measurement is effectively unusable.


# -- 5. Stereo matching --------------------------------------------------------
# Used by match_stereo_detections() to pair a cam0 detection with a cam1
# detection of the same physical drone. Costs are all in roughly the [0, 1]
# range so the relative weights are directly comparable.

STEREO_EPIPOLAR_TOL     = 0.05   # Max |grid_y_L - grid_y_R| accepted as a pair.
                                 # After rectification this is a hard geometric
                                 # gate — tighten to 0.025 with a clean
                                 # calibration if you see false pairings.
STEREO_DISPARITY_WEIGHT = 0.30   # Weight on |grid_x_L - grid_x_R| (disparity).
                                 # Secondary tiebreaker inside the epipolar band.
STEREO_SIZE_WEIGHT      = 0.15   # Weight on bounding-box width mismatch. Two
                                 # views of the same drone have near-equal
                                 # widths; very different widths -> different
                                 # drones.
STEREO_CONF_WEIGHT      = 0.10   # Weight penalising low-confidence pairs —
                                 # prefers pairing two strong detections over
                                 # mixing a strong one with a marginal one.
STEREO_STICKY_BONUS     = 0.03   # Cost discount for a candidate pair whose
                                 # centres match a pair formed on the previous
                                 # frame. Provides pairing hysteresis so jitter
                                 # around the epipolar gate doesn't flicker IDs.
STEREO_STICKY_TOL_PX    = 40.0   # Pixel radius (display space) within which a
                                 # candidate pair is considered the "same" as a
                                 # previous-frame pair for STEREO_STICKY_BONUS.


# -- 6. Track lifecycle --------------------------------------------------------
# STABILITY_FRAMES  : hits a brand-new track must accumulate before it is
#                     promoted to "confirmed" and allowed to appear in the
#                     UI / CSV log. Prevents single-frame YOLO phantoms from
#                     reaching the display.
# STABILITY_MAX_GAPS: max consecutive missed frames tolerated while the track
#                     is still in its pre-confirmation window. Up to this
#                     many gaps don't wipe the accumulated hit_streak; beyond
#                     it, progress resets to 0. Only applies pre-confirmation.
# MAX_MISSED        : after this many consecutive fully-unmatched frames
#                     (NEITHER camera saw the drone), a confirmed track is
#                     moved to the zombie list and removed from self.tracks.
# MAX_MATCH_DIST    : grid-space gate (0..1) for assigning a detection to an
#                     existing track in the Hungarian matcher. Detections
#                     beyond this from the track's last position get the
#                     sentinel cost and can't match that track.
# MIN_SPEED_DIR     : track speed (m/s) below which the direction-consistency
#                     cost is zeroed. Hovering drones give no meaningful
#                     heading signal, so including it just injects noise.

STABILITY_FRAMES   = 10
STABILITY_MAX_GAPS = 3
MAX_MISSED         = 15
MAX_MATCH_DIST     = 0.08
MIN_SPEED_DIR      = 0.3


# -- 7. Zombie re-identification -----------------------------------------------
# When a confirmed track is deleted (missed > MAX_MISSED) we keep a snapshot
# in self.zombies for a short window. If an unmatched detection arrives during
# that window near the zombie's velocity-extrapolated position, we resurrect
# the zombie with its original ID. Covers the common case of a drone being
# briefly occluded by BOTH cameras (1–2 s) and reappearing in the same area.
#
# Tuning trade-off:
#   Longer TTL / larger gates = more successful re-IDs but more risk of
#   re-identifying a DIFFERENT drone as the zombie when two drones cross.

ZOMBIE_TTL_SEC          = 2.0    # Max age of a zombie before it is dropped.
ZOMBIE_MAX_MATCH_DIST_M = 2.5    # 3D Euclidean gate (m) for matching a new
                                 # stereo pair to a zombie. Should cover
                                 # plausible drone motion during the gap
                                 # (e.g. 5 m/s * 2 s = 10 m if you expect fast
                                 # drones) without overlapping neighbouring
                                 # tracks.
ZOMBIE_MAX_MATCH_GRID   = 0.08   # Grid-space gate for matching a new
                                 # single-cam detection to a zombie's last
                                 # grid position. Mirrors MAX_MATCH_DIST.


# -- 8. Kalman filter ----------------------------------------------------------
# 6-state constant-velocity filter (x, y, z, vx, vy, vz) correcting on 3D
# position measurements from triangulation.

KF_MAX_PREDICT       = 15    # After this many consecutive predict() calls
                             # with no correct() (i.e. no stereo triangulation),
                             # the filter zeros its velocity state. Prevents
                             # unbounded extrapolation drift when a drone is
                             # tracked on one camera only for a long time.
KF_PROCESS_NOISE     = 5e-3  # Diagonal of processNoiseCov. Larger -> filter
                             # trusts the model less / the measurements more
                             # (more reactive but jitterier).
KF_MEASUREMENT_NOISE = 5e-2  # Diagonal of measurementNoiseCov. Larger ->
                             # filter trusts each measurement less (smoother
                             # but more laggy).


# -- 9. Physical size calibration ----------------------------------------------
# While a track has 3D, we collect samples of (bounding-box width * depth)
# to infer the drone's real-world size. Once locked, size is used as an
# extra matching signal (size_weight in WEIGHTS_WITH_SIZE).

SIZE_CALIB_FRAMES   = 60     # Samples collected before locking (~2 s at 30 fps)
SIZE_SIMILARITY_THR = 0.05   # Metres — below this, size has no discriminating
                             # power for telling drones apart.
SIZE_OUTLIER_STD    = 1.5    # Std deviations — outlier samples above this are
                             # discarded before averaging.


# -- 10. Cost-matrix weight sets -----------------------------------------------
# Used in TrackManager._build_cost_matrix. Each key corresponds to a partial
# cost in [0, 1]; the total cost is the weighted sum. Two sets are kept for
# whether the track has a locked physical size.

WEIGHTS_WITH_SIZE = {
    'pos'  : 0.45,      # grid-distance from track to detection
    'dir'  : 0.20,      # direction mismatch vs. track velocity vector
    'spd'  : 0.15,      # speed-consistency (implied speed vs. tracked speed)
    'size' : 0.20,      # physical-size mismatch (requires locked size)
}
WEIGHTS_NO_SIZE = {
    'pos'  : 0.55,
    'dir'  : 0.25,
    'spd'  : 0.20,
    'size' : 0.00,
}


# -- 11. UI layout -------------------------------------------------------------

UI_MAX_CARDS     = 2     # Number of drone card slots at the bottom of the window.
                         # Increase for more than 2 simultaneous drones on screen.
UI_PANEL_H       = 280   # Bottom panel height (px)
UI_BAR_H         = 36    # Info bar height (px)
UI_CARD_PADDING  = 12    # Inner padding inside each drone card (px)
UI_CARD_RADIUS   = 6     # Card border corner radius (px)
UI_ROW_H         = 30    # Height of each data row inside a card (px)
UI_LABEL_SCALE   = 0.38  # Font scale for row labels
UI_VALUE_SCALE   = 0.65  # Font scale for row values
UI_TITLE_SCALE   = 0.52  # Font scale for card title (DRONE ID N)
UI_STATE_SCALE   = 0.38  # Font scale for state label inside card


# -- 12. Colour palette (BGR) --------------------------------------------------

C_PANEL  = (255, 255, 255)
C_BORDER = (180, 200, 180)
C_GREEN  = (56,  168, 46)
C_DGREEN = (40,  120, 30)
C_AMBER  = (0,   160, 196)
C_RED    = (50,  60,  192)
C_TEXTD  = (100, 130, 100)
C_BLACK  = (20,  30,  20)
C_BLUE   = (168, 95,  26)


# -- 13. State label strings (DO NOT EDIT) -------------------------------------
# Used as dict keys in UI_COLOR and as the state field on Track instances
# and in the CSV log. Changing the strings will break logic.

TRACKING   = 'TRACKING'     # stereo confirmed, full 3D Kalman active
SINGLE_CAM = 'SINGLE_CAM'   # seen by one camera only, no 3D
COASTING   = 'COASTING'     # had 3D; this frame = single-cam or no detection

SIZE_UNCALIBRATED = 'SIZE_UNCALIBRATED'
SIZE_CALIBRATING  = 'SIZE_CALIBRATING'
SIZE_LOCKED       = 'SIZE_LOCKED'


# Per-state colour for all camera-annotation and card drawing. Edit here to
# change how each state looks everywhere in the UI simultaneously.
UI_COLOR = {
    TRACKING   : C_GREEN,
    SINGLE_CAM : C_AMBER,
    COASTING   : (180, 180, 180),
    'panel'    : C_PANEL,
    'border'   : C_BORDER,
    'label'    : C_TEXTD,
    'title_bg' : (230, 236, 230),
    'empty'    : C_TEXTD,
    'bar_bg'   : (230, 236, 230),
    'bar_text' : C_DGREEN,
    'bar_info' : C_TEXTD,
}

# ==============================================================================
# END OF TUNING PARAMETERS — everything below is implementation.
# ==============================================================================


# STAGE 1 -- CAMERA GRABBER
class CameraGrabber:
    def __init__(self, sensor_id, width, height, fps=30, flip=0, name="cam"):
        self.name = name
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=1920, height=1080, "
            f"framerate={fps}/1 ! "
            f"nvvidconv flip-method={flip} ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=sink emit-signals=True max-buffers=1 drop=True"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink     = self._pipeline.get_by_name('sink')
        self._pipeline.set_state(Gst.State.PLAYING)
        self._frame    = None
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        t = threading.Thread(target=self._loop, daemon=True, name=name)
        t.start()
        print(f"[{name}] grabber started")

    def _loop(self):
        while not self._stop.is_set():
            sample = self._sink.emit('pull-sample')
            if sample is None:
                continue
            buf  = sample.get_buffer()
            caps = sample.get_caps()
            h    = caps.get_structure(0).get_value('height')
            w    = caps.get_structure(0).get_value('width')
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if ok:
                frame = np.ndarray(shape=(h, w, 3), dtype=np.uint8,
                                   buffer=bytes(mapinfo.data)).copy()
                buf.unmap(mapinfo)
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._stop.set()
        self._pipeline.set_state(Gst.State.NULL)


# STAGE 2 -- STEREO CALIBRATION
class StereoCalibration:
    """
    Holds stereo calibration matrices and precomputes rectification maps.

    After build_rectification_maps() is called, the maps rectify raw
    camera frames at DISPLAY resolution — pitch / roll / small rotation
    mismatches between the two physical cameras are cancelled out and
    epipolar lines become horizontal to calibration precision. This is
    what allows the downstream epipolar gate in match_stereo_detections
    to be a hard geometric constraint instead of a soft assumption.
    """

    def __init__(self, npz_path):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Calibration file not found: {npz_path}")
        d = np.load(npz_path)
        self.K_L = d['K_L'].astype(np.float64)
        self.D_L = d['D_L'].astype(np.float64)
        self.K_R = d['K_R'].astype(np.float64)
        self.D_R = d['D_R'].astype(np.float64)
        self.R1  = d['R1'].astype(np.float64)
        self.R2  = d['R2'].astype(np.float64)
        self.P1  = d['P1'].astype(np.float64)
        self.P2  = d['P2'].astype(np.float64)
        self.T = d['T'].astype(np.float64)
        self.image_size = tuple(d['img_size'].tolist())
        baseline_mm = float(np.linalg.norm(self.T)) * 1000
        print(f"[Calib] Loaded  calib_size={self.image_size}  "
              f"baseline={baseline_mm:.1f} mm")

        # Rectification maps are lazily built once build_rectification_maps()
        # is called. Until then triangulate() still works (it applies the
        # point-wise rectification internally), but frames are not remapped.
        self.map_L1 = None
        self.map_L2 = None
        self.map_R1 = None
        self.map_R2 = None
        self._rectified = False

        # ── Calibration sanity check ─────────────────────────────────
        # A healthy stereoRectify output has P1[0,0] and P2[0,0] within
        # a small factor of K_L[0,0] / K_R[0,0]. Some saved calibrations
        # ship with a mis-computed P1 / P2 (wrong stereoRectify args,
        # wrong newImageSize, or just corrupted). In that case the
        # rectified focal length is absurd, rectify_points returns
        # garbage, and point-level rectification makes everything worse
        # — cam0 and cam1 detections of the same drone land many
        # grid-widths apart and never pair.
        #
        # We detect this up front and refuse to enable rectification
        # unless the numbers make sense.
        kL = float(self.K_L[0, 0])
        kR = float(self.K_R[0, 0])
        pL = float(self.P1[0, 0])
        pR = float(self.P2[0, 0])
        ratio_L = pL / kL if kL > 0 else float('inf')
        ratio_R = pR / kR if kR > 0 else float('inf')
        sane = (0.5 <= ratio_L <= 2.0) and (0.5 <= ratio_R <= 2.0)
        self._rectification_trustworthy = sane
        if not sane:
            print(
                "[Calib] WARNING: rectification matrices look unreliable:\n"
                f"          K_L[0,0]={kL:.1f}  P1[0,0]={pL:.1f}  "
                f"(ratio={ratio_L:.2f})\n"
                f"          K_R[0,0]={kR:.1f}  P2[0,0]={pR:.1f}  "
                f"(ratio={ratio_R:.2f})\n"
                "          Point-level rectification is DISABLED — the\n"
                "          pipeline will fall back to raw-coordinate\n"
                "          stereo matching. Recalibrate your cameras if\n"
                "          you need epipolar-accurate pairing for wider\n"
                "          baselines."
            )
        else:
            print(
                f"[Calib] Rectification sanity: ratios "
                f"P1/K_L={ratio_L:.2f}, P2/K_R={ratio_R:.2f}  (OK)"
            )

    def use_point_rectification(self):
        """
        Declare that downstream code will hand triangulate() points that
        have already been rectified (via rectify_points()). Flips the
        fast path on triangulate() without building any cv2.remap maps,
        so display frames stay raw.

        This is the preferred mode when full-frame cv2.remap produces
        visibly zoomed / warped output — which happens when the saved
        calibration used stereoRectify(alpha=0) and/or the two cameras
        are physically toed-in enough that the rectifying rotation is
        aggressive. Geometric correction still happens at the point
        level, where it is essentially free and causes zero display
        artefacts.

        If the calibration self-check in __init__ flagged P1 / P2 as
        untrustworthy we refuse to flip the flag — the rectify_points()
        output is garbage in that case and enabling the fast path would
        silently produce wildly wrong 3D positions.
        """
        if not self._rectification_trustworthy:
            print("[Calib] use_point_rectification ignored — calibration "
                  "failed the sanity check. Raw-coordinate matching only.")
            return
        self._rectified = True

    def build_rectification_maps(self, cam_w, cam_h):
        """
        Precompute cv2.remap maps that rectify raw frames at display
        resolution (cam_w x cam_h). After this is called, the caller is
        expected to run cv2.remap on every frame before detection, and
        triangulate() switches to the "inputs are already rectified"
        fast path.

        NOTE: the resulting frames may look zoomed / warped depending on
        the alpha value used when the calibration was generated. Prefer
        use_point_rectification() for display-friendly behaviour.

        The calibration K / P matrices live in CALIBRATION resolution,
        so we scale the intrinsics to DISPLAY resolution before asking
        OpenCV to build the maps — this way the remap runs directly on
        the display-sized frames YOLO sees, with no intermediate resize.
        """
        calib_w, calib_h = self.image_size
        s_x = cam_w / calib_w
        s_y = cam_h / calib_h

        # Scale K matrices (fx, fy, cx, cy) to display resolution.
        S = np.array([[s_x, 0.0, 0.0],
                      [0.0, s_y, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        K_L_disp = S @ self.K_L
        K_R_disp = S @ self.K_R

        # Scale P matrices (3x4). Rows 0 and 1 scale with s_x and s_y
        # respectively; row 2 is unchanged.
        P1_disp = self.P1.copy()
        P1_disp[0, :] *= s_x
        P1_disp[1, :] *= s_y
        P2_disp = self.P2.copy()
        P2_disp[0, :] *= s_x
        P2_disp[1, :] *= s_y

        self.map_L1, self.map_L2 = cv2.initUndistortRectifyMap(
            K_L_disp, self.D_L, self.R1, P1_disp,
            (cam_w, cam_h), cv2.CV_16SC2)
        self.map_R1, self.map_R2 = cv2.initUndistortRectifyMap(
            K_R_disp, self.D_R, self.R2, P2_disp,
            (cam_w, cam_h), cv2.CV_16SC2)
        self._rectified = True

        print(f"[Calib] Rectification maps built at {cam_w}x{cam_h}  "
              f"(rectified focal_L={P1_disp[0,0]:.1f}px, "
              f"focal_R={P2_disp[0,0]:.1f}px)")

    def rectify_pair(self, frame_L, frame_R):
        """
        Apply full-frame rectification. Only runs when
        build_rectification_maps() has been called AND the pipeline has
        opted into frame-level rectification; otherwise this is a no-op.
        Point-level rectification (rectify_points) is usually preferable
        because it avoids visible zoom / warp in the displayed frames.
        """
        if self.map_L1 is None or self.map_R1 is None:
            return frame_L, frame_R
        rect_L = cv2.remap(frame_L, self.map_L1, self.map_L2, cv2.INTER_LINEAR)
        rect_R = cv2.remap(frame_R, self.map_R1, self.map_R2, cv2.INTER_LINEAR)
        return rect_L, rect_R

    def rectify_points(self, pts_cal, cam_id):
        """
        Rectify a batch of 2D detection centers.

        pts_cal : iterable of (u, v) in CALIBRATION image coordinates.
                  Typical pipeline: take display-pixel center (px, py)
                  and pass (px * scale_x, py * scale_y).
        cam_id  : 0 for the left camera, 1 for the right.

        Returns : np.ndarray of shape (N, 2) in RECTIFIED calibration
                  coordinates, compatible with the triangulate() fast
                  path.

        Under the hood this is cv2.undistortPoints(pts, K, D, R, P) —
        the same transform cv2.initUndistortRectifyMap uses internally,
        but applied only to the handful of detection centers per frame
        rather than to every pixel.
        """
        pts = np.asarray(pts_cal, dtype=np.float64).reshape(-1, 1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if cam_id == 0:
            K, D, R, P = self.K_L, self.D_L, self.R1, self.P1
        else:
            K, D, R, P = self.K_R, self.D_R, self.R2, self.P2
        rect = cv2.undistortPoints(pts, K, D, R=R, P=P)
        return rect.reshape(-1, 2)

    def triangulate(self, pt_L, pt_R):
        """
        Triangulate a 3D point from a pair of 2D observations.

        Inputs are in CALIBRATION image coordinates. When the frames
        have already been rectified by rectify_pair(), the observations
        are in rectified calibration coordinates — we can feed them
        straight to cv2.triangulatePoints(P1, P2, ...) without an
        explicit point-wise undistortPoints() pass.

        When rectification maps have NOT been built yet (backward-
        compatible path), we fall back to per-point undistortPoints
        with R=R1/R2 and P=P1/P2, which is what the old code did.
        """
        if self._rectified:
            p_L_xy = np.array([[pt_L[0]], [pt_L[1]]], dtype=np.float64)
            p_R_xy = np.array([[pt_R[0]], [pt_R[1]]], dtype=np.float64)
        else:
            p_L = cv2.undistortPoints(np.array([[pt_L]], dtype=np.float64),
                                      self.K_L, self.D_L, R=self.R1, P=self.P1)
            p_R = cv2.undistortPoints(np.array([[pt_R]], dtype=np.float64),
                                      self.K_R, self.D_R, R=self.R2, P=self.P2)
            p_L_xy = p_L.reshape(2, 1).astype(np.float64)
            p_R_xy = p_R.reshape(2, 1).astype(np.float64)

        pts4d = cv2.triangulatePoints(self.P1, self.P2, p_L_xy, p_R_xy)
        return (pts4d[:3] / pts4d[3]).flatten()


### SHARED COORDINATE SYSTEM
class SharedGrid:
    """
    Establishes a common normalized [0,1] x [0,1] grid across both cameras.

    The grid spans the COMBINED horizontal field of view of both cameras.
    Camera 0 (left)  occupies the left portion of the grid.
    Camera 1 (right) occupies the right portion of the grid.
    The overlap zone is the region covered by both cameras simultaneously.

    Y axis is shared directly — after rectification, both cameras have
    the same vertical FOV and epipolar lines are horizontal, so
    normalized Y in either camera maps 1:1 to grid Y to calibration
    precision.

    X axis is computed from the calibration:
      - Focal length (P1[0,0] in calibration pixels, the RECTIFIED focal
        length) defines the angular FOV.
      - Baseline defines the horizontal offset between the two cameras.
      - Together they determine what fraction of the combined FOV each
        camera covers and where Camera 1's frame starts relative to
        Camera 0's frame.

    Units are intentionally dimensionless — the grid does NOT represent
    physical metres. Actual 3D distance is computed separately via triangulation
    only when both cameras see the drone simultaneously.
    """

    def __init__(self, calib, cam_w, cam_h):
        """
        calib  : StereoCalibration instance (already loaded)
        cam_w  : display frame width  (pixels) — e.g. 800
        cam_h  : display frame height (pixels) — e.g. 450
        """
        self.cam_w = cam_w
        self.cam_h = cam_h

        # Baseline in metres. StereoCalibration stores T in METRES (that
        # is why its own __init__ diagnostic prints `norm(T) * 1000` to
        # get millimetres). An earlier version of this code divided by
        # 1000 on the assumption that T was in mm, which silently
        # collapsed the baseline to ~0 and made cam0 / cam1 share the
        # exact same [0, 1] grid span — visible as `baseline=0.0cm` in
        # the SharedGrid printout.
        baseline_m = float(np.linalg.norm(calib.T))

        # Focal length in calibration-resolution pixels, converted to
        # display pixels. We use the raw K_L[0,0] rather than the
        # rectified P1[0,0] — P1 is unreliable for calibrations that
        # failed the StereoCalibration sanity check (e.g. rectified
        # focal ~20x the raw focal), and K_L is always a first-order
        # correct description of the camera's angular FOV regardless of
        # whether rectification is active.
        calib_w = calib.image_size[0]
        scale   = cam_w / calib_w
        focal_px = calib.K_L[0, 0] * scale   # focal length in display pixels

        # Angular width of one camera's FOV (in radians)
        # cam_w / focal_px = 2 * tan(half_fov)
        single_fov_rad = 2.0 * np.arctan(cam_w / (2.0 * focal_px))

        # The horizontal offset between the two cameras introduces a shift.
        # In angular terms: angle_offset = arctan(baseline / depth)
        # We don't know depth here, but we can express the camera separation
        # as a fraction of the single-camera FOV at a reference depth.
        # We use the angular separation at a reference depth of 10m — this
        # gives the grid a stable, consistent layout regardless of actual drone depth.
        # The grid is dimensionless; this reference is only used to set proportions.
        REFERENCE_DEPTH_M = 10.0
        angle_offset_rad  = np.arctan(baseline_m / REFERENCE_DEPTH_M)

        # Fraction of one camera FOV that the baseline offset represents
        offset_fraction = angle_offset_rad / single_fov_rad

        # Camera 0 (left) occupies grid X: [0, 1]
        # Camera 1 (right) is shifted right by offset_fraction
        # So Camera 1 occupies grid X: [offset_fraction, 1 + offset_fraction]
        # We then normalize the entire combined span to [0, 1]
        self.cam0_x_start = 0.0
        self.cam0_x_end   = 1.0
        self.cam1_x_start = offset_fraction
        self.cam1_x_end   = 1.0 + offset_fraction

        combined_width = self.cam1_x_end  # = 1 + offset_fraction

        # Normalize all edges to [0, 1] over the full combined FOV
        self.cam0_x_start /= combined_width
        self.cam0_x_end   /= combined_width
        self.cam1_x_start /= combined_width
        self.cam1_x_end   /= combined_width

        # Width of one camera's contribution in grid space
        self.cam0_grid_w = self.cam0_x_end - self.cam0_x_start
        self.cam1_grid_w = self.cam1_x_end - self.cam1_x_start

        # Overlap zone in grid X — where both cameras see simultaneously
        self.overlap_x_start = self.cam1_x_start
        self.overlap_x_end   = self.cam0_x_end

        print(f"[SharedGrid] baseline={baseline_m*100:.1f}cm  "
              f"focal={focal_px:.1f}px  "
              f"cam0=[{self.cam0_x_start:.3f}, {self.cam0_x_end:.3f}]  "
              f"cam1=[{self.cam1_x_start:.3f}, {self.cam1_x_end:.3f}]  "
              f"overlap=[{self.overlap_x_start:.3f}, {self.overlap_x_end:.3f}]")

    # ------------------------------------------------------------------
    # COORDINATE CONVERSION
    # ------------------------------------------------------------------

    def to_grid(self, norm_x, norm_y, cam_id):
        """
        Convert a normalized detection (norm_x, norm_y) in [0,1] from
        camera cam_id (0=left, 1=right) into shared grid coordinates.

        norm_x, norm_y : detection center normalized to [0,1] in the camera frame
                         norm_x = px / cam_w,  norm_y = py / cam_h
        cam_id         : 0 or 1

        Returns (grid_x, grid_y) in [0,1] x [0,1] shared space.
        """
        if cam_id == 0:
            grid_x = self.cam0_x_start + norm_x * self.cam0_grid_w
        else:
            grid_x = self.cam1_x_start + norm_x * self.cam1_grid_w
        grid_y = norm_y   # Y is shared directly
        return grid_x, grid_y

    def to_norm(self, grid_x, grid_y, cam_id):
        """
        Inverse of to_grid — project a shared grid coordinate back into
        a specific camera's normalized frame.

        Returns (norm_x, norm_y) or None if the point is outside that
        camera's field of view.
        """
        if cam_id == 0:
            x_start, grid_w = self.cam0_x_start, self.cam0_grid_w
        else:
            x_start, grid_w = self.cam1_x_start, self.cam1_grid_w

        norm_x = (grid_x - x_start) / grid_w
        norm_y = grid_y

        if not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0):
            return None   # outside this camera's FOV
        return norm_x, norm_y

    def in_fov(self, grid_x, grid_y, cam_id):
        """
        Returns True if the shared grid point is within cam_id's field of view.
        Quick visibility check — does not require full coordinate conversion.
        """
        if cam_id == 0:
            return self.cam0_x_start <= grid_x <= self.cam0_x_end
        else:
            return self.cam1_x_start <= grid_x <= self.cam1_x_end

    def in_overlap(self, grid_x):
        """
        Returns True if grid_x falls in the overlap zone where both
        cameras see simultaneously (stereo triangulation is possible).
        """
        return self.overlap_x_start <= grid_x <= self.overlap_x_end

    def detection_to_grid(self, px, py, cam_id):
        """
        Convenience method: convert raw pixel detection (px, py) directly
        to shared grid coordinates.
        px, py : pixel coordinates in the display-resolution frame
        """
        norm_x = px / self.cam_w
        norm_y = py / self.cam_h
        return self.to_grid(norm_x, norm_y, cam_id)

    def grid_to_pixel(self, grid_x, grid_y, cam_id):
        """
        Convenience method: convert shared grid coordinate to pixel
        coordinates in a specific camera's display frame.
        Returns (px, py) or None if outside that camera's FOV.
        """
        result = self.to_norm(grid_x, grid_y, cam_id)
        if result is None:
            return None
        norm_x, norm_y = result
        return int(norm_x * self.cam_w), int(norm_y * self.cam_h)


def get_all_drones(result, drone_class, conf_thresh, grid, cam_id, cam_w, cam_h,
                   calib=None, scale_x=None, scale_y=None):
    """
    Returns all drone detections above conf_thresh from a single YOLO result.
    Each detection is a dict carrying everything needed downstream.

    When calib / scale_x / scale_y are provided, each detection is
    augmented with RECTIFIED coordinates used by stereo matching and
    triangulation. Raw display-space fields (px, py, box, grid_x,
    grid_y) are left untouched so rendering stays in the original,
    non-warped image coordinates.

    Returns a list of dicts:
    {
        'px'     : float  — center x in display pixels (raw)
        'py'     : float  — center y in display pixels (raw)
        'box'    : (x1, y1, x2, y2) in display pixels — for drawing
        'conf'   : float  — detection confidence
        'norm_x' : float  — normalized x in [0,1] within this camera's frame (raw)
        'norm_y' : float  — normalized y in [0,1] within this camera's frame (raw)
        'grid_x' : float  — shared grid x coordinate (raw)
        'grid_y' : float  — shared grid y coordinate (raw)
        'cam_id' : int    — which camera produced this detection (0 or 1)

        # Only present when calib is provided:
        'px_rect_cal' : float  — rectified x in calibration pixels
        'py_rect_cal' : float  — rectified y in calibration pixels
        'grid_x_rect' : float  — shared grid x for the rectified center
        'grid_y_rect' : float  — shared grid y for the rectified center
    }
    """
    detections = []

    for box in result.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != drone_class or conf < conf_thresh:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        px = (x1 + x2) / 2.0
        py = (y1 + y2) / 2.0

        norm_x = px / cam_w
        norm_y = py / cam_h

        grid_x, grid_y = grid.to_grid(norm_x, norm_y, cam_id)

        detections.append({
            'px'     : px,
            'py'     : py,
            'box'    : (int(x1), int(y1), int(x2), int(y2)),
            'conf'   : conf,
            'norm_x' : norm_x,
            'norm_y' : norm_y,
            'grid_x' : grid_x,
            'grid_y' : grid_y,
            'cam_id' : cam_id,
        })

    # Batch point-level rectification. We do it here rather than in
    # match_stereo_detections / TrackManager so every detection carries
    # its rectified twin and downstream code does not need to juggle
    # the calibration object.
    #
    # Skipped when the calibration self-check flagged P1 / P2 as
    # untrustworthy — running rectify_points() on a bogus P would
    # produce rectified coords many grid-widths off from the raw
    # coords, which in turn would make every cam0/cam1 pair fail the
    # epipolar gate. Falling back to raw grid coords is strictly better
    # than publishing garbage.
    if (calib is not None
            and scale_x is not None
            and scale_y is not None
            and detections
            and getattr(calib, '_rectification_trustworthy', True)):
        pts_cal = np.array(
            [[d['px'] * scale_x, d['py'] * scale_y] for d in detections],
            dtype=np.float64,
        )
        pts_rect = calib.rectify_points(pts_cal, cam_id)
        inv_sx = 1.0 / scale_x
        inv_sy = 1.0 / scale_y
        for d, (u_rect_cal, v_rect_cal) in zip(detections, pts_rect):
            # Rectified pixel in display space, used to recompute the
            # grid coordinate consistently with the raw-grid pipeline.
            u_disp_rect = u_rect_cal * inv_sx
            v_disp_rect = v_rect_cal * inv_sy
            nx_r = u_disp_rect / cam_w
            ny_r = v_disp_rect / cam_h
            gx_r, gy_r = grid.to_grid(nx_r, ny_r, cam_id)

            d['px_rect_cal'] = float(u_rect_cal)
            d['py_rect_cal'] = float(v_rect_cal)
            d['grid_x_rect'] = gx_r
            d['grid_y_rect'] = gy_r

    # Sort by confidence descending so highest-confidence detections
    # get priority during matching
    detections.sort(key=lambda d: d['conf'], reverse=True)
    return detections


def match_stereo_detections(dets_L, dets_R, grid,
                             epipolar_tol=STEREO_EPIPOLAR_TOL,
                             disparity_weight=STEREO_DISPARITY_WEIGHT,
                             size_weight=STEREO_SIZE_WEIGHT,
                             conf_weight=STEREO_CONF_WEIGHT,
                             prev_pairs=None,
                             sticky_bonus=STEREO_STICKY_BONUS,
                             sticky_tol_px=STEREO_STICKY_TOL_PX):
    """
    Pairs Camera 0 detections with Camera 1 detections using the shared
    grid and the Hungarian algorithm.

    Pipeline:
      1. FOV pre-classification — a cam0 detection whose grid point is
         outside cam1's field of view (or a cam1 detection outside cam0's
         FOV) cannot possibly pair. It goes straight into cam0_only /
         cam1_only and is excluded from the cost matrix.

      2. Cost matrix over the remaining candidates, one row per cam0
         candidate, one column per cam1 candidate. Cost components (all
         roughly [0, 1]):
           - Epipolar (Y) residual — the primary signal. Pairs with
             |grid_y_L - grid_y_R| > epipolar_tol are gated out with a
             sentinel cost (1e6), so the Hungarian solver is never
             allowed to use them.
           - Disparity (X) residual — weighted by disparity_weight as a
             secondary tiebreaker when multiple cam1 detections are
             within the epipolar band.
           - Box-size similarity — weighted by size_weight. The same
             drone seen by both cameras has near-equal bounding-box
             widths (stereo disparity changes width by a few percent at
             realistic depths). Wildly different box widths indicate
             two different drones, not a pair.
           - Confidence penalty — weighted by conf_weight. Prefers
             pairing two high-confidence detections over mixing a
             high-conf cam0 with a marginal cam1.
         If prev_pairs is provided, a small sticky_bonus is subtracted
         from the cost of any candidate pair whose (cam0 center, cam1
         center) is close to a pair that was formed on the previous
         frame. This gives the solver hysteresis — frame-to-frame
         pairing stops flickering across the epipolar boundary for
         marginal cases.

         Both residuals are in grid units. The pair's disparity SIGN is
         intentionally not constrained; the shared grid's 10 m reference
         shift makes sign-of-grid-x-delta a range-dependent quantity, so
         gating on it silently drops valid far pairs (see the stereo-
         tracker patch notes).

      3. scipy.optimize.linear_sum_assignment picks the globally minimum
         cost assignment. Any assigned cell with cost >= 1e6 is dropped
         (that pair was gated out). Anything not matched goes to
         cam0_only / cam1_only as before.

    Replacing the previous greedy-left-to-right-by-confidence pass with a
    global assignment is more robust when two cam0 detections both have
    plausible cam1 matches.

    Args:
        prev_pairs    : list of ((cxL, cyL), (cxR, cyR)) pixel centers
                        of stereo pairs formed on the previous frame,
                        used for sticky-pair hysteresis. Pass None or []
                        on the first frame / to disable hysteresis.
        sticky_bonus  : cost discount applied to candidate pairs whose
                        centers are within sticky_tol_px of a prev_pair.
                        Keep small (≤ ~0.05) so a stale pair cannot
                        override strong geometric evidence of a new
                        correct pair.
        sticky_tol_px : pixel radius (in display space) within which a
                        candidate pair is considered the "same" as a
                        previous-frame pair. Should comfortably exceed
                        one-frame drone motion but stay below typical
                        inter-drone separation.

    Returns:
        stereo_pairs : list of (det_L, det_R) dicts ready for triangulation
        cam0_only    : list of cam0 dets with no valid cam1 match
        cam1_only    : list of cam1 dets with no valid cam0 match
    """
    from scipy.optimize import linear_sum_assignment

    cam0_only = []
    cam1_only = []

    # Grid coord selector — use rectified coords when they are present,
    # fall back to raw coords otherwise. All epipolar / disparity math
    # in this function is based on these values.
    def _gx(d):
        return d.get('grid_x_rect', d['grid_x'])

    def _gy(d):
        return d.get('grid_y_rect', d['grid_y'])

    # Step 1 — FOV pre-classification. Only detections visible to the
    # other camera are eligible for pairing.
    cam0_pairable = []
    for det_L in dets_L:
        if grid.in_fov(_gx(det_L), _gy(det_L), cam_id=1):
            cam0_pairable.append(det_L)
        else:
            cam0_only.append(det_L)

    cam1_pairable = []
    for det_R in dets_R:
        if grid.in_fov(_gx(det_R), _gy(det_R), cam_id=0):
            cam1_pairable.append(det_R)
        else:
            cam1_only.append(det_R)

    # Early out: no pairable candidates on at least one side.
    if not cam0_pairable or not cam1_pairable:
        cam0_only.extend(cam0_pairable)
        cam1_only.extend(cam1_pairable)
        return [], cam0_only, cam1_only

    prev_pairs = prev_pairs or []
    sticky_tol_sq = sticky_tol_px * sticky_tol_px

    def _is_sticky(dL, dR):
        """True if this candidate pair is close to any prev-frame pair."""
        for (cxL, cyL), (cxR, cyR) in prev_pairs:
            dL_sq = (dL['px'] - cxL) ** 2 + (dL['py'] - cyL) ** 2
            if dL_sq > sticky_tol_sq:
                continue
            dR_sq = (dR['px'] - cxR) ** 2 + (dR['py'] - cyR) ** 2
            if dR_sq <= sticky_tol_sq:
                return True
        return False

    # Step 2 — cost matrix. 1e6 marks "forbidden" pairs (epipolar gated).
    n_L = len(cam0_pairable)
    n_R = len(cam1_pairable)
    cost = np.full((n_L, n_R), 1e6, dtype=np.float64)

    for i, dL in enumerate(cam0_pairable):
        wL = max(dL['box'][2] - dL['box'][0], 1.0)
        cL = dL['conf']
        gyL = _gy(dL)
        gxL = _gx(dL)
        for j, dR in enumerate(cam1_pairable):
            dy = abs(gyL - _gy(dR))
            if dy > epipolar_tol:
                continue
            dx = abs(gxL - _gx(dR))

            # Box-size similarity — symmetric ratio in [0, 1], where
            # 0 = identical widths and 1 = one box is arbitrarily
            # larger than the other.
            wR = max(dR['box'][2] - dR['box'][0], 1.0)
            size_cost = 1.0 - (min(wL, wR) / max(wL, wR))

            # Confidence cost — low cost when both dets are high
            # confidence, high cost when either is marginal.
            conf_cost = 1.0 - 0.5 * (cL + dR['conf'])

            c = (dy
                 + disparity_weight * dx
                 + size_weight      * size_cost
                 + conf_weight      * conf_cost)

            if prev_pairs and _is_sticky(dL, dR):
                c = max(c - sticky_bonus, 0.0)

            cost[i, j] = c

    # Step 3 — global assignment.
    row_ind, col_ind = linear_sum_assignment(cost)

    stereo_pairs = []
    matched_L    = set()
    matched_R    = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e6:
            continue  # gated pair — the solver took it only because
                      # nothing else was cheaper; drop it.
        stereo_pairs.append((cam0_pairable[r], cam1_pairable[c]))
        matched_L.add(r)
        matched_R.add(c)

    # Unmatched pairable dets fall through to the single-cam pools.
    for i, dL in enumerate(cam0_pairable):
        if i not in matched_L:
            cam0_only.append(dL)
    for j, dR in enumerate(cam1_pairable):
        if j not in matched_R:
            cam1_only.append(dR)

    return stereo_pairs, cam0_only, cam1_only


# STAGE 4 -- KALMAN FILTER
# All tuning lives at the top of the file:
#   KF_MAX_PREDICT, KF_PROCESS_NOISE, KF_MEASUREMENT_NOISE
class DroneKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]], dtype=np.float32)
        self.kf.transitionMatrix    = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * KF_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * KF_MEASUREMENT_NOISE
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self.initialized    = False
        self._last_t        = None
        self._predict_count = 0

    def _set_dt(self, dt):
        F = np.eye(6, dtype=np.float32)
        F[0,3]=F[1,4]=F[2,5]=float(dt)
        self.kf.transitionMatrix = F

    def _init(self, xyz):
        self.kf.statePost = np.array(
            [xyz[0],xyz[1],xyz[2],0.,0.,0.], dtype=np.float32).reshape(6,1)
        self.initialized=True; self._predict_count=0
        print(f"[KF] Initialised at {xyz}")

    def correct(self, xyz_measured):
        xyz=np.array(xyz_measured, dtype=np.float32)
        now=time.time()
        if not self.initialized:
            self._init(xyz); self._last_t=now
            return xyz, np.zeros(3,dtype=np.float32), 0.0
        dt=max(now-self._last_t,1e-3)
        self._last_t=now; self._predict_count=0; self._set_dt(dt)
        self.kf.predict(); self.kf.correct(xyz.reshape(3,1))
        state=self.kf.statePost.flatten()
        pos,vel=state[:3],state[3:]
        return pos, vel, float(np.linalg.norm(vel))

    def predict(self):
        if not self.initialized: return None, None, 0.0
        self._predict_count+=1
        if self._predict_count > KF_MAX_PREDICT:
            self.kf.statePost[3:]=0.0
        now=time.time(); dt=max(now-self._last_t,1e-3)
        self._last_t=now; self._set_dt(dt); self.kf.predict()
        state=self.kf.statePost.flatten()
        return state[:3], state[3:], float(np.linalg.norm(state[3:]))


# Track state labels (TRACKING / SINGLE_CAM / COASTING) live in the TUNING
# block at the top of the file.

class Track:
    """
    Represents a single drone being tracked across frames.

    Each track owns:
      - A unique integer ID
      - Its own DroneKalmanFilter instance
      - Its last known shared grid coordinate (always available)
      - Its last known 3D position (only when has_3d is True)
      - A lifecycle state (SINGLE_CAM, TRACKING, COASTING)
      - Counters for consecutive hits and misses
    """
    _id_counter = 0   # class-level counter — increments for every new track

    def __init__(self, det, focal_px, xyz=None):
        """
        det : detection dict from get_all_drones() — provides initial
              grid coordinate, pixel position, confidence, cam_id
        xyz : 3D position from triangulation if immediately available,
              None for single-camera detections
        """
        Track._id_counter += 1
        self.id          = Track._id_counter

        self.kf          = DroneKalmanFilter()

        # Grid coordinate — always kept up to date regardless of has_3d
        self.grid_x      = det['grid_x']
        self.grid_y      = det['grid_y']

        # 3D state — only valid when has_3d is True
        self.pos         = None
        self.vel         = None
        self.speed       = 0.0
        self.has_3d      = False

        # Which camera(s) last saw this track
        self.cam_id      = det['cam_id']   # most recent camera that detected it

        # Last detection dicts per camera (for bounding box drawing)
        self.det_L       = det if det['cam_id'] == 0 else None
        self.det_R       = det if det['cam_id'] == 1 else None

        # Confidence (most recent)
        self.conf        = det['conf']

        # Lifecycle counters
        self.hit_streak  = 1      # consecutive frames with a detection
        self.missed      = 0      # consecutive frames without a detection

        # Stability latch: once hit_streak reaches STABILITY_FRAMES, this
        # flips True and never flips back. Confirmed tracks remain
        # confirmed for the rest of their lifetime (until MAX_MISSED
        # deletes them), so brief single-frame dropouts don't make them
        # disappear from the display.
        self._confirmed  = False

        # State
        self.state       = SINGLE_CAM

        # Physical size calibration
        self.size_state      = SIZE_UNCALIBRATED
        self.size_buffer     = []        # per-frame raw size estimates
        self.focal_px        = focal_px  # needed for physical size calculation
        self.physical_size_m = None      # locked average — set once, never changed

        # If 3D position is available at birth, initialize immediately
        if xyz is not None:
            self._apply_3d(xyz)

        print(f"[Track] ID={self.id} spawned  state={self.state}  "
              f"grid=({self.grid_x:.3f}, {self.grid_y:.3f})  "
              f"cam={self.cam_id}")

    def _apply_3d(self, xyz):
        """Internal — apply a triangulated 3D measurement to the Kalman filter."""
        self.pos, self.vel, self.speed = self.kf.correct(xyz)
        self.has_3d = True
        self.state  = TRACKING
    
    def _update_size(self, det_L, focal_px=None):
        """
        Compute one physical size estimate from the current cam0 bounding
        box and the current triangulated depth, then add it to the buffer.
        Once SIZE_CALIB_FRAMES samples are collected, filter outliers,
        average, and lock the size.

        physical_size_m = (box_width_px * depth_m) / focal_px

        focal_px is taken from self.focal_px if not provided explicitly.
        """
        fp = focal_px if focal_px is not None else self.focal_px
        if fp is None or fp <= 0:
            return

        depth_m   = float(self.pos[2])
        box_w_px  = det_L['box'][2] - det_L['box'][0]   # x2 - x1

        if depth_m <= 0 or box_w_px <= 0:
            return

        estimate = (box_w_px * depth_m) / fp

        if self.size_state == SIZE_UNCALIBRATED:
            self.size_state = SIZE_CALIBRATING

        self.size_buffer.append(estimate)

        if len(self.size_buffer) >= SIZE_CALIB_FRAMES:
            self._lock_size()

    def _lock_size(self):
        """
        Filter outliers from size_buffer using standard deviation,
        then average the remaining values and lock the result.
        Called automatically once SIZE_CALIB_FRAMES samples are collected.
        """
        arr  = np.array(self.size_buffer, dtype=np.float32)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))

        if std > 0:
            filtered = arr[np.abs(arr - mean) <= SIZE_OUTLIER_STD * std]
        else:
            filtered = arr

        self.physical_size_m = float(np.mean(filtered)) if len(filtered) > 0 else mean
        self.size_state      = SIZE_LOCKED
        self.size_buffer     = []   # free memory — no longer needed

        print(f"[Track ID={self.id}] size locked: "
              f"{self.physical_size_m:.4f} m  "
              f"(from {len(filtered)} samples after outlier removal)")

    def update(self, det_L=None, det_R=None, xyz=None):
        """
        Called each frame when this track has a detection.

        det_L : detection dict from cam0, or None
        det_R : detection dict from cam1, or None
        xyz   : triangulated 3D position, or None

        At least one of det_L or det_R must be provided.
        xyz is only provided when both det_L and det_R are provided.

        Display semantics: a bounding box is drawn on a camera only if
        that camera produced a detection THIS frame. Any prior det_L /
        det_R from a previous frame is cleared on entry so stale boxes
        are never rendered.
        """
        # Strict display rule: drop any stale per-camera detection from
        # previous frames. We rebuild det_L / det_R below from only this
        # frame's inputs.
        self.det_L = None
        self.det_R = None

        if det_L is not None:
            self.det_L  = det_L
            self.grid_x = det_L['grid_x']
            self.grid_y = det_L['grid_y']
            self.cam_id = 0

        if det_R is not None:
            self.det_R  = det_R
            # Grid coordinate from cam0 takes priority when both available
            # because cam0 is the reference camera for stereo
            if det_L is None:
                self.grid_x = det_R['grid_x']
                self.grid_y = det_R['grid_y']
                self.cam_id = 1

        # Confidence reflects only this frame's detection(s) — no carry-over
        # from previous frames. If both cameras saw it, take the better view.
        if det_L is not None and det_R is not None:
            self.conf = max(det_L['conf'], det_R['conf'])
        elif det_L is not None:
            self.conf = det_L['conf']
        elif det_R is not None:
            self.conf = det_R['conf']

        # Apply 3D update if triangulation succeeded
        if xyz is not None:
            self._apply_3d(xyz)
        else:
            # Matched to a single-camera detection — no new 3D measurement.
            # The KF is still advanced at most ONCE per frame per track
            # (see TrackManager.update). For a has_3d track in COASTING we
            # run a single kf.predict() here so the displayed Z / speed
            # continue to evolve from the last stereo fix instead of
            # freezing. For a track that has never had 3D, there is no
            # KF state to advance.
            if self.has_3d:
                self.state = COASTING
                self.pos, self.vel, self.speed = self.kf.predict()
            else:
                self.state = SINGLE_CAM

        # Update lifecycle counters
        self.hit_streak += 1
        self.missed      = 0

        # Physical size calibration — only runs when we have 3D and a cam0 box
        if (xyz is not None
                and det_L is not None
                and self.size_state != SIZE_LOCKED
                and self.pos is not None):
            self._update_size(det_L)

    def predict(self):
        """
        Called each frame when this track has NO detection at all.
        Advances the Kalman filter forward and increments missed counter.

        hit_streak / missed behaviour:
          - Pre-confirmation (self._confirmed is False):
              Up to STABILITY_MAX_GAPS consecutive misses are tolerated
              without touching hit_streak — the track is still building
              toward promotion and one-off YOLO drops shouldn't wipe
              progress. Once consecutive misses exceed the tolerance,
              the track is considered unstable and hit_streak is reset
              to 0 so it has to re-accumulate from scratch.
          - Post-confirmation (self._confirmed is True):
              The latch in is_confirmed keeps the track displayed through
              brief occlusions; hit_streak decrements here are retained
              only for diagnostic continuity. MAX_MISSED, evaluated in
              TrackManager.update, is still the deletion gate.
        """
        self.missed     += 1
        self.det_L       = None
        self.det_R       = None

        if self._confirmed:
            # Decrement is cosmetic once latched.
            self.hit_streak = max(self.hit_streak - 1, 0)
        else:
            if self.missed > STABILITY_MAX_GAPS:
                # Too many gaps during stability window — progress wiped.
                self.hit_streak = 0
            # else: leave hit_streak alone; misses within tolerance
            # don't erode the promotion count.

        # No detection this frame: the last observed confidence is stale
        # and should not be shown as if it still describes the drone's
        # current visibility. Reset to 0 so the UI card reflects the
        # coasting / fully-missed status.
        self.conf        = 0.0

        if self.has_3d:
            self.pos, self.vel, self.speed = self.kf.predict()
            self.state = COASTING
        else:
            self.state = SINGLE_CAM   # lost without ever having 3D

    @property
    def is_confirmed(self):
        """
        Latching confirmation.

        A track is promoted to confirmed the first time hit_streak
        reaches STABILITY_FRAMES. Once promoted, it stays confirmed for
        the rest of its lifetime so short detection dropouts don't
        cause it to flicker in and out of the UI. A confirmed track is
        still deletable via MAX_MISSED if it goes dark for too long.
        """
        if not self._confirmed and self.hit_streak >= STABILITY_FRAMES:
            self._confirmed = True
        return self._confirmed



class TrackManager:
    """
    Owns and manages all active Track objects.

    Each frame, the manager:
      1. Receives stereo pairs (with 3D positions) and single-camera detections
      2. Matches detections to existing tracks using grid proximity +
         optional velocity direction cost (Hungarian algorithm)
      3. Updates matched tracks, spawns new tracks for unmatched detections
      4. Calls predict() on tracks with no detection this frame
      5. Deletes tracks that have been missing too long
      6. Returns only confirmed tracks for display and logging
    """

    def __init__(self, calib, grid, scale_x, scale_y):
        self.calib    = calib
        self.grid     = grid
        self.scale_x  = scale_x
        self.scale_y  = scale_y
        self.tracks   = []

        # Focal length in display pixels — needed for physical size calculation
        # and for converting implied grid displacement into metres. We use
        # the raw K_L[0,0] rather than the rectified P1[0,0] because some
        # calibrations ship with a garbage P1 (see the sanity check in
        # StereoCalibration.__init__). K_L is always a first-order correct
        # description of the camera's optics.
        calib_w        = calib.image_size[0]
        scale_to_disp  = CAM_W / calib_w
        self.focal_px  = calib.K_L[0, 0] * scale_to_disp

        # Wall-clock delta between successive update() calls. Initialized
        # to the nominal inter-frame time and refreshed at the top of each
        # update(). Used by cost functions that need to convert a per-frame
        # displacement into a speed. Using measured dt rather than the
        # FPS constant keeps those costs accurate when the pipeline is
        # running slower than 30 FPS under load.
        self._dt           = 1.0 / FPS
        self._last_update_t = None

        # Each TrackManager owns its own track ID namespace. Resetting the
        # class-level counter here ensures re-instantiating the manager in
        # the same Python process starts IDs at 1, not from the previous
        # run's last value.
        Track._id_counter = 0

        # Zombie list — recently-deceased confirmed tracks available for
        # resurrection. Each entry is (Track, died_at_wallclock_time).
        # Pruned against ZOMBIE_TTL_SEC on every update(). See
        # _revive_via_stereo / _revive_via_single and the notes near
        # ZOMBIE_TTL_SEC at the top of the file.
        self.zombies = []

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _triangulate(self, det_L, det_R):
        """
        Triangulate a stereo pair.

        When point-level rectification is active (the default) each
        detection already carries px_rect_cal / py_rect_cal — rectified
        calibration-space pixels — and we feed them straight into
        triangulate()'s fast path. Otherwise we fall back to the raw
        display-to-calibration scaling and let triangulate() do the
        per-point undistortPoints internally.

        Returns xyz array or None if depth is out of valid range.
        """
        if 'px_rect_cal' in det_L and 'px_rect_cal' in det_R:
            cL_cal = (det_L['px_rect_cal'], det_L['py_rect_cal'])
            cR_cal = (det_R['px_rect_cal'], det_R['py_rect_cal'])
        else:
            cL_cal = (det_L['px'] * self.scale_x, det_L['py'] * self.scale_y)
            cR_cal = (det_R['px'] * self.scale_x, det_R['py'] * self.scale_y)

        xyz = self.calib.triangulate(cL_cal, cR_cal)

        if not (MIN_Z_M < xyz[2] < MAX_Z_M):
            return None   # depth out of valid range
        return xyz

    def _grid_distance(self, track, det):
        """Euclidean distance in shared grid space between a track and a detection."""
        dx = track.grid_x - det['grid_x']
        dy = track.grid_y - det['grid_y']
        return np.sqrt(dx*dx + dy*dy)

    def _direction_cost(self, track, new_grid_x, new_grid_y):
        """
        Secondary matching cost based on velocity direction consistency.
        Returns a value in [0, 1]:
          0   = detection is in the exact direction the track is moving
          0.5 = perpendicular
          1   = detection is in the opposite direction
        Returns 0 if track speed is below MIN_SPEED_DIR (hovering/slow).
        Only meaningful when track has_3d — uses 3D velocity vector
        projected onto the grid plane (X and Z components).

        All other matching costs in _build_cost_matrix are in [0, 1], so
        normalizing this one by 2 keeps the weight values in
        WEIGHTS_WITH_SIZE / WEIGHTS_NO_SIZE apples-to-apples across
        signals.
        """
        if not track.has_3d or track.vel is None:
            return 0.0
        speed = float(np.linalg.norm(track.vel))
        if speed < MIN_SPEED_DIR:
            return 0.0

        # Implied movement direction in grid space
        implied = np.array([new_grid_x - track.grid_x,
                            new_grid_y - track.grid_y], dtype=np.float32)
        implied_norm = np.linalg.norm(implied)
        if implied_norm < 1e-6:
            return 0.0

        # Project 3D velocity onto grid plane (X and Z → grid X and Y)
        vel_grid = np.array([track.vel[0], track.vel[2]], dtype=np.float32)
        vel_norm = np.linalg.norm(vel_grid)
        if vel_norm < 1e-6:
            return 0.0

        cos_sim = np.dot(implied / implied_norm, vel_grid / vel_norm)
        return float((1.0 - cos_sim) / 2.0)   # 0=same, 0.5=perp, 1=opposite

    def _select_weights(self):
        """
        Check all pairs of SIZE_LOCKED tracks. If any pair has sufficiently
        different physical size signatures, size has discriminating power
        and WEIGHTS_WITH_SIZE is returned. Otherwise WEIGHTS_NO_SIZE.
        """
        locked = [t for t in self.tracks if t.size_state == SIZE_LOCKED]

        if len(locked) < 2:
            return WEIGHTS_NO_SIZE

        for i in range(len(locked)):
            for j in range(i + 1, len(locked)):
                diff = abs(locked[i].physical_size_m - locked[j].physical_size_m)
                if diff > SIZE_SIMILARITY_THR:
                    return WEIGHTS_WITH_SIZE

        return WEIGHTS_NO_SIZE

    def _speed_consistency_cost(self, track, det):
        """
        Penalizes matches where the implied speed change is physically
        implausible given the track's current speed.

        Computes the ratio between the implied speed (how fast the drone
        would have to move to reach this detection in one frame) and the
        track's current Kalman-estimated speed. A large ratio means the
        detection requires an implausible acceleration.

        Returns a value in [0, 1]:
          0 = implied speed is consistent with current speed
          1 = implied speed is completely implausible
        Only active when track has_3d and speed > MIN_SPEED_DIR.

        Uses the manager's measured inter-frame dt rather than the FPS
        constant so the implied speed stays accurate under load.
        """
        if not track.has_3d or track.speed < MIN_SPEED_DIR:
            return 0.0

        # Implied displacement in grid space over one frame
        dx = det['grid_x'] - track.grid_x
        dy = det['grid_y'] - track.grid_y
        implied_grid_dist = np.sqrt(dx*dx + dy*dy)

        # Convert implied grid displacement to approximate metres.
        # Grid is normalized so we scale by a rough scene width estimate.
        # We use the track's current depth (Z) and focal length to get
        # the physical width of the scene at that depth.
        if track.pos is None:
            return 0.0

        scene_w_m     = (CAM_W / self.focal_px) * float(track.pos[2])
        dt            = max(self._dt, 1e-3)
        implied_spd_m = implied_grid_dist * scene_w_m / dt

        # Ratio of implied speed to current speed
        ratio = implied_spd_m / (track.speed + 1e-6)

        # Cost increases as ratio deviates from 1.0 (perfect consistency)
        # Clamped to [0, 1]
        cost = min(abs(ratio - 1.0) / 5.0, 1.0)
        return float(cost)

    def _size_consistency_cost(self, track, det_L, weights):
        """
        Penalizes matches where the detection's implied physical size
        is inconsistent with the track's locked size signature.

        Only active when:
          - weights['size'] > 0 (size has discriminating power this frame)
          - track.size_state == SIZE_LOCKED
          - det_L is not None (need a cam0 box for size estimation)
          - track has a valid depth (has_3d)

        Returns a value in [0, 1]:
          0 = detection size perfectly matches track's locked size
          1 = detection size is maximally inconsistent
        """
        if weights['size'] == 0.0:
            return 0.0
        if track.size_state != SIZE_LOCKED:
            return 0.0
        if det_L is None or not track.has_3d:
            return 0.0

        depth_m  = float(track.pos[2])
        box_w_px = det_L['box'][2] - det_L['box'][0]

        if depth_m <= 0 or box_w_px <= 0:
            return 0.0

        implied_size = (box_w_px * depth_m) / self.focal_px
        diff         = abs(implied_size - track.physical_size_m)

        # Normalize by the locked size so the cost is scale-independent
        cost = min(diff / (track.physical_size_m + 1e-6), 1.0)
        return float(cost)
    
    def _build_cost_matrix(self, tracks, detections, weights, det_L_map):
        """
        Build the N_tracks x N_detections cost matrix.

        tracks      : list of Track objects
        detections  : list of detection dicts (det_L for stereo, any det for single-cam)
        weights     : dict from _select_weights() — determines active signals
        det_L_map   : dict mapping detection index to its cam0 det dict (or None)
                      needed for size cost which requires a cam0 bounding box

        Each cell is a weighted sum of normalized [0,1] cost signals.
        Cells exceeding MAX_MATCH_DIST in grid distance are set to 1e6
        to prevent bad matches being forced by the Hungarian algorithm.
        """
        n_t  = len(tracks)
        n_d  = len(detections)
        cost = np.full((n_t, n_d), fill_value=1e6, dtype=np.float64)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):

                # Primary gate — grid distance
                dist = self._grid_distance(track, det)
                if dist > MAX_MATCH_DIST:
                    continue

                # Normalize grid distance to [0, 1] over MAX_MATCH_DIST
                pos_cost  = dist / MAX_MATCH_DIST

                dir_cost  = self._direction_cost(
                                track, det['grid_x'], det['grid_y'])

                spd_cost  = self._speed_consistency_cost(track, det)

                size_cost = self._size_consistency_cost(
                                track, det_L_map.get(j), weights)

                cost[i, j] = (weights['pos']  * pos_cost
                            + weights['dir']  * dir_cost
                            + weights['spd']  * spd_cost
                            + weights['size'] * size_cost)

        return cost

    def _run_hungarian(self, tracks, detections, weights, det_L_map):
        """
        Runs the Hungarian algorithm on the cost matrix.
        Returns:
          matched   : list of (track_idx, det_idx) pairs
          unmatched_tracks : list of track indices with no detection
          unmatched_dets   : list of detection indices with no track
        """
        from scipy.optimize import linear_sum_assignment

        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = self._build_cost_matrix(tracks, detections, weights, det_L_map)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched          = []
        unmatched_tracks = []
        unmatched_dets   = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e6:
                unmatched_tracks.append(r)
            else:
                matched.append((r, c))
                unmatched_dets.remove(c)

        matched_track_indices = {m[0] for m in matched}
        for r in range(len(tracks)):
            if r not in matched_track_indices and r not in unmatched_tracks:
                unmatched_tracks.append(r)

        return matched, unmatched_tracks, unmatched_dets

    # ------------------------------------------------------------------
    # ZOMBIE RE-IDENTIFICATION
    # ------------------------------------------------------------------
    # The trio _prune_zombies / _revive_via_stereo / _revive_via_single
    # implements the "same drone came back after a brief gap" path. See
    # the ZOMBIE_* constants near the top of the file for rationale and
    # the call sites in update() for where each runs.

    def _prune_zombies(self, now):
        """Drop zombies older than ZOMBIE_TTL_SEC."""
        self.zombies = [
            (z, died_at) for (z, died_at) in self.zombies
            if now - died_at <= ZOMBIE_TTL_SEC
        ]

    def _revive_via_stereo(self, unmatched_det_indices, stereo_xyz, now):
        """
        Try to resurrect zombies using unmatched stereo detections.

        Only zombies that had a valid 3D state at death (has_3d) are
        eligible — those are the cases where we can meaningfully
        extrapolate position forward through the gap. The gate is
        Euclidean distance in metres between the zombie's predicted
        position (pos + vel * dt_since_death) and the new detection's
        triangulated xyz.

        A Hungarian solve resolves ambiguity when multiple zombies
        could claim multiple detections.

        Returns a list of (zombie_track, det_idx) resurrections.
        Resurrected zombies are NOT removed from self.zombies here —
        the caller is responsible for that after applying the update.
        """
        if not self.zombies or not unmatched_det_indices:
            return []

        candidates = [
            (z, died_at) for (z, died_at) in self.zombies
            if z.has_3d and z.pos is not None
        ]
        if not candidates:
            return []

        from scipy.optimize import linear_sum_assignment

        n_z = len(candidates)
        n_d = len(unmatched_det_indices)
        cost = np.full((n_z, n_d), 1e6, dtype=np.float64)

        for i, (z, died_at) in enumerate(candidates):
            dt = now - died_at
            vel = z.vel if z.vel is not None else np.zeros(3)
            pred_pos = np.asarray(z.pos, dtype=np.float64) + \
                       np.asarray(vel, dtype=np.float64) * dt
            for j, d_idx in enumerate(unmatched_det_indices):
                xyz, _, _ = stereo_xyz[d_idx]
                dist = float(np.linalg.norm(pred_pos - np.asarray(xyz)))
                if dist > ZOMBIE_MAX_MATCH_DIST_M:
                    continue
                cost[i, j] = dist

        if not (cost < 1e6).any():
            return []

        row_ind, col_ind = linear_sum_assignment(cost)
        resurrections = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e6:
                continue
            zombie, died_at = candidates[r]
            resurrections.append(
                (zombie, unmatched_det_indices[c], now - died_at)
            )
        return resurrections

    def _revive_via_single(self, unmatched_det_indices, all_single, now):
        """
        Try to resurrect zombies using unmatched single-camera detections.

        Gate is grid-space distance from the zombie's last known grid
        position (ZOMBIE_MAX_MATCH_GRID). This path covers the case
        where a drone was occluded in both cameras, then only one
        camera catches it on re-entry (typical at the overlap
        boundary).

        Because we have no fresh 3D from a single-cam detection, we
        do not velocity-extrapolate the zombie — we simply match
        against its last grid coordinate. That works well for
        drones moving slowly during the gap and is why the gate is
        intentionally tight (0.08 grid units, same as the live-track
        MAX_MATCH_DIST).

        Returns list of (zombie_track, det_idx) resurrections.
        """
        if not self.zombies or not unmatched_det_indices:
            return []

        from scipy.optimize import linear_sum_assignment

        n_z = len(self.zombies)
        n_d = len(unmatched_det_indices)
        cost = np.full((n_z, n_d), 1e6, dtype=np.float64)

        for i, (z, _died_at) in enumerate(self.zombies):
            for j, d_idx in enumerate(unmatched_det_indices):
                det = all_single[d_idx]
                dgx = z.grid_x - det['grid_x']
                dgy = z.grid_y - det['grid_y']
                dist = float(np.hypot(dgx, dgy))
                if dist > ZOMBIE_MAX_MATCH_GRID:
                    continue
                cost[i, j] = dist

        if not (cost < 1e6).any():
            return []

        row_ind, col_ind = linear_sum_assignment(cost)
        resurrections = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e6:
                continue
            zombie, died_at = self.zombies[r]
            resurrections.append(
                (zombie, unmatched_det_indices[c], now - died_at)
            )
        return resurrections

    # ------------------------------------------------------------------
    # MAIN UPDATE — called once per frame
    # ------------------------------------------------------------------

    def update(self, stereo_pairs, cam0_only, cam1_only):
        """
        Main entry point — called once per frame.
        stereo_pairs : list of (det_L, det_R) tuples
        cam0_only    : list of cam0-only detection dicts
        cam1_only    : list of cam1-only detection dicts
        Returns list of confirmed Track objects.

        Each track's Kalman filter is advanced exactly ONCE per frame:
          - matched with stereo xyz  -> kf.correct() inside Track._apply_3d
          - matched single-cam only  -> one kf.predict() inside Track.update
                                        (has_3d branch, for display drift)
          - unmatched                -> kf.predict() inside Track.predict()
        Similarly, hit_streak / missed counters are modified exactly once per
        track per frame (by either Track.update or Track.predict, never both).
        """

        # Refresh measured inter-frame dt — used by cost functions that
        # convert a per-frame displacement to a speed.
        now = time.time()
        if self._last_update_t is not None:
            self._dt = max(now - self._last_update_t, 1e-3)
        self._last_update_t = now

        # Step 1 — select weight set for this frame (based on current track state)
        weights = self._select_weights()

        # Step 2 — handle stereo pairs
        # Triangulation may fail (depth out of [MIN_Z_M, MAX_Z_M]). A failed
        # pair is split: det_L is treated as a cam0-only detection and
        # det_R as a cam1-only detection. This keeps each 2D observation
        # available for single-cam matching below instead of discarding
        # them or forcing a track into COASTING despite both cameras
        # seeing the drone.
        stereo_dets     = []
        stereo_xyz      = {}
        det_L_map_s     = {}   # index → det_L for size cost
        fallback_cam0   = []
        fallback_cam1   = []

        for det_L, det_R in stereo_pairs:
            xyz = self._triangulate(det_L, det_R)
            if xyz is None:
                fallback_cam0.append(det_L)
                fallback_cam1.append(det_R)
                continue
            idx = len(stereo_dets)
            stereo_dets.append(det_L)
            stereo_xyz[idx]  = (xyz, det_L, det_R)
            det_L_map_s[idx] = det_L

        matched_s, unmatched_t_s, unmatched_d_s = self._run_hungarian(
            self.tracks, stereo_dets, weights, det_L_map_s)

        # Track identity (by object id) of every track that got matched this frame
        matched_track_oids = set()

        for t_idx, d_idx in matched_s:
            track       = self.tracks[t_idx]
            xyz, dL, dR = stereo_xyz[d_idx]
            track.update(det_L=dL, det_R=dR, xyz=xyz)
            matched_track_oids.add(id(track))

        # Zombie resurrection via stereo. Runs BEFORE the new-track
        # spawn block below so any unmatched stereo detection whose xyz
        # lines up with a recent zombie's predicted position inherits
        # the zombie's ID instead of getting a fresh one.
        self._prune_zombies(now)
        resurrected_zombie_oids = set()
        stereo_resurrections = self._revive_via_stereo(
            unmatched_d_s, stereo_xyz, now)
        stereo_used_dets = set()
        for zombie, d_idx, age_sec in stereo_resurrections:
            xyz, dL, dR = stereo_xyz[d_idx]
            zombie.missed = 0
            zombie.update(det_L=dL, det_R=dR, xyz=xyz)
            self.tracks.append(zombie)
            matched_track_oids.add(id(zombie))
            resurrected_zombie_oids.add(id(zombie))
            stereo_used_dets.add(d_idx)
            print(f"[Track ID={zombie.id}] resurrected (stereo) after "
                  f"{age_sec:.2f}s gap")
        # Only unmatched stereo dets that were NOT resurrected spawn a
        # brand-new track below.
        unmatched_d_s = [d for d in unmatched_d_s if d not in stereo_used_dets]

        # Spawn new tracks for unmatched stereo detections. Both camera
        # detections are attached so the cam1 box can render on the
        # spawn frame once the track confirms.
        for d_idx in unmatched_d_s:
            xyz, dL, dR = stereo_xyz[d_idx]
            new_track = Track(dL, focal_px=self.focal_px, xyz=xyz)
            new_track.det_R = dR
            self.tracks.append(new_track)
            matched_track_oids.add(id(new_track))  # just-spawned; do not predict

        # Step 3 — handle single-camera detections
        # Only tracks not already matched to a stereo pair are candidates.
        # Failed-triangulation fallbacks are merged into the single-cam
        # pools; we build local copies so the caller's lists (used for
        # display counts) are not mutated.
        unmatched_tracks = [t for t in self.tracks
                            if id(t) not in matched_track_oids]

        cam0_pool = list(cam0_only) + fallback_cam0
        cam1_pool = list(cam1_only) + fallback_cam1
        all_single  = cam0_pool + cam1_pool
        det_L_map_c = {}   # size cost not used for single-cam (no depth)

        matched_sc, unmatched_t_sc, unmatched_d_sc = self._run_hungarian(
            unmatched_tracks, all_single, weights, det_L_map_c)

        for t_idx, d_idx in matched_sc:
            track = unmatched_tracks[t_idx]
            det   = all_single[d_idx]
            if det['cam_id'] == 0:
                track.update(det_L=det)
            else:
                track.update(det_R=det)
            matched_track_oids.add(id(track))

        # Zombie resurrection via single-cam. Only zombies that were NOT
        # already claimed by the stereo resurrection pass above are
        # eligible (we filter them out of self.zombies at the end of
        # update(), but the in-flight resurrected_zombie_oids set is
        # consulted here so we don't double-claim one inside a frame).
        eligible_zombies = [
            (z, t) for (z, t) in self.zombies
            if id(z) not in resurrected_zombie_oids
        ]
        saved_zombies = self.zombies
        self.zombies = eligible_zombies
        single_resurrections = self._revive_via_single(
            unmatched_d_sc, all_single, now)
        self.zombies = saved_zombies

        single_used_dets = set()
        for zombie, d_idx, age_sec in single_resurrections:
            det = all_single[d_idx]
            zombie.missed = 0
            if det['cam_id'] == 0:
                zombie.update(det_L=det)
            else:
                zombie.update(det_R=det)
            self.tracks.append(zombie)
            matched_track_oids.add(id(zombie))
            resurrected_zombie_oids.add(id(zombie))
            single_used_dets.add(d_idx)
            print(f"[Track ID={zombie.id}] resurrected (single-cam) after "
                  f"{age_sec:.2f}s gap")
        unmatched_d_sc = [d for d in unmatched_d_sc if d not in single_used_dets]

        # Spawn new tracks for unmatched single-camera detections
        for d_idx in unmatched_d_sc:
            det = all_single[d_idx]
            new_track = Track(det, focal_px=self.focal_px, xyz=None)
            self.tracks.append(new_track)
            matched_track_oids.add(id(new_track))  # just-spawned; do not predict

        # Step 4 — predict (advance KF, decrement streak) only the tracks that
        # were not matched to any detection this frame. This is the single point
        # where unmatched tracks age.
        for track in self.tracks:
            if id(track) not in matched_track_oids:
                track.predict()

        # Step 5 — retire tracks missing too long.
        # Confirmed tracks move to the zombie list (available for
        # resurrection for up to ZOMBIE_TTL_SEC if a new detection
        # appears in approximately the same area). Unconfirmed tracks
        # are dropped outright — they were never stable enough to
        # warrant carrying an ID across a gap.
        surviving = []
        for t in self.tracks:
            if t.missed <= MAX_MISSED:
                surviving.append(t)
            elif t.is_confirmed:
                self.zombies.append((t, now))
            # else: unconfirmed → discarded silently
        self.tracks = surviving

        # Also remove zombies that were resurrected earlier this frame
        # (they are back in self.tracks with missed=0).
        if resurrected_zombie_oids:
            self.zombies = [
                (z, t) for (z, t) in self.zombies
                if id(z) not in resurrected_zombie_oids
            ]

        # Step 6 — return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed]



# DISPLAY HELPERS

def draw_text(img, text, pos, scale=0.5, color=C_BLACK,
              thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)



# UI layout sizes, font scales, colour palette, and the per-state UI_COLOR
# map all live in the TUNING block at the top of the file.


# ==============================================================================
# CAMERA FRAME ANNOTATION
# Draws bounding boxes and ID labels directly onto a camera frame.
# Called once per frame per camera before compositing.
# ==============================================================================

def annotate_frames(frame_L, frame_R, confirmed_tracks):
    """
    Draw bounding boxes and ID labels on both camera frames.

    frame_L, frame_R  : OpenCV BGR frames (modified in place)
    confirmed_tracks  : list of confirmed Track objects from TrackManager

    Box color is determined by track state via UI_COLOR.
    Each box is labeled with the drone ID and confidence score.
    """
    for track in confirmed_tracks:
        color = UI_COLOR.get(track.state, UI_COLOR[COASTING])

        # Camera 0 frame
        if track.det_L is not None:
            x1, y1, x2, y2 = track.det_L['box']
            cv2.rectangle(frame_L, (x1, y1), (x2, y2), color, 2)
            draw_text(frame_L,
                      f"ID{track.id}  {track.det_L['conf']:.2f}",
                      (x1, max(y1 - 6, 14)), 0.5, color, 1)

        # Camera 1 frame
        if track.det_R is not None:
            x1, y1, x2, y2 = track.det_R['box']
            cv2.rectangle(frame_R, (x1, y1), (x2, y2), color, 2)
            draw_text(frame_R,
                      f"ID{track.id}  {track.det_R['conf']:.2f}",
                      (x1, max(y1 - 6, 14)), 0.5, color, 1)


# ==============================================================================
# INFO BAR
# Thin horizontal bar between the camera feeds and the drone cards.
# Shows global system health at a glance.
# ==============================================================================

def draw_info_bar(bar, fps, n_tracks, n_stereo, n_single):
    """
    Draw the global info bar.

    bar      : blank image of shape (UI_BAR_H, CAM_W*2, 3)
    fps      : current averaged FPS
    n_tracks : number of confirmed active tracks
    n_stereo : number of stereo pairs this frame
    n_single : number of single-camera detections this frame
    """
    bar[:] = UI_COLOR['bar_bg']
    cv2.line(bar, (0, 0), (bar.shape[1], 0), UI_COLOR['border'], 1)
    cv2.line(bar, (0, UI_BAR_H - 1), (bar.shape[1], UI_BAR_H - 1),
             UI_COLOR['border'], 1)

    # Left side — FPS
    draw_text(bar, f"FPS  {fps:.1f}",
              (12, UI_BAR_H - 10), 0.45, UI_COLOR['bar_text'], 1)

    # Center items — track and detection counts
    items = [
        f"ACTIVE TRACKS : {n_tracks}",
        f"STEREO PAIRS  : {n_stereo}",
        f"SINGLE CAM    : {n_single}",
    ]
    # Space items evenly across the bar width
    total_w   = bar.shape[1]
    spacing   = total_w // (len(items) + 1)
    for i, text in enumerate(items):
        x = spacing * (i + 1) - 80
        draw_text(bar, text, (x, UI_BAR_H - 10),
                  0.38, UI_COLOR['bar_info'], 1)

    # Right side — color legend
    legend = [
        (TRACKING,   "TRACKING"),
        (SINGLE_CAM, "SINGLE CAM"),
        (COASTING,   "COASTING"),
    ]
    x = total_w - 320
    for state, label in legend:
        color = UI_COLOR[state]
        cv2.rectangle(bar, (x, 10), (x + 12, UI_BAR_H - 10), color, -1)
        draw_text(bar, label, (x + 16, UI_BAR_H - 10),
                  0.32, UI_COLOR['label'], 1)
        x += 110


# ==============================================================================
# DRONE CARD
# One card per confirmed track in the bottom panel.
# ==============================================================================

def draw_drone_card(panel, x_start, card_w, track=None):
    """
    Draw a single drone card into a region of the bottom panel.

    panel    : the full bottom panel image (drawn into in place)
    x_start  : left pixel edge of this card's region
    card_w   : width of this card's region in pixels
    track    : a confirmed Track object, or None for an empty slot

    The card is self-contained — all positioning is relative to x_start.
    To add a new data row, add one entry to the `rows` list below.
    To change a label or unit, edit that entry — nothing else needs changing.
    """
    h = panel.shape[0]
    p = UI_CARD_PADDING

    # Card background
    cv2.rectangle(panel,
                  (x_start, 0), (x_start + card_w, h),
                  UI_COLOR['panel'], -1)

    # Vertical divider between cards
    if x_start > 0:
        cv2.line(panel, (x_start, 0), (x_start, h), UI_COLOR['border'], 1)

    # Empty slot
    if track is None:
        draw_text(panel, "-- NO DRONE --",
                  (x_start + card_w // 2 - 60, h // 2),
                  0.5, UI_COLOR['empty'], 1)
        return

    # Card border color matches track state
    border_color = UI_COLOR.get(track.state, UI_COLOR[COASTING])
    cv2.rectangle(panel,
                  (x_start + 2, 2),
                  (x_start + card_w - 2, h - 2),
                  border_color, 2)

    # Title bar
    title_y = 32
    cv2.rectangle(panel,
                  (x_start, 0), (x_start + card_w, title_y + 6),
                  UI_COLOR['title_bg'], -1)
    draw_text(panel,
              f"DRONE  ID {track.id}",
              (x_start + p, title_y),
              UI_TITLE_SCALE, border_color, 1)

    # State label — right side of title bar
    draw_text(panel,
              track.state,
              (x_start + card_w - 110, title_y),
              UI_STATE_SCALE, UI_COLOR['label'], 1)

    cv2.line(panel,
             (x_start, title_y + 8),
             (x_start + card_w, title_y + 8),
             UI_COLOR['border'], 1)

    # ── Data rows ──────────────────────────────────────────────────────
    # To add a new row: append a (label, value_string) tuple to this list.
    # To remove a row: delete its entry.
    # To reorder rows: reorder the list.
    # Nothing outside this list needs to change.

    if track.has_3d:
        dist = float(np.linalg.norm(track.pos))
        cam_str = "BOTH"
        rows = [
            ("X OFFSET",  f"{track.pos[0]:+.3f} m"),
            ("Y OFFSET",  f"{track.pos[1]:+.3f} m"),
            ("DEPTH  Z",  f"{track.pos[2]:.3f} m"),
            ("DISTANCE",  f"{dist:.3f} m"),
            ("SPEED",     f"{track.speed:.2f} m/s"),
            ("CONF",      f"{track.conf:.2f}"),
            ("CAMERA",    cam_str),
        ]
    else:
        cam_str = f"CAM {track.cam_id}"
        rows = [
            ("X OFFSET",  "--"),
            ("Y OFFSET",  "--"),
            ("DEPTH  Z",  "--"),
            ("DISTANCE",  "--"),
            ("SPEED",     "--"),
            ("CONF",      f"{track.conf:.2f}"),
            ("CAMERA",    cam_str),
        ]

    # Render each row
    row_x_label = x_start + p
    row_x_value = x_start + card_w // 2
    row_y_start = title_y + 8 + UI_ROW_H

    for i, (label, value) in enumerate(rows):
        y = row_y_start + i * UI_ROW_H

        # Alternating row background for readability
        if i % 2 == 0:
            cv2.rectangle(panel,
                          (x_start + 3, y - UI_ROW_H + 6),
                          (x_start + card_w - 3, y + 6),
                          (245, 248, 245), -1)

        draw_text(panel, label, (row_x_label, y),
                  UI_LABEL_SCALE, UI_COLOR['label'], 1)
        draw_text(panel, value, (row_x_value, y),
                  UI_VALUE_SCALE, border_color, 1)


# ==============================================================================
# BOTTOM PANEL — assembles all drone cards side by side
# ==============================================================================

def draw_bottom_panel(confirmed_tracks, panel_w, panel_h):
    """
    Build and return the full bottom panel image containing all drone cards.

    confirmed_tracks : list of confirmed Track objects (may be empty)
    panel_w          : total width (should equal CAM_W * 2)
    panel_h          : total height (UI_PANEL_H)

    Cards are laid out left to right. Slots beyond len(confirmed_tracks)
    up to UI_MAX_CARDS are drawn as empty.
    To support more drones, change UI_MAX_CARDS at the top of the UI section.
    """
    panel    = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = UI_COLOR['panel']

    card_w = panel_w // UI_MAX_CARDS

    for slot in range(UI_MAX_CARDS):
        x_start = slot * card_w
        track   = confirmed_tracks[slot] if slot < len(confirmed_tracks) else None
        draw_drone_card(panel, x_start, card_w, track)

    return panel


# ==============================================================================
# FULL CANVAS COMPOSITOR
# Assembles the final display from all sub-panels.
# To change the overall layout, edit only this function.
# ==============================================================================

def build_canvas(frame_L, frame_R, confirmed_tracks,
                 fps, n_stereo, n_single):
    """
    Assemble the complete display canvas from all components.

    Layout (top to bottom):
      1. Camera feeds     — frame_L and frame_R side by side
      2. Info bar         — global system stats
      3. Bottom panel     — one card per drone slot

    To change the layout order, reorder the np.vstack() call at the bottom.
    To add a new panel, create a draw function and add it to the vstack.

    Returns the final BGR canvas ready for cv2.imshow().
    """
    total_w = frame_L.shape[1] + frame_R.shape[1]

    # Row 1 — camera feeds
    cam_row = np.hstack([frame_L, frame_R])

    # Row 2 — info bar
    bar = np.zeros((UI_BAR_H, total_w, 3), dtype=np.uint8)
    draw_info_bar(bar, fps, len(confirmed_tracks), n_stereo, n_single)

    # Row 3 — drone cards
    bottom = draw_bottom_panel(confirmed_tracks, total_w, UI_PANEL_H)

    return np.vstack([cam_row, bar, bottom])



def run():
    print("[*] Loading YOLO model ...")
    model = YOLO(MODEL_PATH, task='detect')

    print("[*] Warming up TensorRT engine ...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(3):
        model([dummy], conf=CONF_THRESH, verbose=False)
        print(f"[*] Warmup {i+1}/3 done")
    torch.cuda.empty_cache()
    print("[*] Warmup complete.")

    print("[*] Loading calibration ...")
    try:
        calib = StereoCalibration(CALIB_NPZ)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    calib_W, calib_H = calib.image_size

    scale_x = calib_W / CAM_W
    scale_y = calib_H / CAM_H

    # Point-level rectification only. Display frames stay raw (no
    # visible zoom / warp), but each detection center will be rectified
    # in get_all_drones() so the stereo matcher's epipolar gate and
    # triangulate() both operate on calibration-correct coordinates.
    calib.use_point_rectification()

    # Shared coordinate grid — established from calibration
    grid    = SharedGrid(calib, CAM_W, CAM_H)

    # Track manager — owns all drone tracks
    manager = TrackManager(calib, grid, scale_x, scale_y)

    # CSV log
    log_file   = open(LOG_PATH, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'timestamp', 'track_id', 'state',
        'cam_id', 'conf',
        'grid_x', 'grid_y',
        'x', 'y', 'z',
        'distance', 'speed'
    ])
    print(f"[*] Logging to {LOG_PATH}")

    print("[*] Opening cameras at display resolution ...")
    cam_L = CameraGrabber(0, CAM_W, CAM_H, FPS, FLIP_METHOD, "cam_L")
    cam_R = CameraGrabber(1, CAM_W, CAM_H, FPS, FLIP_METHOD, "cam_R")
    time.sleep(1.5)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    fps_buf    = collections.deque(maxlen=30)
    fps_disp   = 30.0
    fps_last_t = time.time()

    # Sticky-pair hysteresis state — list of
    # ((cxL, cyL), (cxR, cyR)) box-center pairs formed on the previous
    # frame, consumed by match_stereo_detections to give frame-to-frame
    # pairing stability. Empty on the first iteration.
    prev_stereo_pairs = []

    print("[*] Running -- Q to quit.\n")
    cv2.namedWindow("Drone Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Tracker", CAM_W * 2, CAM_H + UI_BAR_H + UI_PANEL_H)

    try:
        while True:
            t_start = time.time()

            # ── Read both frames simultaneously ───────────────────────
            fL      = executor.submit(cam_L.read)
            fR      = executor.submit(cam_R.read)
            frame_L = fL.result()
            frame_R = fR.result()

            if frame_L is None or frame_R is None:
                time.sleep(0.005)
                continue

            # ── Batched YOLO inference ────────────────────────────────
            # Frames are NOT remapped — we keep the raw image for both
            # detection (YOLO accuracy is best on the image the network
            # was trained on) and display (no zoom / warp artefacts).
            # Geometric correction happens per-detection inside
            # get_all_drones() via calib.rectify_points().
            results  = model([frame_L, frame_R],
                             conf=CONF_THRESH, verbose=False)
            res_L, res_R = results[0], results[1]

            # ── Step 1: get all detections from each camera ───────────
            dets_L = get_all_drones(res_L, DRONE_CLASS, CONF_THRESH,
                                    grid, cam_id=0, cam_w=CAM_W, cam_h=CAM_H,
                                    calib=calib, scale_x=scale_x, scale_y=scale_y)
            dets_R = get_all_drones(res_R, DRONE_CLASS, CONF_THRESH,
                                    grid, cam_id=1, cam_w=CAM_W, cam_h=CAM_H,
                                    calib=calib, scale_x=scale_x, scale_y=scale_y)

            # ── Step 2: match detections across cameras ───────────────
            # prev_stereo_pairs gives the matcher hysteresis so that a
            # pair which solved cleanly last frame does not flicker
            # across the epipolar boundary this frame for marginal
            # geometric cases (e.g. slight YOLO jitter on a silhouette
            # against a bright sky).
            stereo_pairs, cam0_only, cam1_only = match_stereo_detections(
                dets_L, dets_R, grid,
                prev_pairs=prev_stereo_pairs)

            # Rebuild the sticky-pair state from the pairs we actually
            # kept this frame. Using display-pixel centers keeps the
            # tolerance in match_stereo_detections dimensionally
            # consistent with drone motion.
            prev_stereo_pairs = [
                ((dL['px'], dL['py']), (dR['px'], dR['py']))
                for dL, dR in stereo_pairs
            ]

            # ── Step 3: update track manager ──────────────────────────
            confirmed_tracks = manager.update(stereo_pairs, cam0_only, cam1_only)

            # ── Annotate camera frames ────────────────────────────────
            annotate_frames(frame_L, frame_R, confirmed_tracks)

            # ── Build canvas ──────────────────────────────────────────
            canvas = build_canvas(
                frame_L, frame_R, confirmed_tracks,
                fps_disp, len(stereo_pairs),
                len(cam0_only) + len(cam1_only)
            )

            # ── Terminal output ───────────────────────────────────────
            print(f"[fps={fps_disp:.1f}]  "
                  f"stereo={len(stereo_pairs)}  "
                  f"cam0_only={len(cam0_only)}  "
                  f"cam1_only={len(cam1_only)}  "
                  f"tracks={len(confirmed_tracks)}")

            for t in confirmed_tracks:
                pos_str = (f"X={t.pos[0]:+.2f} Y={t.pos[1]:+.2f} "
                           f"Z={t.pos[2]:.2f}m  spd={t.speed:.2f}m/s"
                           if t.has_3d else "no 3D")
                print(f"  ID={t.id}  state={t.state}  "
                      f"grid=({t.grid_x:.3f},{t.grid_y:.3f})  {pos_str}")

            # ── CSV logging — one row per confirmed track per frame ───
            # Use t_start as the timestamp so every row reflects when the
            # frame entered the pipeline, not when the CSV write happened.
            ts = round(t_start, 4)
            for t in confirmed_tracks:
                log_writer.writerow([
                    ts,
                    t.id,
                    t.state,
                    t.cam_id,
                    round(t.conf, 4),
                    round(t.grid_x, 4),
                    round(t.grid_y, 4),
                    round(float(t.pos[0]), 4) if t.has_3d else '',
                    round(float(t.pos[1]), 4) if t.has_3d else '',
                    round(float(t.pos[2]), 4) if t.has_3d else '',
                    round(float(np.linalg.norm(t.pos)), 4) if t.has_3d else '',
                    round(t.speed, 4),
                ])
            # Flush per frame so a hard kill (e.g. power loss on the
            # Jetson) doesn't lose the tail of the log.
            log_file.flush()

            # ── Show canvas ───────────────────────────────────────────
            # Snapshot pipeline cost before display so the FPS calculation
            # excludes cv2.imshow / waitKey driver timing, matching the
            # original behaviour.  The cap sleep extends this to the full
            # frame budget when FPS_CAP > 0.
            t_pipeline_end = time.time()

            cv2.imshow("Drone Tracker", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # ── FPS cap — yield spare time back to the OS ─────────────
            if FPS_CAP > 0:
                spare = (1.0 / FPS_CAP) - (time.time() - t_start)
                if spare > 0:
                    time.sleep(spare)

            # ── FPS averaging ─────────────────────────────────────────
            # When uncapped: use t_pipeline_end so display overhead is
            # excluded (same as before).  When capped: use the post-sleep
            # time so the displayed FPS reflects the actual capped rate.
            t_measure = time.time() if FPS_CAP > 0 else t_pipeline_end
            frame_ms = (t_measure - t_start) * 1000
            fps_now = 1000.0 / max(frame_ms, 1e-1)
            fps_buf.append(fps_now)
            if time.time() - fps_last_t >= 1.0:
                fps_disp   = sum(fps_buf) / len(fps_buf)
                fps_last_t = time.time()

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C -- stopping.")
    finally:
        cam_L.release()
        cam_R.release()
        executor.shutdown(wait=False)
        log_file.close()
        print(f"[*] Log saved to {LOG_PATH}")
        cv2.destroyAllWindows()
        print("[*] Stopped.")

if __name__ == '__main__':
    run()
