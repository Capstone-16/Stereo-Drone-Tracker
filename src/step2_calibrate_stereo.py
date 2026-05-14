"""
step2_calibrate_stereo.py
==========================
Runs stereo calibration from the images captured by step1.

Input:  /workspace/calib/left/*.png
        /workspace/calib/right/*.png
Output: /workspace/stereo_calib.npz  (overwrites previous calibration)

Run inside Docker container:
  python3 /workspace/step2_calibrate_stereo.py
"""

import sys
import os
import glob
import cv2
import numpy as np

# ── CONFIG — match your physical checkerboard ─────────────────────────────────
CHECKERBOARD = (9, 6)    # inner corners: (cols-1, rows-1)
                          # e.g. a 10x7 square board has 9x6 inner corners
SQUARE_SIZE  = 0.020     # side length of one square in METRES (0.020 = 2 cm)
# ─────────────────────────────────────────────────────────────────────────────

HERE        = os.path.dirname(os.path.abspath(__file__))
INPUT_LEFT  = os.path.join(HERE, 'calib', 'left',  '*.png')
INPUT_RIGHT = os.path.join(HERE, 'calib', 'right', '*.png')
OUTPUT_FILE = os.path.join(HERE, 'stereo_calib.npz')

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# ── 3-D object points for one board ──────────────────────────────────────────
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
)

# ── Load image paths ──────────────────────────────────────────────────────────
images_L = sorted(glob.glob(INPUT_LEFT))
images_R = sorted(glob.glob(INPUT_RIGHT))

print(f"Looking for images in:")
print(f"  L: {INPUT_LEFT}")
print(f"  R: {INPUT_RIGHT}\n")

if len(images_L) == 0 or len(images_R) == 0:
    print(f"[ERROR] No images found.")
    print(f"  Left  found : {len(images_L)}")
    print(f"  Right found : {len(images_R)}")
    print("  -> Run step1_capture_calib_images.py first.")
    sys.exit(1)

if len(images_L) != len(images_R):
    print(f"[ERROR] Mismatch: {len(images_L)} left vs {len(images_R)} right.")
    sys.exit(1)

print(f"Found {len(images_L)} image pairs.")
print(f"Checkerboard: {CHECKERBOARD}  square size: {SQUARE_SIZE*100:.1f} cm\n")

# ── Detect corners ────────────────────────────────────────────────────────────
obj_points, img_points_L, img_points_R = [], [], []
img_size = None
skipped  = 0

for i, (pl, pr) in enumerate(zip(images_L, images_R)):
    imgL = cv2.imread(pl)
    imgR = cv2.imread(pr)
    if imgL is None or imgR is None:
        print(f"  [SKIP] Pair {i:02d} — could not read file")
        skipped += 1
        continue

    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = gL.shape[::-1]   # (width, height)
        print(f"  Image size: {img_size[0]} x {img_size[1]} px")

    retL, cL = cv2.findChessboardCorners(gL, CHECKERBOARD, None)
    retR, cR = cv2.findChessboardCorners(gR, CHECKERBOARD, None)

    if retL and retR:
        cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        obj_points.append(objp)
        img_points_L.append(cL)
        img_points_R.append(cR)
        print(f"  [OK]   Pair {i:02d}")
    else:
        print(f"  [SKIP] Pair {i:02d} — board not found "
              f"(L={'OK' if retL else 'FAIL'}, R={'OK' if retR else 'FAIL'})")
        skipped += 1

print(f"\nUsable pairs: {len(obj_points)}  |  Skipped: {skipped}")

if len(obj_points) == 0:
    print("\n[ERROR] No corners detected. Common causes:")
    print("  1. CHECKERBOARD setting is wrong — count INNER corners only")
    print(f"     Current: {CHECKERBOARD}")
    print("  2. Images are blurry or poorly lit")
    print("  3. Board not fully visible in both frames")
    sys.exit(1)

if len(obj_points) < 10:
    print(f"[ERROR] Only {len(obj_points)} valid pairs — need at least 10.")
    sys.exit(1)

# ── Individual camera calibration ─────────────────────────────────────────────
print("\nCalibrating LEFT camera...")
rms_L, K_L, D_L, _, _ = cv2.calibrateCamera(
    obj_points, img_points_L, img_size, None, None)
print(f"  RMS reprojection error: {rms_L:.4f} px")

print("Calibrating RIGHT camera...")
rms_R, K_R, D_R, _, _ = cv2.calibrateCamera(
    obj_points, img_points_R, img_size, None, None)
print(f"  RMS reprojection error: {rms_R:.4f} px")

if rms_L > 1.0 or rms_R > 1.0:
    print("[WARNING] RMS > 1.0 px — consider recapturing with a flatter board "
          "and better lighting.")

# ── Stereo calibration ────────────────────────────────────────────────────────
print("\nRunning stereo calibration...")
rms_s, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_L, img_points_R,
    K_L, D_L, K_R, D_R, img_size,
    criteria=STEREO_CRITERIA,
    flags=cv2.CALIB_FIX_INTRINSIC
)
baseline_mm = np.linalg.norm(T) * 1000
print(f"  Stereo RMS : {rms_s:.4f} px")
print(f"  Baseline   : {baseline_mm:.1f} mm")

if rms_s > 1.0:
    print("[WARNING] Stereo RMS > 1.0 px — results may be inaccurate.")
else:
    print("[OK] Stereo calibration quality is good.")

# ── Rectification ─────────────────────────────────────────────────────────────
# alpha=1 preserves ALL source pixels (with black borders where
# rectification runs out of data). This keeps the rectified focal
# length P1[0,0] / P2[0,0] close to the raw K_L[0,0] / K_R[0,0],
# which is what the runtime tracker expects for point-level
# rectification.
#
# alpha=0 (the previous setting) zooms in until only valid pixels
# remain. On a rig with any significant relative rotation between
# cameras that produces an absurd rectified focal length — in one
# observed case P1[0,0] was ~20x K_L[0,0], making every rectified
# detection coordinate land outside the usable grid and breaking
# stereo pairing entirely.
#
# CALIB_ZERO_DISPARITY forces the two rectified cameras to share the
# same principal y (and puts principal x at the same row), which is
# the standard stereo configuration and what the tracker assumes.
print("\nComputing rectification maps...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, img_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=1,
)
map1x, map1y = cv2.initUndistortRectifyMap(
    K_L, D_L, R1, P1, img_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(
    K_R, D_R, R2, P2, img_size, cv2.CV_32FC1)

# ── Rectification sanity diagnostics ──────────────────────────────────────────
# Relative rotation between the two physical cameras — if this is large
# (more than a few degrees) it usually signals a mounting problem or
# a bad calibration dataset, not just "tolerable stereo toe-in".
rvec, _ = cv2.Rodrigues(R)
angle_deg = float(np.degrees(np.linalg.norm(rvec)))
print(f"  Relative rotation between cameras: {angle_deg:.2f} deg")
if angle_deg > 10.0:
    print(f"  [WARNING] Inter-camera rotation > 10 deg — check that the "
          f"cameras are mounted level and parallel, or recapture with more "
          f"varied board poses.")

# Ratio of rectified focal to raw focal. For alpha=1 this should be
# within a small factor of 1.0; anything outside ~[0.5, 2.0] means the
# rectification will behave badly in the tracker.
ratio_L = float(P1[0, 0]) / float(K_L[0, 0])
ratio_R = float(P2[0, 0]) / float(K_R[0, 0])
print(f"  Rectified focal ratios: "
      f"P1[0,0]/K_L[0,0]={ratio_L:.2f}, "
      f"P2[0,0]/K_R[0,0]={ratio_R:.2f}")
if not (0.5 <= ratio_L <= 2.0 and 0.5 <= ratio_R <= 2.0):
    print(f"  [WARNING] Rectified focal is far from the raw focal. The "
          f"stereo tracker will refuse to use this calibration for "
          f"point-level rectification and fall back to raw-coordinate "
          f"matching. Recapture calibration images (more variety of board "
          f"angles, sharper focus, less motion blur).")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(
    OUTPUT_FILE,
    K_L=K_L, D_L=D_L, K_R=K_R, D_R=D_R,
    R=R, T=T, E=E, F=F,
    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    map1x=map1x, map1y=map1y,
    map2x=map2x, map2y=map2y,
    img_size=np.array(img_size)
)

print(f"\n[SAVED] -> {OUTPUT_FILE}")
print(f"\nSummary:")
print(f"  Left   fx={K_L[0,0]:.1f}  fy={K_L[1,1]:.1f} px")
print(f"  Right  fx={K_R[0,0]:.1f}  fy={K_R[1,1]:.1f} px")
print(f"  Baseline  : {baseline_mm:.1f} mm")
print(f"  Stereo RMS: {rms_s:.4f} px")
print(f"\nNext step: run  stereo_drone_tracker.py")
