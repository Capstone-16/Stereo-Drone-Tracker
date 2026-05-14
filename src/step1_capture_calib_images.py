"""
step1_capture_calib_images.py
==============================
Captures stereo image pairs for calibration.

No display required — preview streams to your browser.
Open  http://<jetson-ip>:8081  on any device on the same network.

Controls (type in THIS terminal, then press Enter):
  s   — save a pair
  d   — delete the last saved pair
  q   — quit

Saved to:
  /workspace/calib/left/frame_00.png ...
  /workspace/calib/right/frame_00.png ...
"""

import os
import time
import threading

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAVE_DIR   = '/workspace/calib'
WIDTH      = 1920
HEIGHT     = 1080
FPS        = 30
FLIP       = 0
MIN_PAIRS  = 20
STREAM_PORT = 8081
PREV_W     = 960
PREV_H     = 540
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(f'{SAVE_DIR}/left',  exist_ok=True)
os.makedirs(f'{SAVE_DIR}/right', exist_ok=True)


# ── MJPEG stream ──────────────────────────────────────────────────────────────

class MJPEGStream:
    def __init__(self, port):
        self._frame_bytes = b''
        self._lock        = threading.Lock()
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == '/':
                    html = (
                        b'<html><body style="background:#111;margin:0">'
                        b'<img src="/stream" style="width:100%;display:block">'
                        b'</body></html>'
                    )
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', len(html))
                    self.end_headers()
                    self.wfile.write(html)
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type',
                                     'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    try:
                        while True:
                            with stream._lock:
                                data = stream._frame_bytes
                            if data:
                                self.wfile.write(
                                    b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n'
                                    + data + b'\r\n'
                                )
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        server = HTTPServer(('0.0.0.0', port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[Stream] Preview at http://<jetson-ip>:{port}")

    def push(self, frame):
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._frame_bytes = buf.tobytes()


# ── Camera grabber ────────────────────────────────────────────────────────────

def make_pipeline(sensor_id):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, "
        f"framerate={FPS}/1 ! "
        f"nvvidconv flip-method={FLIP} ! "
        f"video/x-raw, width={WIDTH}, height={HEIGHT}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink name=sink emit-signals=True max-buffers=1 drop=True"
    )


def grab(sink):
    sample = sink.emit('pull-sample')
    if sample is None:
        return None
    buf  = sample.get_buffer()
    caps = sample.get_caps()
    h    = caps.get_structure(0).get_value('height')
    w    = caps.get_structure(0).get_value('width')
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    frame = np.ndarray(shape=(h, w, 3), dtype=np.uint8,
                       buffer=bytes(mapinfo.data)).copy()
    buf.unmap(mapinfo)
    return frame


# ── Open cameras ──────────────────────────────────────────────────────────────
print("Opening cameras...")
pipe_L = Gst.parse_launch(make_pipeline(0))
pipe_R = Gst.parse_launch(make_pipeline(1))
sink_L = pipe_L.get_by_name('sink')
sink_R = pipe_R.get_by_name('sink')
pipe_L.set_state(Gst.State.PLAYING)
pipe_R.set_state(Gst.State.PLAYING)
time.sleep(1.5)

stream = MJPEGStream(STREAM_PORT)
count  = 0

print(f"\nCapture at least {MIN_PAIRS} pairs.")
print("Tips for good calibration:")
print("  - Hold the board STILL before saving — motion blur ruins corners")
print("  - Cover all regions: corners, edges, centre, close, far")
print("  - Tilt the board at various angles (+-30 deg)")
print("  - Board must be FULLY visible in BOTH cameras simultaneously")
print("\nControls (type then press Enter):")
print("  s = save pair")
print("  d = delete last pair")
print("  q = quit\n")

# ── Frame push thread ─────────────────────────────────────────────────────────
latest = {'L': None, 'R': None}
lock   = threading.Lock()
stop   = threading.Event()

def grab_loop():
    while not stop.is_set():
        fL = grab(sink_L)
        fR = grab(sink_R)
        if fL is not None and fR is not None:
            with lock:
                latest['L'] = fL
                latest['R'] = fR

grabber = threading.Thread(target=grab_loop, daemon=True)
grabber.start()

def push_loop():
    while not stop.is_set():
        with lock:
            fL = latest['L']
            fR = latest['R']
        if fL is None or fR is None:
            time.sleep(0.05)
            continue
        vis_L = cv2.resize(fL, (PREV_W, PREV_H))
        vis_R = cv2.resize(fR, (PREV_W, PREV_H))
        preview = np.hstack([vis_L, vis_R])
        # Overlay status
        colour = (0, 255, 0) if count >= MIN_PAIRS else (0, 165, 255)
        cv2.putText(preview, f"Pairs saved: {count}  (need >= {MIN_PAIRS})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(preview,
                    "terminal: s=save  d=delete last  q=quit",
                    (10, PREV_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)
        stream.push(preview)
        time.sleep(0.04)

pusher = threading.Thread(target=push_loop, daemon=True)
pusher.start()

# ── Main input loop ───────────────────────────────────────────────────────────
try:
    while True:
        cmd = input("cmd> ").strip().lower()

        if cmd == 'q':
            break

        elif cmd == 's':
            with lock:
                fL = latest['L']
                fR = latest['R']
            if fL is None or fR is None:
                print("  [WARN] No frame yet — try again")
                continue
            pl = f'{SAVE_DIR}/left/frame_{count:02d}.png'
            pr = f'{SAVE_DIR}/right/frame_{count:02d}.png'
            cv2.imwrite(pl, fL)
            cv2.imwrite(pr, fR)
            print(f"  [OK] Saved pair {count:02d}  ({count+1}/{MIN_PAIRS})")
            count += 1

        elif cmd == 'd':
            if count == 0:
                print("  [WARN] Nothing to delete")
                continue
            count -= 1
            pl = f'{SAVE_DIR}/left/frame_{count:02d}.png'
            pr = f'{SAVE_DIR}/right/frame_{count:02d}.png'
            for p in (pl, pr):
                if os.path.isfile(p):
                    os.remove(p)
            print(f"  [OK] Deleted pair {count:02d}")

        else:
            print("  Unknown command. Use: s / d / q")

except KeyboardInterrupt:
    pass

finally:
    stop.set()
    pipe_L.set_state(Gst.State.NULL)
    pipe_R.set_state(Gst.State.NULL)

print(f"\nDone. {count} pairs saved to {SAVE_DIR}/left and {SAVE_DIR}/right")
if count < MIN_PAIRS:
    print(f"[WARNING] Only {count} pairs — recommend at least {MIN_PAIRS}.")
else:
    print("Next step: run  step2_calibrate_stereo.py")
