"""Shared building blocks for ISL recognition: camera, hand tracking, features, classifier, speech.

Pipeline:  camera frame -> MediaPipe hand landmarks (2 hands x 21 points)
           -> per-frame features (position/scale invariant hand shape + wrist trajectory)
           -> sliding window of WINDOW frames (+ velocities)
           -> tiny dilated 1D-CNN (pure numpy at runtime) -> class probabilities
"""
import os
import queue
import threading
import time
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "isl_classifier.npz")
LANDMARKER_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")

WINDOW = 20             # frames the classifier sees (~0.7 s at 30 fps)
MIN_PRESENT = 0.6       # fraction of a window that must contain a hand before we classify it
NONE_LABEL = "_none"    # "no sign / transition / resting" class
HAND_DIM = 64           # 60 wrist-relative shape + wrist xy + palm size + present flag
FRAME_DIM = 2 * HAND_DIM + 2  # two hand slots + vector between the wrists

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# --------------------------------------------------------------------------- camera
class _RpicamPipe:
    """Raspberry Pi ribbon camera read from the `rpicam-vid` program as raw YUV420 on a pipe.
    Works from ANY Python version (picamera2 only exists for the system Python), e.g. the
    Python 3.12 venv that MediaPipe 0.10.x needs on a Pi 4 running Trixie (Python 3.13)."""

    def __init__(self, size):
        import shutil
        import subprocess
        exe = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not exe:
            raise RuntimeError("Neither picamera2 nor rpicam-vid found: is this a Raspberry Pi?")
        # widths that are a multiple of 64 have no row padding, so a frame is exactly w*h*1.5 bytes
        self._w, self._h = (size[0] // 64) * 64, (size[1] // 2) * 2
        self._n = self._w * self._h * 3 // 2
        self._proc = subprocess.Popen(
            [exe, "-t", "0", "-n", "--codec", "yuv420", "--width", str(self._w),
             "--height", str(self._h), "--framerate", "15", "-o", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self._buf = bytearray(self._n)

    def isOpened(self):
        return self._proc.poll() is None

    def read(self):
        view, got = memoryview(self._buf), 0
        while got < self._n:
            k = self._proc.stdout.readinto(view[got:])
            if not k:
                return False, None
            got += k
        yuv = np.frombuffer(self._buf, np.uint8).reshape(self._h * 3 // 2, self._w)
        return True, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def set(self, *_):
        return False

    def release(self):
        self._proc.terminate()
        self._proc.wait(timeout=3)


def _open_picam(size):
    try:
        import picamera2  # noqa: F401
    except ImportError:
        return _RpicamPipe(size)
    return _PiCamera(size)


class _PiCamera:
    """Raspberry Pi ribbon camera (CSI, e.g. Camera v1.3 / v2 / v3) via Picamera2, wrapped to
    look like cv2.VideoCapture. OpenCV can't open these cameras on Raspberry Pi OS Bookworm."""

    def __init__(self, size):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        # "RGB888" in libcamera naming = B,G,R byte order = exactly what OpenCV expects
        cfg = self._cam.create_video_configuration(main={"size": size, "format": "RGB888"},
                                                   controls={"FrameRate": 15}, buffer_count=2)
        self._cam.configure(cfg)
        self._cam.start()

    def isOpened(self):
        return True

    def read(self):
        return True, self._cam.capture_array()

    def set(self, *_):
        return False

    def release(self):
        self._cam.stop()
        self._cam.close()


class Camera:
    """Reads frames on a background thread so the main loop always gets the newest frame
    (no driver buffer lag). `source` is a webcam index, "picam" for the Raspberry Pi ribbon
    camera, or a video file / network stream URL."""

    def __init__(self, source="0", width=640, height=480):
        self._src = int(source) if str(source).isdigit() else source
        self._size = (width, height)
        self._cond = threading.Condition()
        self._frame, self._seq = None, 0
        self.running = True
        self._cap = self._open()
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source {source!r}")
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        if self._src == "picam":
            return _open_picam(self._size)
        if isinstance(self._src, int):
            cap = cv2.VideoCapture(self._src, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self._src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        fails = 0
        while self.running:
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                if fails > 50 and not isinstance(self._src, int) and self._src != "picam":  # stream dropped: reconnect
                    self._cap.release()
                    self._cap, fails = self._open(), 0
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq=0):
        """Blocks until a frame newer than `last_seq` arrives. Returns (seq, frame or None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self.running, timeout=2.0)
            return self._seq, (self._frame if self._seq != last_seq else None)

    def release(self):
        self.running = False
        time.sleep(0.05)
        self._cap.release()


# --------------------------------------------------------------------------- hands
class HandTracker:
    """MediaPipe HandLandmarker. VIDEO mode tracks hands between frames instead of re-detecting
    every frame (faster, steadier); video=False treats every call as an unrelated photo."""

    def __init__(self, num_hands=2, video=True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not os.path.exists(LANDMARKER_PATH):
            os.makedirs(MODEL_DIR, exist_ok=True)
            print("Downloading MediaPipe hand landmark model ...")
            urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH + ".part")
            os.replace(LANDMARKER_PATH + ".part", LANDMARKER_PATH)
        with open(LANDMARKER_PATH, "rb") as f:
            model = f.read()
        self._mp = mp
        self._video = video
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self._t0 = time.monotonic()
        self._last_ts = -1

    def __call__(self, frame_bgr):
        """Returns (hands[2,21,3] in normalised image coords, present[2]).
        Slot 0 = left hand, slot 1 = right hand (frame must already be mirrored)."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self._video:
            ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
            self._last_ts = ts
            res = self._landmarker.detect_for_video(img, ts)
        else:
            res = self._landmarker.detect(img)
        hands = np.zeros((2, 21, 3), np.float32)
        present = np.zeros(2, bool)
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            slot = 0 if handed[0].category_name == "Left" else 1
            if present[slot]:  # both hands got the same label: use the free slot
                slot = 1 - slot
            hands[slot] = [(p.x, p.y, p.z) for p in lms]
            present[slot] = True
        return hands, present

    def close(self):
        self._landmarker.close()


def _hand_worker(qin, qout, num_hands):
    """Child process: its own MediaPipe tracker, fed every N-th frame in time order."""
    tracker = HandTracker(num_hands=num_hands)
    while True:
        item = qin.get()
        if item is None:
            break
        qout.put(tracker(item))
    tracker.close()


class ParallelHandTracker:
    """Runs `workers` MediaPipe trackers in separate processes and hands them frames in turn.

    On a Raspberry Pi 4 one tracker manages ~5.5 fps with hands in view and uses about two of
    the four cores; two in parallel reach ~10 fps (measured). Each result comes back about one
    frame later than with a single tracker, which the doubled frame rate more than repays.

    step(frame, t) queues a frame and returns the OLDEST finished one as (t, frame, hands,
    present), in capture order, or None while the pipeline is still filling."""

    def __init__(self, workers=2, num_hands=2):
        import collections
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # fresh interpreter per worker: MediaPipe is not fork-safe
        self.n = max(1, workers)
        self._qin = [ctx.Queue(maxsize=2) for _ in range(self.n)]
        self._qout = [ctx.Queue() for _ in range(self.n)]
        self._procs = [ctx.Process(target=_hand_worker, args=(self._qin[i], self._qout[i], num_hands),
                                   daemon=True) for i in range(self.n)]
        for p in self._procs:
            p.start()
        self._pending = collections.deque()
        self._next = 0

    def step(self, frame, t):
        w = self._next % self.n
        self._next += 1
        self._qin[w].put(frame)
        self._pending.append((w, t, frame))
        if len(self._pending) < self.n:
            return None
        w0, t0, f0 = self._pending.popleft()
        hands, present = self._qout[w0].get(timeout=30)  # first call waits for model loading
        return t0, f0, hands, present

    def close(self):
        for q in self._qin:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()


class SerialHandTracker:
    """Same step() interface with a single in-process tracker (fast PCs)."""

    def __init__(self, num_hands=2):
        self._t = HandTracker(num_hands=num_hands)

    def step(self, frame, t):
        hands, present = self._t(frame)
        return t, frame, hands, present

    def close(self):
        self._t.close()


# --------------------------------------------------------------------------- features
def frame_features(hands, present, aspect):
    """hands (..., 2, 21, 3) normalised image coords, present (..., 2) -> (..., FRAME_DIM).

    Hand shape is wrist-relative and divided by palm size, so it does not depend on where
    the hand is or how far it is from the camera. Absent hands are all zeros."""
    h = np.array(hands, np.float32)
    h[..., 0] *= aspect                                   # x and y in the same unit
    p = np.asarray(present, np.float32)[..., None]        # (..., 2, 1)
    scale = np.maximum(np.linalg.norm(h[..., 9, :2] - h[..., 0, :2], axis=-1), 1e-3)[..., None]
    local = ((h[..., 1:, :] - h[..., :1, :]) / scale[..., None]).reshape(*h.shape[:-2], 60)
    per_hand = np.concatenate([local, h[..., 0, :2], scale, np.ones_like(scale)], -1) * p
    both = p[..., 0, :] * p[..., 1, :]
    inter = (h[..., 1, 0, :2] - h[..., 0, 0, :2]) / (0.5 * (scale[..., 0, :] + scale[..., 1, :])) * both
    return np.concatenate([per_hand.reshape(*h.shape[:-3], 2 * HAND_DIM), inter], -1)


def hand_presence(frames):
    """(..., T, FRAME_DIM) -> (..., T) bool: any hand visible in the frame."""
    return (frames[..., HAND_DIM - 1] + frames[..., 2 * HAND_DIM - 1]) > 0


def window_ready(frames):
    """Same gate in training and at runtime: enough hand frames, and a hand in the newest one."""
    pres = hand_presence(frames)
    return (pres.mean(-1) >= MIN_PRESENT) & pres[..., -1]


TRAIN_FPS = 30.0  # frame rate the training clips were recorded at


class TimedWindow:
    """Recent per-frame features, resampled to T steps spaced 1/TRAIN_FPS apart.

    The classifier was trained on 20 frames at ~30 fps (~0.63 s of motion). A slower device
    (Raspberry Pi 4: ~8 fps) would otherwise feed it ~2.5 s of motion and need far more frames
    before it can answer. Resampling by time keeps what the model sees, and the delay, the same
    at any frame rate; at 30 fps it is essentially the plain sliding window."""

    _PRESENT = (HAND_DIM - 1, 2 * HAND_DIM - 1)  # "hand present" flag of each hand slot

    def __init__(self, T, fps=TRAIN_FPS):
        self.T, self.dt = T, 1.0 / fps
        self.span = (T - 1) * self.dt
        self._t, self._f = [], []

    def clear(self):
        self._t, self._f = [], []

    def append(self, t, feat):
        self._t.append(t)
        self._f.append(np.asarray(feat, np.float32))
        # keep one frame from before the window start, to interpolate the first step
        while len(self._t) >= 2 and self._t[1] <= t - self.span:
            del self._t[0], self._f[0]

    def __len__(self):
        return len(self._t)

    def window(self):
        """(T, FRAME_DIM). Steps before the first buffered frame are 'no hand' (zeros), exactly
        like the left padding of a short buffer before."""
        ts, F = np.array(self._t), np.stack(self._f)
        targets = ts[-1] - self.span + self.dt * np.arange(self.T)
        out = np.zeros((self.T, F.shape[1]), np.float32)
        idx = np.searchsorted(ts, targets + 1e-6, side="right") - 1  # frame at or before each step
        for j, i in enumerate(idx):
            if i < 0:
                continue
            if i >= len(ts) - 1:
                out[j] = F[-1]
                continue
            a, b = F[i], F[i + 1]
            alpha = (targets[j] - ts[i]) / max(ts[i + 1] - ts[i], 1e-6)
            v = a + alpha * (b - a)
            near = a if alpha < 0.5 else b
            pa = [a[k] > 0 for k in self._PRESENT]
            pb = [b[k] > 0 for k in self._PRESENT]
            for s in (0, 1):  # a hand appears/disappears: don't blend it with "no hand"
                if pa[s] != pb[s]:
                    v[s * HAND_DIM:(s + 1) * HAND_DIM] = near[s * HAND_DIM:(s + 1) * HAND_DIM]
            if (pa[0] and pa[1]) != (pb[0] and pb[1]):
                v[2 * HAND_DIM:] = near[2 * HAND_DIM:]
            out[j] = v
        return out


def resample_clip(ts, hands, present, n_frames, fps=TRAIN_FPS):
    """Recorded (times, hands (N,2,21,3), present (N,2)) at any frame rate -> the last
    `n_frames` frames at `fps`, as the training data expects. Landmarks of a hand seen in both
    neighbouring frames are interpolated; a hand appearing/disappearing takes the nearer frame."""
    ts, hands, present = np.asarray(ts, float), np.asarray(hands, np.float32), np.asarray(present, bool)
    targets = ts[-1] - (n_frames - 1) / fps + np.arange(n_frames) / fps
    out_h = np.zeros((n_frames, 2, 21, 3), np.float32)
    out_p = np.zeros((n_frames, 2), bool)
    idx = np.searchsorted(ts, targets + 1e-6, side="right") - 1
    for j, i in enumerate(idx):
        if i < 0:
            i, a = 0, 0.0  # before the first frame: hold the first frame
        elif i >= len(ts) - 1:
            i, a = len(ts) - 1, 0.0
        else:
            a = (targets[j] - ts[i]) / max(ts[i + 1] - ts[i], 1e-6)
        k = min(i + 1, len(ts) - 1)
        for s in (0, 1):
            if present[i, s] and present[k, s]:
                out_h[j, s] = hands[i, s] + a * (hands[k, s] - hands[i, s])
                out_p[j, s] = True
            else:
                near = i if a < 0.5 else k
                out_h[j, s], out_p[j, s] = hands[near, s], present[near, s]
    return out_h, out_p


def window_features(w):
    """(..., T, FRAME_DIM) -> (..., T, 2*FRAME_DIM).

    Wrist positions become a trajectory relative to the window's mean position, in palm-size
    units (so movement is distance/position invariant); then frame-to-frame velocities are
    appended (masked where a hand appears/disappears)."""
    w = np.array(w, np.float32)
    for o in (0, HAND_DIM):
        p = w[..., o + 63:o + 64]
        n = np.maximum(p.sum(-2, keepdims=True), 1.0)
        ref = np.maximum(w[..., o + 62:o + 63].sum(-2, keepdims=True) / n, 1e-3)
        center = w[..., o + 60:o + 62].sum(-2, keepdims=True) / n
        w[..., o + 60:o + 62] = (w[..., o + 60:o + 62] - center) / ref * p
        w[..., o + 62:o + 63] = (w[..., o + 62:o + 63] / ref - 1.0) * p
    p0, p1 = w[..., 63:64], w[..., 127:128]
    mask = np.concatenate([np.repeat(p0, HAND_DIM, -1), np.repeat(p1, HAND_DIM, -1),
                           p0 * p1, p0 * p1], -1)
    vel = np.zeros_like(w)
    vel[..., 1:, :] = (w[..., 1:, :] - w[..., :-1, :]) * mask[..., 1:, :] * mask[..., :-1, :]
    return np.concatenate([w, vel], -1)


# --------------------------------------------------------------------------- classifier
def conv1d(x, w, b, dil):
    """'same'-padded dilated conv over time. x (..., T, Cin), w (K, Cin, Cout)."""
    k, t = w.shape[0], x.shape[-2]
    pad = dil * (k // 2)
    xp = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(pad, pad), (0, 0)])
    return b + sum(xp[..., i * dil:i * dil + t, :] @ w[i] for i in range(k))


class SignClassifier:
    """Numpy inference of the trained CNN (BatchNorm folded into the convs). ~0.2 ms/window,
    no torch needed at runtime, so it runs anywhere NumPy runs."""

    def __init__(self, path=CLASSIFIER_PATH):
        d = np.load(path)
        self.labels = [str(s) for s in d["labels"]]
        self.window = int(d["window"])
        self.mu, self.sd = d["mu"], d["sd"]
        self.convs = [(d[f"w{i}"], d[f"b{i}"], int(d[f"dil{i}"])) for i in range(int(d["n_conv"]))]
        self.fc_w, self.fc_b = d["fc_w"], d["fc_b"]

    def __call__(self, frames):
        """frames (T, FRAME_DIM) or (N, T, FRAME_DIM) -> class probabilities."""
        x = (window_features(frames) - self.mu) / self.sd
        for w, b, dil in self.convs:
            x = np.maximum(conv1d(x, w, b, dil), 0.0)
        z = np.concatenate([x.mean(-2), x.max(-2)], -1) @ self.fc_w + self.fc_b
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)


# --------------------------------------------------------------------------- speech
class Speaker:
    """Non-blocking text-to-speech. Utterances queued while speaking are merged into one,
    so speech never lags behind fast continuous signing."""

    def __init__(self, enabled=True, rate=0):
        self.enabled = enabled
        self._rate = rate
        self._q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text):
        if self.enabled and text:
            self._q.put(text)

    def _engine(self):
        if os.name == "nt":
            try:  # Windows SAPI directly: lowest latency
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Rate = self._rate
                return lambda text: voice.Speak(text)
            except Exception:
                pass
        else:
            import shutil
            import subprocess
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            if exe:  # Linux / Raspberry Pi: call espeak-ng directly (pyttsx3 is flaky there)
                wpm = str(175 + 15 * self._rate)
                # -a 200 = maximum loudness (default 100): a speaker on the Pi's jack has no amplifier
                return lambda text: subprocess.run([exe, "-a", "200", "-s", wpm, text], check=True,
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print("TTS: espeak-ng not found (sudo apt install espeak-ng), trying pyttsx3")
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175 + 15 * self._rate)

        def speak(text):
            engine.say(text)
            engine.runAndWait()
        return speak

    def _loop(self):
        try:
            speak = self._engine()
        except Exception as e:  # don't die silently: say why there is no voice
            print("TTS disabled, could not start speech engine:", e)
            return
        print("TTS ready")
        while True:
            parts = [self._q.get()]
            while not self._q.empty():
                parts.append(self._q.get_nowait())
            try:
                speak(" ".join(parts))
            except Exception as e:
                print("TTS error:", e)


# --------------------------------------------------------------------------- drawing
def pretty(label):
    return label.replace("_", " ").strip()


def draw_hands(img, hands, present):
    h, w = img.shape[:2]
    for s, color in ((0, (255, 170, 0)), (1, (0, 200, 255))):
        if not present[s]:
            continue
        pts = (hands[s, :, :2] * (w, h)).astype(int).tolist()
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
