"""Real-time ISL recognition with speech output.

    python run.py                                      # laptop webcam
    python run.py --source picam --headless             # Raspberry Pi camera, no window (autostart mode)
    python run.py --spell                              # letter signs are spelled into a word, spoken at a pause

Keys: q quit · c clear text · v voice on/off
"""
import argparse
import collections
import os
import signal
import time

import cv2
import numpy as np

from isl_core import (CLASSIFIER_PATH, NONE_LABEL, TRAIN_FPS, Camera, ParallelHandTracker,
                      SerialHandTracker, SignClassifier, Speaker, TimedWindow, draw_hands, frame_features,
                      pretty, put_text, window_ready)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, \"picam\" (Raspberry Pi ribbon camera), or stream URL")
    ap.add_argument("--threshold", type=float, default=0.5, help="min smoothed confidence to accept a sign")
    ap.add_argument("--hold", type=int, default=3, help="confident frames needed to accept (counted at 30 fps; "
                                                          "scaled to the real frame rate, min --min-hold)")
    ap.add_argument("--min-hold", type=int, default=1,
                    help="at least this many confident frames (1 = answer on the first confident frame; "
                         "slow devices like the Pi 4 at ~6 fps would otherwise wait ~0.4 s more)")
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA weight of the newest prediction at 30 fps "
                                                            "(1 = none); scaled so the smoothing TIME is the same at any fps")
    ap.add_argument("--reset", type=int, default=6, help="frames without hands that end a sign (at 30 fps; scaled, min 3)")
    ap.add_argument("--pause", type=float, default=0.8, help="seconds without hands that end a spelled word")
    ap.add_argument("--spell", action="store_true", help="join single-letter signs into words")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--rate", type=int, default=1, help="speech rate, -10..10")
    ap.add_argument("--log", action="store_true",
                    help="print fps and the top-3 guesses with confidence for every classified frame")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel hand trackers (default 1; 2 gives ~18%% more fps on a Pi 4 but more delay)")
    ap.add_argument("--headless", action="store_true",
                    help="no window (automatic when there is no display, e.g. started at boot)")
    args = ap.parse_args()
    headless = args.headless or (os.name != "nt" and not os.environ.get("DISPLAY")
                                 and not os.environ.get("WAYLAND_DISPLAY"))
    if headless:  # systemd stops services with SIGTERM: exit cleanly so the camera is released
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    if not os.path.exists(CLASSIFIER_PATH):
        raise SystemExit("No trained model. Record signs with collect.py, then run train.py.")
    clf = SignClassifier()
    labels, T = clf.labels, clf.window
    none_idx = labels.index(NONE_LABEL) if NONE_LABEL in labels else -1

    def per_fps(n, minimum):  # a frame count given at 30 fps -> same duration at the real rate
        return max(minimum, round(n * fps / TRAIN_FPS))

    speaker = Speaker(enabled=not args.no_voice, rate=args.rate)
    cam = Camera(args.source)
    workers = args.workers or 1  # measured on the Pi 4 with live signing: 2 trackers = +18% fps but more delay and stutter
    tracker = ParallelHandTracker(workers) if workers > 1 else SerialHandTracker()
    win = "ISL - recognise"
    print("Ready" + (" (headless)" if headless else ""), flush=True)
    if headless:
        speaker.say("ready")  # no screen: the voice tells the user it has started

    buf = TimedWindow(T)
    ema, cand, cand_n, last, released, idle, absent = None, -1, 0, -1, True, 0, 0
    transcript = collections.deque(maxlen=12)
    spelling = ""
    flash, flash_t = "", 0.0
    seq, fps, t_prev, t_absent = 0, 15.0, time.perf_counter(), time.monotonic()

    def emit(label):
        nonlocal spelling, flash, flash_t
        flash, flash_t = pretty(label), time.monotonic()
        if args.spell and len(label) == 1:
            spelling += label.upper()
            return
        flush()
        transcript.append(pretty(label))
        speaker.say(pretty(label))
        print("sign:", pretty(label), flush=True)

    def flush():
        nonlocal spelling
        if spelling:
            transcript.append(spelling)
            print("word:", spelling, flush=True)
            speaker.say(spelling.lower() if len(spelling) > 1 else spelling)
            spelling = ""

    try:
        while True:
            seq, frame = cam.read(seq)
            if frame is None:
                if not cam.running:
                    break
                continue
            res = tracker.step(cv2.flip(frame, 1), time.monotonic())
            if res is None:  # parallel trackers still filling up
                continue
            t_cap, frame, hands, present = res
            buf.append(t_cap, frame_features(hands, present, frame.shape[1] / frame.shape[0]))
            hold, reset = per_fps(args.hold, args.min_hold), per_fps(args.reset, 3)
            # same smoothing time at any frame rate: k frames at 30 fps == 1 frame at 30/k fps
            alpha = 1 - (1 - args.smooth) ** (TRAIN_FPS / max(fps, 1.0))

            # ---- hands gone: end the current sign (and spelled word after a longer pause)
            if present.any():
                absent = 0
            else:
                absent += 1
                if absent == 1:
                    t_absent = time.monotonic()
                if absent >= reset:
                    buf.clear()
                    ema, cand, cand_n, released = None, -1, 0, True
                if spelling and time.monotonic() - t_absent >= args.pause:
                    flush()

            # ---- classify the last ~0.63 s, resampled to T steps (left part = "no hand" when short)
            if len(buf):
                window = buf.window()
                if window_ready(window):
                    p = clf(window)
                    ema = p if ema is None else alpha * p + (1 - alpha) * ema
                    k = int(ema.argmax())
                    if args.log:
                        top = " | ".join(f"{labels[i]} {ema[i]:.2f}" for i in np.argsort(-ema)[:3])
                        print(f"{time.monotonic():.2f} fps {fps:4.1f}  {top}", flush=True)
                    if k != none_idx and ema[k] >= args.threshold:
                        cand_n = cand_n + 1 if k == cand else 1
                        cand, idle = k, 0
                        # accept after `hold` stable frames; repeat the same sign only after a release
                        if cand_n >= hold and (k != last or released):
                            emit(labels[k])
                            last, released = k, False
                    else:
                        cand, cand_n = -1, 0
                        idle += 1
                        if idle >= hold:
                            released = True

            # ---- UI
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            if headless:
                continue
            draw_hands(frame, hands, present)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
            text = " ".join(list(transcript)) + (" " + spelling + "_" if spelling else "")
            put_text(frame, text[-48:], (10, h - 17), 0.8)
            if ema is not None and present.any():
                for r, i in enumerate(np.argsort(-ema)[:3]):
                    y = 30 + r * 26
                    cv2.rectangle(frame, (10, y - 16), (10 + int(160 * ema[i]), y + 4), (0, 180, 0) if r == 0 else (90, 90, 90), -1)
                    put_text(frame, f"{pretty(labels[i]) or labels[i]} {ema[i]:.2f}", (14, y), 0.55)
            if flash and time.monotonic() - flash_t < 1.0:
                put_text(frame, flash, (w // 2 - 15 * len(flash), h // 2), 1.6, (0, 255, 255), 3)
            put_text(frame, f"{fps:4.1f} fps  voice {'on' if speaker.enabled else 'off'}", (w - 210, 25), 0.55)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c"):
                transcript.clear()
                spelling = ""
            if key == ord("v"):
                speaker.enabled = not speaker.enabled
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        tracker.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
