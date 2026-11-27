"""Record training clips for ISL signs.

    python collect.py A B C hello thank_you _none --clips 30

For each label press SPACE, then perform the sign; clips are recorded back to back with a
short pause in between. Between clips, change your position, distance and angle a little,
and for moving signs vary the speed - variety is what makes the model accurate.

Always record a "_none" label as well (hands resting, scratching, moving between signs,
random gestures - hands visible) so the recogniser knows when NOT to speak.

Keys: SPACE start label · n next label · q quit
"""
import argparse
import os
import time

import cv2
import numpy as np

from isl_core import (DATA_DIR, TRAIN_FPS, Camera, ParallelHandTracker, SerialHandTracker, draw_hands,
                      pretty, put_text, resample_clip)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="sign names, e.g. A B hello thank_you _none")
    ap.add_argument("--clips", type=int, default=30, help="clips per label (default 30)")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames per clip at 30 fps (default 40 = 1.33 s). Recording is timed, so slower "
                         "cameras/CPUs (Raspberry Pi) give the same duration, resampled to 30 fps")
    ap.add_argument("--gap", type=float, default=0.8, help="pause between clips in seconds")
    ap.add_argument("--source", default="0", help="camera index, \"picam\" (Raspberry Pi ribbon camera), or stream URL")
    args = ap.parse_args()

    cam = Camera(args.source)
    workers = 1
    tracker = ParallelHandTracker(workers) if workers > 1 else SerialHandTracker()
    clip_secs = args.frames / TRAIN_FPS
    win = "ISL - collect"
    seq = 0
    try:
        for label in args.labels:
            out_dir = os.path.join(DATA_DIR, label)
            os.makedirs(out_dir, exist_ok=True)
            existing = len([f for f in os.listdir(out_dir) if f.endswith(".npz")])
            state, t_state, clip = "wait", 0.0, 0
            rec_t, rec_h, rec_p = [], [], []
            while clip < args.clips:
                seq, frame = cam.read(seq)
                if frame is None:
                    if not cam.running:
                        return
                    continue
                res = tracker.step(cv2.flip(frame, 1), time.monotonic())
                if res is None:
                    continue
                t_cap, frame, hands, present = res
                now = t_cap

                if state == "rec":
                    if not rec_t:
                        t_rec0 = now
                    rec_t.append(now)
                    rec_h.append(hands)
                    rec_p.append(present)
                    if now - t_rec0 >= clip_secs:
                        h30, p30 = resample_clip(rec_t, rec_h, rec_p, args.frames)
                        np.savez_compressed(os.path.join(out_dir, f"{int(time.time() * 1000)}.npz"),
                                            hands=h30, present=p30, aspect=frame.shape[1] / frame.shape[0])
                        clip += 1
                        rec_t, rec_h, rec_p = [], [], []
                        state, t_state = "gap", now
                elif state in ("countdown", "gap") and now - t_state >= (3.0 if state == "countdown" else args.gap):
                    state = "rec"

                draw_hands(frame, hands, present)
                put_text(frame, f"Sign: {pretty(label) or label}", (10, 35), 1.0, (0, 255, 255))
                put_text(frame, f"clip {clip}/{args.clips}   (saved before: {existing})", (10, 70), 0.6)
                if state == "wait":
                    put_text(frame, "SPACE = start   n = next   q = quit", (10, frame.shape[0] - 15), 0.6)
                elif state == "countdown":
                    put_text(frame, str(3 - int(now - t_state)), (frame.shape[1] // 2 - 20, frame.shape[0] // 2), 3.0, (0, 255, 255), 5)
                elif state == "gap":
                    put_text(frame, "reposition...", (10, frame.shape[0] - 15), 0.8, (0, 255, 255))
                else:
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                    if not present.any():
                        put_text(frame, "no hands detected!", (10, frame.shape[0] - 15), 0.8, (0, 0, 255))
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key == ord("n"):
                    break
                if key == ord(" ") and state == "wait":
                    state, t_state = "countdown", now
            print(f"{label}: recorded {clip} clips")
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
