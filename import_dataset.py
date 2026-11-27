"""Convert public ISL datasets into training data (data/<label>/*.npz) for train.py.

    python import_dataset.py images "C:\\Users\\KIIT\\Downloads\\archive (1).zip"   # class-folder images (zip or folder)
    python import_dataset.py csv "C:\\Users\\KIIT\\Downloads\\archive.zip"          # landmark CSV, target 0-25 = A-Z
    python import_dataset.py videos C:\\Users\\KIIT\\Downloads\\INCLUDE              # INCLUDE videos (zips or extracted)

Images / CSV rows become "static" samples (one hand pose = a held sign); videos become
normal clips, resampled to 30 fps and trimmed to the part where hands are visible.
"""
import argparse
import csv
import io
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from isl_core import DATA_DIR

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".webm")
FPS = 30


def save(label, name, **arrays):
    out = os.path.join(DATA_DIR, label)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, name + ".npz"), **arrays)


def workers():
    return max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------------------------- images
def _image_worker(job):
    src, label, names = job
    from isl_core import HandTracker
    tracker = HandTracker(video=False)
    zf = zipfile.ZipFile(src) if zipfile.is_zipfile(src) else None
    hands, present, aspect = [], [], 1.0
    for name in names:
        data = zf.read(name) if zf else np.fromfile(name, np.uint8)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, p = tracker(img)
        if p.any():
            hands.append(h)
            present.append(p)
            aspect = img.shape[1] / img.shape[0]
    tracker.close()
    if hands:
        save(label, "import_images", hands=np.array(hands), present=np.array(present),
             aspect=aspect, static=True)
    return label, len(hands), len(names)


def import_images(src, per_class):
    if zipfile.is_zipfile(src):
        names = zipfile.ZipFile(src).namelist()
    else:
        names = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs]
    groups = {}
    for n in names:
        parts = re.split(r"[\\/]", n)
        if n.lower().endswith(IMAGE_EXT) and len(parts) >= 2 and ".ipynb_checkpoints" not in parts:
            groups.setdefault(parts[-2], []).append(n)
    print(f"{len(groups)} classes, {sum(map(len, groups.values()))} images - extracting hand landmarks ...")
    jobs = [(src, label, files[:per_class]) for label, files in sorted(groups.items())]
    with ProcessPoolExecutor(workers()) as ex:
        for label, kept, total in ex.map(_image_worker, jobs):
            print(f"  {label:>12s}: hands found in {kept}/{total}")


# --------------------------------------------------------------------------- csv
def import_csv(src, aspect, numeric):
    """CSV columns: target, uses_two_hands, left_hand_{x,y,z}_{0..20}, right_hand_... (-1 = no hand)."""
    if zipfile.is_zipfile(src):
        zf = zipfile.ZipFile(src)
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        f = io.TextIOWrapper(zf.open(member), encoding="utf-8")
    else:
        f = open(src, encoding="utf-8")
    with f:
        reader = csv.reader(f)
        next(reader)
        rows = np.array([r for r in reader if r], np.float32)
    y = rows[:, 0].astype(int)
    hands = rows[:, 2:2 + 126].reshape(-1, 2, 21, 3)
    present = ~(hands == -1).all(axis=(2, 3))
    hands[~present] = 0
    for t in np.unique(y):
        m = (y == t) & present.any(1)
        label = str(t) if numeric else chr(ord("A") + t)
        save(label, "import_csv", hands=hands[m], present=present[m], aspect=aspect, static=True)
        print(f"  {label:>3s}: {m.sum()} samples")


# --------------------------------------------------------------------------- videos
def word_label(folder):
    """'48. Good Morning' -> 'good_morning'"""
    name = re.sub(r"^\s*\d+\s*[.)-]?\s*", "", folder).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _video_worker(job):
    path, label, name = job
    from isl_core import HandTracker
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > 640:
            f = cv2.resize(f, (640, round(f.shape[0] * 640 / f.shape[1])), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        return label, 0
    idx = np.round(np.arange(0, len(frames), src_fps / FPS)).astype(int)
    idx = idx[idx < len(frames)]
    tracker = HandTracker(video=True)
    hands, present = zip(*(tracker(frames[i]) for i in idx))
    tracker.close()
    hands, present = np.array(hands), np.array(present)
    seen = np.flatnonzero(present.any(1))
    if len(seen) < 8:
        return label, 0
    a, b = max(seen[0] - 3, 0), min(seen[-1] + 4, len(hands))  # trim idle start/end
    save(label, name, hands=hands[a:b], present=present[a:b],
         aspect=frames[0].shape[1] / frames[0].shape[0])
    return label, b - a


def import_videos(root, tmp):
    jobs = []
    for z in sorted(f for f in os.listdir(root) if f.lower().endswith(".zip")):
        with zipfile.ZipFile(os.path.join(root, z)) as zf:  # extract videos only, to a temp folder
            for n in zf.namelist():
                if n.lower().endswith(VIDEO_EXT) and not os.path.exists(os.path.join(tmp, n)):
                    print("extracting", n)
                    zf.extract(n, tmp)
    inside = os.path.abspath(tmp).startswith(os.path.abspath(root) + os.sep)
    for base in ([root] if inside else [root, tmp]):
        for r, _, fs in os.walk(base):
            for f in fs:
                if f.lower().endswith(VIDEO_EXT):
                    label = word_label(os.path.basename(r))
                    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.relpath(os.path.join(r, f), base))
                    jobs.append((os.path.join(r, f), label, "video_" + stem))
    print(f"{len(jobs)} videos - extracting hand landmarks ...")
    counts = {}
    with ProcessPoolExecutor(workers()) as ex:
        for label, n in ex.map(_video_worker, jobs):
            counts.setdefault(label, []).append(n)
    for label, ns in sorted(counts.items()):
        print(f"  {label:>20s}: {sum(n > 0 for n in ns)}/{len(ns)} videos usable")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("images")
    a.add_argument("src", help="zip or folder with one sub-folder per class")
    a.add_argument("--per-class", type=int, default=100000, help="max images per class")
    b = sub.add_parser("csv")
    b.add_argument("src", help="csv or zip containing it")
    b.add_argument("--aspect", type=float, default=1.0, help="width/height of the images the CSV came from")
    b.add_argument("--numeric", action="store_true", help="keep targets as numbers instead of A-Z")
    c = sub.add_parser("videos")
    c.add_argument("root", help="folder with INCLUDE zips and/or extracted videos")
    c.add_argument("--tmp", help="where to extract videos (default <root>/extracted)")
    args = ap.parse_args()

    if args.kind == "images":
        import_images(args.src, args.per_class)
    elif args.kind == "csv":
        import_csv(args.src, args.aspect, args.numeric)
    else:
        import_videos(args.root, args.tmp or os.path.join(args.root, "extracted"))


if __name__ == "__main__":
    main()
