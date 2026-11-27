"""Train the ISL sign classifier on data/<label>/*.npz (from collect.py and import_dataset.py).

    python train.py                  # default 60 epochs
    python train.py --no-mirror      # don't treat mirrored signing as the same sign

Writes models/isl_classifier.npz, which run.py loads.
"""
import argparse
import copy
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isl_core import (CLASSIFIER_PATH, DATA_DIR, FRAME_DIM, MODEL_DIR, NONE_LABEL, WINDOW,
                      SignClassifier, frame_features, window_features, window_ready)

CHANNELS = 96
DILATIONS = (1, 2, 4)   # receptive field 15 frames: covers the whole window
STRIDE = 2              # step between training windows inside a clip


class Net(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        layers, c = [nn.Dropout(0.1)], n_in
        for d in DILATIONS:
            layers += [nn.Conv1d(c, CHANNELS, 3, padding=d, dilation=d), nn.BatchNorm1d(CHANNELS), nn.ReLU()]
            c = CHANNELS
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * CHANNELS, n_classes)

    def forward(self, x):                      # x (N, T, C)
        x = self.conv(x.transpose(1, 2))
        return self.fc(self.drop(torch.cat([x.mean(2), x.amax(2)], 1)))


# --------------------------------------------------------------------------- data
class Dataset:
    """clips: recorded / video sequences (y, hands[L,2,21,3], present[L,2], aspect).
    static: single hand poses from image / landmark datasets, used as a held sign."""

    def __init__(self, labels):
        self.labels = labels
        self.clips = []
        st_h, st_p, st_a, st_y, st_f = [], [], [], [], []
        self.sources = []   # name of each static file, e.g. 'import_csv'
        for y, label in enumerate(labels):
            for f in sorted(glob.glob(os.path.join(DATA_DIR, label, "*.npz"))):
                d = np.load(f)
                hands, present, aspect = d["hands"].astype(np.float32), d["present"].astype(bool), float(d["aspect"])
                if "static" in d:
                    st_h.append(hands)
                    st_p.append(present)
                    st_a.append(np.full(len(hands), aspect, np.float32))
                    st_y.append(np.full(len(hands), y, np.int64))
                    st_f.append(np.full(len(hands), len(self.sources)))
                    self.sources.append(os.path.splitext(os.path.basename(f))[0])
                else:
                    self.clips.append((y, hands, present, aspect))
        empty = np.zeros((0, 2, 21, 3), np.float32)
        self.st_h = np.concatenate(st_h) if st_h else empty
        self.st_p = np.concatenate(st_p) if st_p else np.zeros((0, 2), bool)
        self.st_a = np.concatenate(st_a) if st_a else np.zeros(0, np.float32)
        self.st_y = np.concatenate(st_y) if st_y else np.zeros(0, np.int64)
        self.st_f = np.concatenate(st_f) if st_f else np.zeros(0, int)

    def split(self, rng, val_frac):
        """Per class: hold out whole clips, and the tail block of each static file (neighbouring
        dataset images are near-duplicates, so a random split would overstate accuracy)."""
        clip_tr, clip_va = [], []
        by_class = {}
        for i, c in enumerate(self.clips):
            by_class.setdefault(c[0], []).append(i)
        for ids in by_class.values():
            rng.shuffle(ids)
            k = max(1, round(val_frac * len(ids))) if len(ids) >= 4 else 0
            clip_va += ids[:k]
            clip_tr += ids[k:]
        st_tr, st_va = [], []
        for f in np.unique(self.st_f):
            ids = np.flatnonzero(self.st_f == f)
            k = int(round(val_frac * len(ids))) if len(ids) >= 10 else 0
            st_tr.append(ids[:len(ids) - k])
            st_va.append(ids[len(ids) - k:])
        cat = lambda a: np.concatenate(a) if a else np.zeros(0, int)
        return (clip_tr, cat(st_tr)), (clip_va, cat(st_va))

    def windows(self, part, rng, per_class, aug=False, mirror=False):
        """Training windows (N, WINDOW, FRAME_DIM) and labels. Static samples are capped at
        `per_class` per class and source file (re-drawn every epoch), so huge datasets stay cheap
        and every source gets equal weight."""
        clip_ids, st_ids = part
        xs, ys = [], []
        for i in clip_ids:
            y, hands, present, aspect = self.clips[i]
            hands, present, aspect = hands[None], present[None], np.array([aspect], np.float32)
            if aug:
                n = max(int(round(hands.shape[1] / rng.uniform(0.75, 1.3))), 2)   # signing speed
                idx = np.round(np.linspace(0, hands.shape[1] - 1, n)).astype(int)
                hands, present = augment(hands[:, idx], present[:, idx], aspect, rng, mirror)
                aspect = np.ones(1, np.float32)
            w = clip_windows(frame_features(hands[0], present[0], aspect[0]))
            xs.append(w)
            ys.append(np.full(len(w), y, np.int64))
        if len(st_ids):
            pick = []
            for f in np.unique(self.st_f[st_ids]):
                ids = st_ids[self.st_f[st_ids] == f]
                pick.append(rng.choice(ids, min(per_class, len(ids)), replace=False))
            pick = np.concatenate(pick)
            hands = np.repeat(self.st_h[pick][:, None], WINDOW, 1)               # held pose
            present = np.repeat(self.st_p[pick][:, None], WINDOW, 1)
            aspect = self.st_a[pick]
            if aug:
                hands, present = augment(hands, present, aspect, rng, mirror)
                aspect = np.ones_like(aspect)
            w = frame_features(hands, present, aspect[:, None, None, None])
            ok = window_ready(w)
            xs.append(w[ok])
            ys.append(self.st_y[pick][ok])
        return np.concatenate(xs), np.concatenate(ys)


def augment(hands, present, aspect, rng, mirror):
    """Batch augmentation of clips hands (N, L, 2, 21, 3), present (N, L, 2), aspect (N,):
    rotation, scale, aspect stretch, shear, shift, landmark jitter, tracker dropouts, mirroring.
    Returns coordinates with the aspect ratio applied (use aspect=1 afterwards)."""
    n = len(hands)
    h, p = hands.copy(), present.copy()
    a = np.asarray(aspect, np.float32).reshape(n, 1, 1, 1)
    h[..., 0] *= a
    ang = np.deg2rad(rng.uniform(-15, 15, n))
    s = rng.uniform(0.8, 1.2, n)
    m = np.zeros((n, 2, 2))
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = np.cos(ang), -np.sin(ang), np.sin(ang), np.cos(ang)
    shear = np.tile(np.eye(2), (n, 1, 1))
    shear[:, 0, 1], shear[:, 1, 0] = rng.uniform(-0.1, 0.1, n), rng.uniform(-0.1, 0.1, n)
    stretch = np.tile(np.eye(2), (n, 1, 1))
    stretch[:, 0, 0] = rng.uniform(0.88, 1.12, n)            # unknown camera aspect ratios
    m = s[:, None, None] * m @ shear @ stretch
    c = np.stack([0.5 * a[:, 0, 0, 0], np.full(n, 0.5)], -1)[:, None, None, None]
    shift = rng.uniform(-0.1, 0.1, (n, 1, 1, 1, 2))
    h[..., :2] = np.einsum("nltkj,nij->nltki", h[..., :2] - c, m) + c + shift
    h[..., 2] *= s.reshape(n, 1, 1, 1)
    h += rng.normal(0, 0.004, h.shape).astype(np.float32)
    p &= rng.random(p.shape) > 0.03                          # tracker dropouts
    if mirror:
        f = rng.random(n) < 0.5
        h[f, ..., 0] = a[f] - h[f, ..., 0]
        h[f], p[f] = h[f][:, :, ::-1], p[f][:, :, ::-1]
    return h.astype(np.float32), p


def clip_windows(feats):
    """All usable WINDOW-length windows of one clip's frame features -> (n, T, FRAME_DIM)."""
    if len(feats) < WINDOW:
        feats = np.concatenate([np.zeros((WINDOW - len(feats), FRAME_DIM), np.float32), feats])
    starts = np.arange(0, len(feats) - WINDOW + 1, STRIDE)
    w = feats[starts[:, None] + np.arange(WINDOW)]
    return w[window_ready(w)]


# --------------------------------------------------------------------------- export
def export(net, labels, mu, sd, path):
    net.eval()
    d = {"labels": np.array(labels), "window": np.int32(WINDOW), "mu": mu, "sd": sd}
    convs = [m for m in net.conv if isinstance(m, nn.Conv1d)]
    bns = [m for m in net.conv if isinstance(m, nn.BatchNorm1d)]
    with torch.no_grad():
        for i, (cv, bn) in enumerate(zip(convs, bns)):
            g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            d[f"w{i}"] = (cv.weight * g[:, None, None]).permute(2, 1, 0).contiguous().numpy()
            d[f"b{i}"] = ((cv.bias - bn.running_mean) * g + bn.bias).numpy()
            d[f"dil{i}"] = np.int32(cv.dilation[0])
        d["n_conv"] = np.int32(len(convs))
        d["fc_w"] = net.fc.weight.T.contiguous().numpy()
        d["fc_b"] = net.fc.bias.numpy()
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(path, **{k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in d.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val", type=float, default=0.15, help="fraction held out for validation")
    ap.add_argument("--per-class", type=int, default=500, help="static samples per class per epoch")
    ap.add_argument("--no-mirror", action="store_true", help="don't learn mirrored (other-hand) signing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    mirror = not args.no_mirror

    labels = sorted(d for d in os.listdir(DATA_DIR) if glob.glob(os.path.join(DATA_DIR, d, "*.npz")))
    if len(labels) < 2:
        raise SystemExit(f"Need at least 2 labels in {DATA_DIR} (collect.py / import_dataset.py).")
    if NONE_LABEL not in labels:
        print(f"WARNING: no '{NONE_LABEL}' class recorded - expect false detections between signs.")
    data = Dataset(labels)
    train_part, val_part = data.split(rng, args.val)
    print(f"{len(labels)} classes | {len(data.clips)} clips, {len(data.st_y)} static samples | "
          f"held out: {len(val_part[0])} clips, {len(val_part[1])} static")

    x_plain, _ = data.windows(train_part, rng, args.per_class)
    feats = window_features(x_plain).reshape(-1, 2 * FRAME_DIM)
    mu = feats.mean(0).astype(np.float32)
    sd = feats.std(0).astype(np.float32)
    sd[sd < 1e-4] = 1.0
    del feats

    def to_tensor(x):
        return torch.from_numpy((window_features(x) - mu) / sd)

    has_val = len(val_part[0]) + len(val_part[1]) > 0
    if has_val:
        xv, yv = data.windows(val_part, rng, 200)
        xv_t, yv_t = to_tensor(xv), torch.from_numpy(yv)

    net = Net(2 * FRAME_DIM, len(labels))
    print(f"model parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    n_per_epoch = max(len(x_plain), 4 * args.batch)
    steps = -(-n_per_epoch // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps)

    best, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        xt, yt = data.windows(train_part, rng, args.per_class, aug=True, mirror=mirror)  # fresh augmentation
        counts = np.bincount(yt, minlength=len(labels)).astype(float)
        weights = 1.0 / counts[yt]                                       # class-balanced sampling
        order = rng.choice(len(yt), n_per_epoch, p=weights / weights.sum())
        net.train()
        total = correct = 0
        loss_sum = 0.0
        for s in range(steps):
            idx = order[s * args.batch:(s + 1) * args.batch]
            xb, yb = to_tensor(xt[idx]), torch.from_numpy(yt[idx])
            logits = net(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(idx)
        msg = f"epoch {epoch:3d}  loss {loss_sum / total:.3f}  train acc {correct / total:.3f}"
        score = correct / total
        if has_val:
            net.eval()
            with torch.no_grad():
                score = (net(xv_t).argmax(1) == yv_t).float().mean().item()
            msg += f"  val acc {score:.3f}"
        if score >= best:
            best, best_state = score, copy.deepcopy(net.state_dict())
            msg += "  *"
        print(msg, flush=True)

    net.load_state_dict(best_state)
    export(net, labels, mu, sd, CLASSIFIER_PATH)
    print(f"\nsaved {CLASSIFIER_PATH}  (best {'val' if has_val else 'train'} acc {best:.3f})")

    if has_val:  # check the numpy runtime matches torch, then report per-class accuracy
        clf = SignClassifier(CLASSIFIER_PATH)
        with torch.no_grad():
            ref = F.softmax(net(xv_t[:256]), 1).numpy()
        assert np.allclose(clf(xv[:256]), ref, atol=1e-4), "numpy export mismatch"
        clip_va, st_va = val_part
        groups = {"recorded/video clips": (clip_va, st_va[:0])}
        for name in sorted(set(data.sources)):
            ids = [i for i, s in enumerate(data.sources) if s == name]
            groups[name] = ([], st_va[np.isin(data.st_f[st_va], ids)])
        print("\nval accuracy by source:")
        for name, part in groups.items():
            if len(part[0]) + len(part[1]):
                xs, ys = data.windows(part, rng, 200)
                print(f"  {name:>22s}  {(clf(xs).argmax(1) == ys).mean():6.1%}  (n={len(ys)})")
        pred = clf(xv).argmax(1)
        conf = np.zeros((len(labels), len(labels)), int)
        np.add.at(conf, (yv, pred), 1)
        print("\nper-class val accuracy:")
        for i, label in enumerate(labels):
            n = conf[i].sum()
            if n:
                wrong = [(conf[i, j], labels[j]) for j in np.argsort(-conf[i]) if j != i and conf[i, j]]
                extra = "  confused with: " + ", ".join(f"{l}({c})" for c, l in wrong[:3]) if wrong else ""
                print(f"  {label:>14s}  {conf[i, i] / n:6.1%}  (n={n}){extra}")


if __name__ == "__main__":
    main()
