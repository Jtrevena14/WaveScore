"""
Train the CNN phantom-note filter on MAPS.

    python -m eval.train_cnn --backend piano

Each predicted note becomes one example: a constant-Q patch around it plus its per-note
features (cnn_filter.py), labelled 1 if it matches a real note in the ground truth and 0 if
it is a phantom. Early-stops and picks the keep threshold on the validation split, reports
the test split, and saves models/note_filter_<backend>.pt (used by --filter cnn / auto).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from backends import BACKENDS
from cnn_filter import (FRAMES, PATCH_CONFIG, PITCH_BINS, PITCH_JITTER, TIME_JITTER, build_model,
                        centre, compute_cqt, cut_patches, score_notes)
from eval.cache import DEFAULT_CACHE, load_dataset
from eval.maps import DEFAULT_ROOT, DEFAULT_SCHEME, SCHEMES, SUBSETS, find_files
from eval.metrics import correct_labels
from eval.train_classifier import best_threshold, f1_at
from features import FEATURE_NAMES, note_features
from pipeline import cnn_model_path, learned_merge_gap
from postprocess import merge_fragments


def cached_patches(f, notes, cache_dir):
    """Patches for `notes` in MAPS file `f`, from the patch cache if it still matches the notes."""
    path = Path(cache_dir) / f"{f.stem}.npz"
    key = np.array([[n.midi, round(n.start * 1000)] for n in notes], dtype=np.int64).reshape(-1, 2)
    if path.exists():
        cached = np.load(path)
        if np.array_equal(cached["key"], key) and json.loads(str(cached["config"])) == PATCH_CONFIG:
            return cached["patches"]
    patches = cut_patches(compute_cqt(f.audio), notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, patches=patches, key=key, config=json.dumps(PATCH_CONFIG))
    return patches


def build_split(data, merge_gap, patch_dir):
    """Patches, features, labels, and the number of reference notes (for recall)."""
    patches, feats, labels, n_reference = [], [], [], 0
    for i, (f, raw, reference) in enumerate(data, 1):
        notes = merge_fragments(raw, merge_gap) if merge_gap > 0 else raw
        n_reference += len(reference)
        if not notes:
            continue
        if i % 200 == 0:
            print(f"  patches: {i}/{len(data)} files", flush=True)
        patches.append(cached_patches(f, notes, patch_dir))
        feats.append(note_features(notes))
        labels.append(correct_labels(notes, reference))
    if not labels:
        return None
    return np.concatenate(patches), np.vstack(feats), np.concatenate(labels).astype(np.float32), n_reference


def jitter(batch, rng):
    """Random crop within the cached margins, plus a random contrast change."""
    out = np.empty((len(batch), PITCH_BINS, FRAMES), dtype=np.float32)
    for i, patch in enumerate(batch):
        r = rng.randint(0, 2 * PITCH_JITTER)
        c = rng.randint(0, 2 * TIME_JITTER)
        out[i] = patch[r:r + PITCH_BINS, c:c + FRAMES] * rng.uniform(0.85, 1.15)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--subset", nargs="+", choices=SUBSETS, default=["UCHO", "RAND", "MUS"])
    parser.add_argument("--limit", type=int, help="Files per piano and subset (fixed random sample)")
    parser.add_argument("--maps", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Raw-notes cache folder")
    parser.add_argument("--patch-cache", help="Spectrogram-patch cache folder (default <maps>/../MAPS_cache/patches)")
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME, help="Train/val/test split (eval/maps.py)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without a better val F1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch import nn

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    patch_dir = Path(args.patch_cache or Path(args.maps).resolve().parent / "MAPS_cache" / "patches") / args.backend

    merge_gap = learned_merge_gap(args.backend)
    splits = {}
    for split in ("train", "val", "test"):
        files = find_files(args.maps, args.subset, split, args.limit, scheme=args.scheme)
        print(f"{split}: {len(files)} files", flush=True)
        splits[split] = build_split(load_dataset(files, args.backend, args.cache), merge_gap, patch_dir) if files else None

    if splits["train"] is None or splits["val"] is None:
        raise SystemExit(f"Need both training and validation files under split scheme {args.scheme}")
    p_train, x_train, y_train, _ = splits["train"]
    p_val, x_val, y_val, n_val = splits["val"]
    p_val = centre(p_val)
    print(f"\nTraining notes: {len(y_train)} ({y_train.mean():.1%} real); validation notes: {len(y_val)}")
    print(f"No filter on validation: F1 {f1_at(np.ones(len(y_val)), y_val, n_val, 0.5)[0]:.3f}")
    print(f"Device: {device}\n")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train_std = ((x_train - mean) / std).astype(np.float32)

    model = build_model().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_real = float(y_train.sum())
    pos_weight = torch.tensor((len(y_train) - n_real) / max(n_real, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best = None  # (val F1, threshold, state_dict, epoch)
    order = list(range(len(y_train)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            patches = torch.from_numpy(jitter(p_train[idx], rng)).to(device)
            feats = torch.from_numpy(x_train_std[idx]).to(device)
            labels = torch.from_numpy(y_train[idx]).to(device)
            loss = loss_fn(model(patches, feats), labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)

        scores = score_notes(model, p_val, x_val, mean, std, device)
        threshold, f1, p, r = best_threshold(scores, y_val, n_val)
        marker = ""
        if best is None or f1 > best[0]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = (f1, threshold, state, epoch)
            marker = "  *"
        print(f"epoch {epoch:>3}: loss {total / len(order):.4f}  val F1 {f1:.3f} (P {p:.3f}, R {r:.3f}) "
              f"at threshold {threshold:.2f}{marker}", flush=True)
        if epoch - best[3] >= args.patience:
            print(f"No improvement for {args.patience} epochs; stopping")
            break

    val_f1, threshold, state, epoch = best
    model.load_state_dict(state)
    print(f"\nBest: epoch {epoch}, val F1 {val_f1:.3f} at threshold {threshold:.2f}")

    report = {}
    if splits["test"] is not None:
        p_test, x_test, y_test, n_test = splits["test"]
        base = f1_at(np.ones(len(y_test)), y_test, n_test, 0.5)[0]
        f1, p, r = f1_at(score_notes(model, centre(p_test), x_test, mean, std, device), y_test, n_test, threshold)
        report["test_f1"] = f1
        print(f"Test: cnn F1 {f1:.3f} (P {p:.3f}, R {r:.3f}); no filter F1 {base:.3f}")

    path = cnn_model_path(args.backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state, "threshold": threshold, "features": FEATURE_NAMES, "patch": PATCH_CONFIG,
        "mean": mean.tolist(), "std": std.tolist(), "backend": args.backend, "merge_gap": merge_gap,
        "val_f1": val_f1, "epoch": epoch, "scheme": args.scheme, **report,
    }, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
