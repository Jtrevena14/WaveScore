"""
Train the learned phantom-note filter on MAPS.

    python -m eval.train_classifier --backend piano

Each predicted note becomes one row of features (features.py), labelled 1 if it matches a
real note in the ground truth and 0 if it is a phantom. Trains logistic regression (prints
its weights and bias) and gradient boosting, picks each one's threshold for best F1 on the
validation piano, keeps the better model, and saves models/note_filter_<backend>.joblib.
"""

import argparse

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backends import BACKENDS
from eval.cache import DEFAULT_CACHE, load_dataset
from eval.maps import DEFAULT_ROOT, DEFAULT_SCHEME, SCHEMES, SUBSETS, find_files
from eval.metrics import correct_labels
from features import FEATURE_NAMES, note_features
from pipeline import learned_merge_gap, learned_model_path
from postprocess import merge_fragments


def build_rows(data, merge_gap):
    """Features, labels, and the number of reference notes (for recall)."""
    xs, ys, n_reference = [], [], 0
    for _f, raw, reference in data:
        notes = merge_fragments(raw, merge_gap) if merge_gap > 0 else raw
        n_reference += len(reference)
        if notes:
            xs.append(note_features(notes))
            ys.append(correct_labels(notes, reference))
    x = np.vstack(xs) if xs else np.zeros((0, len(FEATURE_NAMES)))
    y = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    return x, y, n_reference


def f1_at(scores, y, n_reference, threshold):
    keep = scores >= threshold
    tp = int((keep & (y == 1)).sum())
    fp = int((keep & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_reference if n_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def best_threshold(scores, y, n_reference):
    thresholds = np.linspace(0.05, 0.95, 91)
    results = [(f1_at(scores, y, n_reference, t), t) for t in thresholds]
    (f1, p, r), t = max(results, key=lambda item: item[0][0])
    return float(t), f1, p, r


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--subset", nargs="+", choices=SUBSETS, default=["UCHO", "RAND", "MUS"])
    parser.add_argument("--limit", type=int, help="Files per piano and subset (fixed random sample)")
    parser.add_argument("--maps", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Raw-notes cache folder")
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME, help="Train/val/test split (eval/maps.py)")
    args = parser.parse_args()

    merge_gap = learned_merge_gap(args.backend)
    splits = {}
    for split in ("train", "val", "test"):
        files = find_files(args.maps, args.subset, split, args.limit, scheme=args.scheme)
        splits[split] = build_rows(load_dataset(files, args.backend, args.cache), merge_gap) if files else None

    x_train, y_train, _ = splits["train"]
    if splits["val"] is None:
        raise SystemExit(f"No validation files found for split scheme {args.scheme}; can't choose a threshold")
    x_val, y_val, n_val = splits["val"]
    print(f"Training rows: {len(y_train)} ({y_train.mean():.1%} real notes); validation rows: {len(y_val)}")
    print(f"No filter on validation: F1 {f1_at(np.ones(len(y_val)), y_val, n_val, 0.5)[0]:.3f}\n")

    candidates = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
    }
    best = None
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        threshold, f1, p, r = best_threshold(model.predict_proba(x_val)[:, 1], y_val, n_val)
        print(f"{name}: validation F1 {f1:.3f} (P {p:.3f}, R {r:.3f}) at threshold {threshold:.2f}")

        if name == "logistic":
            lr = model[-1]
            print("  learned weights (on standardised features; + means 'more likely a real note'):")
            for feature, weight in sorted(zip(FEATURE_NAMES, lr.coef_[0]), key=lambda fw: -abs(fw[1])):
                print(f"    {weight:+7.3f}  {feature}")
            print(f"    {lr.intercept_[0]:+7.3f}  (bias)")

        if best is None or f1 > best[2]:
            best = (name, model, f1, threshold)

    name, model, val_f1, threshold = best
    if splits["test"] is not None:
        x_test, y_test, n_test = splits["test"]
        base = f1_at(np.ones(len(y_test)), y_test, n_test, 0.5)[0]
        f1, p, r = f1_at(model.predict_proba(x_test)[:, 1], y_test, n_test, threshold)
        print(f"\nTest:{name} F1 {f1:.3f} (P {p:.3f}, R {r:.3f}); no filter F1 {base:.3f}")

    path = learned_model_path(args.backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "kind": name, "threshold": threshold, "features": FEATURE_NAMES,
        "backend": args.backend, "merge_gap": merge_gap, "val_f1": val_f1,
    }, path)
    print(f"Saved {path} ({name})")


if __name__ == "__main__":
    main()
