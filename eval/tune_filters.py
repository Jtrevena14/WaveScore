"""
Find hand-filter settings automatically by random search on the MAPS training split.

    python -m eval.tune_filters --backend piano --trials 300

Maximises note F1 on the train split, reports the val/test score of the best settings,
and saves them to models/filter_settings_<backend>.json (used by --filter tuned / auto).
"""

import argparse
import random
from dataclasses import asdict

from backends import BACKENDS
from eval.cache import DEFAULT_CACHE, load_dataset
from eval.maps import DEFAULT_ROOT, DEFAULT_SCHEME, SCHEMES, SUBSETS, find_files
from eval.metrics import Counts, note_counts
from pipeline import default_settings, save_settings, tuned_settings_path
from postprocess import FilterSettings, filter_notes

# Search range for each setting. 0 (filter off) is always reachable.
RANGES = {
    "merge_gap": (0.0, 0.1),
    "min_duration": (0.0, 0.2),
    "min_amplitude": (0.0, 0.8),
    "relative_amplitude": (0.0, 0.8),
    "harmonic_ratio": (0.0, 1.5),
    "onset_tolerance": (0.01, 0.1),
}
P_OFF = 0.2  # chance that a filter is switched off in a trial


def random_settings(rng):
    values = {}
    for name, (lo, hi) in RANGES.items():
        off = name != "onset_tolerance" and rng.random() < P_OFF
        values[name] = 0.0 if off else round(rng.uniform(lo, hi), 3)
    return FilterSettings(**values)


def f1_of(data, settings):
    total = Counts()
    for _f, raw, reference in data:
        total += note_counts(filter_notes(raw, settings)[0], reference)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--subset", nargs="+", choices=SUBSETS, default=["UCHO", "RAND", "MUS"])
    parser.add_argument("--limit", type=int, default=30, help="Files per piano and subset (default 30)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maps", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Raw-notes cache folder")
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME, help="Train/val/test split (eval/maps.py)")
    args = parser.parse_args()

    train = load_dataset(find_files(args.maps, args.subset, "train", args.limit, scheme=args.scheme), args.backend, args.cache)
    print(f"Tuning on {len(train)} training files, {args.trials} trials")

    rng = random.Random(args.seed)
    candidates = [default_settings(args.backend)] + [random_settings(rng) for _ in range(args.trials)]
    best, best_counts = None, None
    for i, settings in enumerate(candidates):
        counts = f1_of(train, settings)
        if best_counts is None or counts.f1 > best_counts.f1:
            best, best_counts = settings, counts
            print(f"  trial {i:>4}: F1 {counts.f1:.3f} (P {counts.precision:.3f}, R {counts.recall:.3f})  {asdict(settings)}")

    report = {"train_f1": round(best_counts.f1, 4)}
    for split in ("val", "test"):
        files = find_files(args.maps, args.subset, split, args.limit, scheme=args.scheme)
        if files:
            counts = f1_of(load_dataset(files, args.backend, args.cache), best)
            report[f"{split}_f1"] = round(counts.f1, 4)
            print(f"{split}: F1 {counts.f1:.3f} (P {counts.precision:.3f}, R {counts.recall:.3f})")

    path = tuned_settings_path(args.backend)
    save_settings(path, best, backend=args.backend, trials=args.trials, **report)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
