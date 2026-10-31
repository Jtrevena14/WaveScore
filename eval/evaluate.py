"""
Score backends and filters against MAPS ground truth.

    python -m eval.evaluate --backend basic-pitch piano --filter none default tuned learned cnn --split test

Prints one row per backend x filter. Chord scores use only the UCHO/RAND files (chord sets).
"""

import argparse

from backends import BACKENDS
from eval.cache import DEFAULT_CACHE, load_dataset
from eval.maps import DEFAULT_ROOT, DEFAULT_SCHEME, SCHEMES, SPLITS, SUBSETS, find_files
from eval.metrics import OFFSET_RATIO, ChordScore, Counts, note_counts
from pipeline import FILTER_MODES, make_filter

CHORD_SUBSETS = {"UCHO", "RAND"}


def score(data, apply_filter):
    onset, onset_offset, chords = Counts(), Counts(), ChordScore()
    for f, raw, reference in data:
        predicted = apply_filter(raw, f.audio)
        onset += note_counts(predicted, reference)
        onset_offset += note_counts(predicted, reference, offset_ratio=OFFSET_RATIO)
        if f.subset in CHORD_SUBSETS:
            chords.add(predicted, reference)
    return onset, onset_offset, chords


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--filter", nargs="+", choices=FILTER_MODES, default=["none", "default"])
    parser.add_argument("--subset", nargs="+", choices=SUBSETS, default=["UCHO", "RAND", "MUS"])
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--limit", type=int, help="Files per piano and subset (fixed random sample)")
    parser.add_argument("--maps", default=str(DEFAULT_ROOT), help="MAPS root folder")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Raw-notes cache folder")
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME, help="Train/val/test split (eval/maps.py)")
    args = parser.parse_args()

    files = find_files(args.maps, args.subset, args.split, args.limit, scheme=args.scheme)
    print(f"{len(files)} files ({args.split} split, subsets {' '.join(args.subset)})\n")
    if not files:
        return

    header = f"{'backend':<12} {'filter':<8} {'P':>6} {'R':>6} {'F1':>6}  {'F1+off':>6}  {'chords':>7} {'cnt err':>7}"
    rows = []
    for backend in args.backend:
        data = load_dataset(files, backend, args.cache)
        for mode in args.filter:
            try:
                apply_filter, _description = make_filter(mode, backend)
            except FileNotFoundError as e:
                print(f"skip {backend}/{mode}: {e}")
                continue
            onset, onset_offset, chords = score(data, apply_filter)
            rows.append(
                f"{backend:<12} {mode:<8} {onset.precision:6.3f} {onset.recall:6.3f} {onset.f1:6.3f}  "
                f"{onset_offset.f1:6.3f}  {chords.exact_rate:7.3f} {chords.mean_count_error:7.2f}"
            )

    print(header)
    print("-" * len(header))
    print("\n".join(rows))
    print("\nP = precision (low means phantom notes), R = recall (low means missed notes), F1 on onsets.")
    print("F1+off also requires offsets within 20%. chords = share of reference chords matched exactly;")
    print("cnt err = mean |predicted - reference| onset groups per file.")


if __name__ == "__main__":
    main()
