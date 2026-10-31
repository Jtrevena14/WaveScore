"""
Finds MAPS (MIDI Aligned Piano Sounds) audio files and their ground-truth labels.

Each MAPS .wav sits next to a .txt with the same stem whose rows are
`OnsetTime  OffsetTime  MidiPitch`. File stems look like
`MAPS_<SUBSET>[-piece]_..._<PIANO>`, e.g. MAPS_UCHO_C0-4-7_I32-96_S0_n1_ENSTDkCl.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

from postprocess import NoteEvent

# Set SHEETCREATOR_MAPS to keep the dataset on another drive, e.g. `setx SHEETCREATOR_MAPS T:\MAPS`.
DEFAULT_ROOT = Path(os.environ.get("SHEETCREATOR_MAPS") or Path(__file__).resolve().parent.parent / "data" / "MAPS")

SUBSETS = ("ISOL", "RAND", "UCHO", "MUS")

SPLITS = ("train", "val", "test", "all")

# Each scheme maps (piano, subset) to "train", "val" or "test".
# enstdk: only the two real Disklaviers. Train on one piano, validate and test on the other,
#   so scores show how well a filter carries over to a piano it has not heard; the test is real music.
# maps-full: the full MAPS release. Seven software pianos train, StbgTGd2 validates, the Disklaviers test.
SCHEMES = {
    "enstdk": lambda piano, subset: (
        "train" if piano == "ENSTDkAm" else "test" if subset == "MUS" else "val"
    ),
    "maps-full": lambda piano, subset: (
        "test" if piano in ("ENSTDkAm", "ENSTDkCl") else "val" if piano == "StbgTGd2" else "train"
    ),
}
DEFAULT_SCHEME = "enstdk"


@dataclass
class MapsFile:
    audio: Path
    labels: Path
    piano: str
    subset: str

    @property
    def stem(self):
        return self.audio.stem


def _piano_and_subset(stem):
    parts = stem.split("_")
    return parts[-1], parts[1].split("-")[0]


def _in_split(piano, subset, split, scheme=DEFAULT_SCHEME):
    return split == "all" or SCHEMES[scheme](piano, subset) == split


def find_files(root=DEFAULT_ROOT, subsets=("UCHO", "RAND", "MUS"), split="all", limit_per_piano=None, seed=0,
               scheme=DEFAULT_SCHEME):
    """
    Return MapsFiles under `root` matching the subsets and split (under split `scheme`).

    `limit_per_piano` keeps a fixed random sample of that many files per piano and subset,
    so quick runs stay comparable between backends.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MAPS not found at {root}. Pass --maps or set SHEETCREATOR_MAPS (see PREPROCESSING.md).")

    groups = {}
    for audio in sorted(root.rglob("*.wav")):
        labels = audio.with_suffix(".txt")
        if not audio.stem.startswith("MAPS_") or not labels.exists():
            continue
        piano, subset = _piano_and_subset(audio.stem)
        if subset in subsets and _in_split(piano, subset, split, scheme):
            groups.setdefault((piano, subset), []).append(MapsFile(audio, labels, piano, subset))

    files = []
    for key in sorted(groups):
        group = groups[key]
        if limit_per_piano and len(group) > limit_per_piano:
            group = sorted(random.Random(seed).sample(group, limit_per_piano), key=lambda f: f.stem)
        files.extend(group)
    return files


def load_labels(path):
    """Ground-truth notes as NoteEvents (amplitude 1.0)."""
    notes = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            onset, offset, midi = float(parts[0]), float(parts[1]), int(float(parts[2]))
        except ValueError:
            continue  # header row
        notes.append(NoteEvent(start=onset, stop=max(offset, onset + 0.001), midi=midi, amplitude=1.0))
    return sorted(notes, key=lambda n: (n.start, n.midi))
