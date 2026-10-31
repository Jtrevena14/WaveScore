"""
Note-level and chord-level scores against ground truth.

Precision is the phantom-note number: of the notes we output, how many were really played.
Recall is the missed-note number: of the notes played, how many we found.
"""

from dataclasses import dataclass

import mir_eval
import numpy as np
import pretty_midi

from chords import group_by_onset

ONSET_TOLERANCE = 0.05  # seconds, the MIREX standard
OFFSET_RATIO = 0.2      # for the stricter onset+offset score


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def _to_mir(notes):
    intervals = np.array([[n.start, n.stop] for n in notes], dtype=float).reshape(-1, 2)
    pitches = np.array([pretty_midi.note_number_to_hz(n.midi) for n in notes], dtype=float)
    return intervals, pitches


def match(predicted, reference, offset_ratio=None):
    """Return [(ref_index, pred_index)] pairs, matched by pitch and onset (and offset if offset_ratio)."""
    if not predicted or not reference:
        return []
    ref_i, ref_p = _to_mir(reference)
    est_i, est_p = _to_mir(predicted)
    return mir_eval.transcription.match_notes(
        ref_i, ref_p, est_i, est_p, onset_tolerance=ONSET_TOLERANCE, offset_ratio=offset_ratio)


def note_counts(predicted, reference, offset_ratio=None):
    tp = len(match(predicted, reference, offset_ratio))
    return Counts(tp=tp, fp=len(predicted) - tp, fn=len(reference) - tp)


def correct_labels(predicted, reference):
    """1 for each predicted note that matches a real note, 0 for phantoms. Used as training labels."""
    labels = np.zeros(len(predicted), dtype=int)
    for _ref, pred in match(predicted, reference):
        labels[pred] = 1
    return labels


@dataclass
class ChordScore:
    reference_chords: int = 0
    exact_chords: int = 0
    count_error: int = 0  # sum over files of |predicted onset groups - reference onset groups|
    files: int = 0

    def add(self, predicted, reference):
        ref_groups = [g for g in group_by_onset(reference) if len(g) >= 2]
        pred_groups = group_by_onset(predicted)
        pred_index = [(min(n.start for n in g), {n.midi for n in g}) for g in pred_groups]
        for group in ref_groups:
            start = min(n.start for n in group)
            pitches = {n.midi for n in group}
            if any(abs(s - start) <= ONSET_TOLERANCE * 2 and p == pitches for s, p in pred_index):
                self.exact_chords += 1
        self.reference_chords += len(ref_groups)
        self.count_error += abs(len(pred_groups) - len(group_by_onset(reference)))
        self.files += 1

    @property
    def exact_rate(self):
        return self.exact_chords / self.reference_chords if self.reference_chords else float("nan")

    @property
    def mean_count_error(self):
        return self.count_error / self.files if self.files else float("nan")
