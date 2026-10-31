"""
Per-note features for the learned phantom-note filter.

Shared by eval/train_classifier.py (training) and transcribe.py (inference),
so both always compute exactly the same columns.
"""

import math

import numpy as np

from postprocess import HARMONIC_INTERVALS

SAME_ONSET = 0.05  # seconds; onsets this close count as struck together
RATIO_CAP = 5.0

FEATURE_NAMES = [
    "amplitude",
    "onset_strength",
    "log_duration",
    "pitch",                    # 0 = A0, 1 = C8
    "share_of_loudest_overlap",  # amplitude / loudest note sounding with it (1 = it is the loudest)
    "share_of_loudest_onset",   # same, for notes struck within SAME_ONSET
    "rank_in_onset_group",      # 0 = loudest in its group, 1 = quietest
    "onset_group_size",
    "harmonic_parent_ratio",    # loudest note 12/19/24/28 below, struck together, / this note (0 = none)
    "louder_neighbour_ratio",   # loudest overlapping note 1 semitone away / this note (0 = none)
    "octave_above_ratio",       # note 12 above struck together / this note (catches sub-octave leakage)
    "gap_since_same_pitch",     # seconds since the same pitch last stopped (capped at 2)
    "overlapping_count",
]


def _ratio(numerator, denominator):
    if denominator <= 0:
        return RATIO_CAP
    return min(numerator / denominator, RATIO_CAP)


def _max_or_zero(values):
    return float(values.max()) if values.size else 0.0


def note_features(notes):
    """Return an (n_notes, len(FEATURE_NAMES)) float array, rows in the order of `notes`."""
    starts = np.array([n.start for n in notes])
    stops = np.array([n.stop for n in notes])
    midis = np.array([n.midi for n in notes])
    amps = np.array([n.amplitude for n in notes])
    harmonic = np.array(HARMONIC_INTERVALS)

    rows = []
    for i, note in enumerate(notes):
        others = np.arange(len(notes)) != i
        overlapping = others & (starts < note.stop) & (stops > note.start)
        together = others & (np.abs(starts - note.start) <= SAME_ONSET)
        group_size = int(together.sum()) + 1

        parents = together & np.isin(note.midi - midis, harmonic)
        neighbours = overlapping & (np.abs(midis - note.midi) == 1)
        octave_above = together & (midis - note.midi == 12)

        earlier_same = others & (midis == note.midi) & (stops <= note.start + 1e-6)
        gap = note.start - stops[earlier_same].max() if earlier_same.any() else 2.0

        loudest_overlap = _max_or_zero(amps[overlapping])
        loudest_together = _max_or_zero(amps[together])
        louder_in_group = int((amps[together] > note.amplitude).sum())

        rows.append([
            note.amplitude,
            note.onset_strength if note.onset_strength is not None else note.amplitude,
            math.log(max(note.duration, 1e-3)),
            (note.midi - 21) / 87,
            note.amplitude / max(note.amplitude, loudest_overlap, 1e-6),
            note.amplitude / max(note.amplitude, loudest_together, 1e-6),
            louder_in_group / max(group_size - 1, 1),
            group_size,
            _ratio(amps[parents].max(), note.amplitude) if parents.any() else 0.0,
            _ratio(amps[neighbours].max(), note.amplitude) if neighbours.any() else 0.0,
            _ratio(amps[octave_above].max(), note.amplitude) if octave_above.any() else 0.0,
            min(gap, 2.0),
            int(overlapping.sum()),
        ])
    return np.array(rows, dtype=float).reshape(len(notes), len(FEATURE_NAMES))


def apply_learned_filter(notes, bundle):
    """Keep the notes the trained classifier scores at or above its threshold."""
    if not notes:
        return notes
    scores = bundle["model"].predict_proba(note_features(notes))[:, 1]
    return [n for n, s in zip(notes, scores) if s >= bundle["threshold"]]
