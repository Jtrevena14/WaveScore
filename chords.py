"""
Group notes into chord events by onset and give each a best-guess chord name.
"""

from dataclasses import dataclass

import pretty_midi

# Gap between consecutive onsets that still counts as the same chord. Human-played chords
# spread over 100+ ms (2-5-1.mp3's G chord spans 120 ms), so this is looser than MAPS needs.
DEFAULT_ONSET_GAP = 0.12
# A chord event never spans more than this from its first to its last onset.
DEFAULT_MAX_SPAN = 0.25

PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Intervals above the root, as pitch classes. Extensions are folded into one octave (9 = 2, 13 = 9).
CHORD_TEMPLATES = {
    "": {0, 4, 7},
    "m": {0, 3, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
    "6": {0, 4, 7, 9},
    "m6": {0, 3, 7, 9},
    "7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "m7": {0, 3, 7, 10},
    "m7b5": {0, 3, 6, 10},
    "dim7": {0, 3, 6, 9},
    "mMaj7": {0, 3, 7, 11},
    "9": {0, 4, 7, 10, 2},
    "maj9": {0, 4, 7, 11, 2},
    "m9": {0, 3, 7, 10, 2},
    "13": {0, 4, 7, 10, 2, 9},
}


@dataclass
class ChordEvent:
    start: float
    notes: list  # NoteEvents, sorted low to high
    name: str


def group_by_onset(notes, onset_gap=DEFAULT_ONSET_GAP, max_span=DEFAULT_MAX_SPAN):
    """Split notes into groups whose onsets are chained within `onset_gap` seconds."""
    groups = []
    for note in sorted(notes, key=lambda n: (n.start, n.midi)):
        if groups:
            group = groups[-1]
            if note.start - group[-1].start <= onset_gap and note.start - group[0].start <= max_span:
                group.append(note)
                continue
        groups.append([note])
    return groups


def name_chord(midis):
    """
    Best-guess chord name for a set of MIDI notes, e.g. [50, 65, 72, 76] -> 'Dm9'.

    Tries every root and template, rewarding notes the template explains and penalising
    notes it doesn't. Missing chord tones cost little, so rootless or shell voicings still
    match (D F E -> Dm9). The bass note wins ties.
    """
    pitch_classes = {m % 12 for m in midis}
    if len(pitch_classes) == 1:
        return pretty_midi.note_number_to_name(min(midis))
    if len(pitch_classes) == 2:
        low, high = sorted(midis)[0], sorted(midis)[-1]
        return f"{pretty_midi.note_number_to_name(low)}+{pretty_midi.note_number_to_name(high)}"

    bass = min(midis) % 12
    best = (float("-inf"), 0, "")
    for root in range(12):
        relative = {(pc - root) % 12 for pc in pitch_classes}
        for suffix, template in CHORD_TEMPLATES.items():
            matched = len(relative & template)
            extra = len(relative - template)
            missing = len(template - relative)
            score = 2 * matched - 3 * extra - 0.5 * missing + (1 if root == bass else 0)
            if score > best[0]:
                best = (score, root, suffix)

    _, root, suffix = best
    name = PITCH_CLASS_NAMES[root] + suffix
    return name if root == bass else f"{name}/{PITCH_CLASS_NAMES[bass]}"


def find_chords(notes, onset_gap=DEFAULT_ONSET_GAP, max_span=DEFAULT_MAX_SPAN):
    """Return one ChordEvent per onset group."""
    events = []
    for group in group_by_onset(notes, onset_gap, max_span):
        group = sorted(group, key=lambda n: n.midi)
        events.append(ChordEvent(
            start=min(n.start for n in group),
            notes=group,
            name=name_chord([n.midi for n in group]),
        ))
    return events


def is_chord(event):
    """True for events with at least three distinct pitch classes."""
    return len({n.midi % 12 for n in event.notes}) >= 3
