"""
Post-filters for Basic Pitch note events, aimed at removing phantom notes.
See PREPROCESSING.md section 2.3 for the reasoning behind each filter.

web/src/app/filters.ts is a TypeScript port used for live tuning in the
visualizer. Keep the two in sync: same settings, defaults and step order.
"""

from dataclasses import dataclass, replace
from typing import Optional

# Semitone offsets of the strongest overtones above a fundamental:
# octave, octave + fifth, two octaves, two octaves + major third.
HARMONIC_INTERVALS = (12, 19, 24, 28)

# Gaps at or below this many seconds count as touching, not as a fragment gap.
TOUCH_EPSILON = 0.001


@dataclass
class NoteEvent:
    start: float
    stop: float
    midi: int
    amplitude: float
    # Model's onset activation at the note's start (0-1), if the backend provides one.
    onset_strength: Optional[float] = None

    @property
    def duration(self):
        return self.stop - self.start


@dataclass
class FilterSettings:
    """Tunable thresholds. Setting any value to 0 turns that filter off."""
    merge_gap: float = 0.03           # seconds; join same-pitch notes closer than this
    min_duration: float = 0.06        # seconds; drop notes shorter than this
    min_amplitude: float = 0.35       # drop notes quieter than this (Basic Pitch amplitude is 0-1)
    relative_amplitude: float = 0.25  # drop notes quieter than this fraction of an overlapping note
    harmonic_ratio: float = 0.8       # drop overtone notes quieter than this fraction of their fundamental
    onset_tolerance: float = 0.05     # seconds; how close onsets must be to count as "same time"


def merge_fragments(notes, gap):
    """
    Join same-pitch notes separated by a gap of at most `gap` seconds.

    Notes that touch exactly (gap 0) are left alone: Basic Pitch ends a note
    right where it detects a new onset, so those are real re-strikes.
    """
    merged = []
    by_pitch = {}
    for note in sorted(notes, key=lambda n: (n.midi, n.start)):
        by_pitch.setdefault(note.midi, []).append(note)

    for pitch_notes in by_pitch.values():
        current = pitch_notes[0]
        for note in pitch_notes[1:]:
            if TOUCH_EPSILON < note.start - current.stop <= gap:
                # Duration-weighted amplitude so a short blip doesn't dominate.
                total = current.duration + note.duration
                amplitude = (
                    (current.amplitude * current.duration + note.amplitude * note.duration) / total
                    if total > 0 else max(current.amplitude, note.amplitude)
                )
                current = replace(current, stop=max(current.stop, note.stop), amplitude=amplitude)
            else:
                merged.append(current)
                current = note
        merged.append(current)

    return sorted(merged, key=lambda n: n.start)


def drop_relatively_quiet(notes, ratio):
    """Drop notes much quieter than another note sounding at the same time."""
    notes = sorted(notes, key=lambda n: n.start)
    kept = []
    for note in notes:
        quiet = False
        for other in notes:
            if other.start >= note.stop:
                break
            if other is not note and other.stop > note.start and note.amplitude < ratio * other.amplitude:
                quiet = True
                break
        if not quiet:
            kept.append(note)
    return kept


def suppress_harmonics(notes, ratio, onset_tolerance):
    """
    Drop a note that sits a harmonic interval above a louder note starting
    at about the same time, since it is likely that note's overtone.
    """
    notes = sorted(notes, key=lambda n: n.start)
    kept = []
    for note in notes:
        overtone = False
        for other in notes:
            if other.start > note.start + onset_tolerance:
                break
            if (
                abs(other.start - note.start) <= onset_tolerance
                and note.midi - other.midi in HARMONIC_INTERVALS
                and note.amplitude < ratio * other.amplitude
            ):
                overtone = True
                break
        if not overtone:
            kept.append(note)
    return kept


def filter_notes(notes, settings):
    """
    Run all enabled filters in order and return (filtered_notes, stats),
    where stats maps each filter name to how many notes it removed.
    """
    stats = {}

    def step(name, enabled, func):
        nonlocal notes
        if not enabled:
            return
        before = len(notes)
        notes = func(notes)
        stats[name] = before - len(notes)

    step("merged fragments", settings.merge_gap > 0,
         lambda ns: merge_fragments(ns, settings.merge_gap))
    step("too short", settings.min_duration > 0,
         lambda ns: [n for n in ns if n.duration >= settings.min_duration])
    step("too quiet", settings.min_amplitude > 0,
         lambda ns: [n for n in ns if n.amplitude >= settings.min_amplitude])
    step("quieter than a simultaneous note", settings.relative_amplitude > 0,
         lambda ns: drop_relatively_quiet(ns, settings.relative_amplitude))
    step("likely overtones", settings.harmonic_ratio > 0,
         lambda ns: suppress_harmonics(ns, settings.harmonic_ratio, settings.onset_tolerance))

    return sorted(notes, key=lambda n: (n.start, n.midi)), stats
