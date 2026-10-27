#!/usr/bin/env python3
"""
Simple audio transcription script using Spotify's Basic Pitch library.
Outputs detected notes with pitch, start time, and stop time.
"""

import argparse
import json
import sys
from pathlib import Path

import basic_pitch.inference
import pretty_midi

from postprocess import FilterSettings, NoteEvent, filter_notes


def midi_to_note_name(midi_num):
    """Convert MIDI note number to note name (e.g., 60 -> C4)."""
    return pretty_midi.note_number_to_name(midi_num)


def transcribe_audio(audio_path, json_path=None, filter_settings=None):
    """
    Transcribe an audio file and print detected notes.

    Args:
        audio_path: Path to .wav or .mp3 file
        json_path: Optional path to also write the notes as JSON
        filter_settings: Optional FilterSettings; None skips post-filtering
    """
    audio_path = Path(audio_path)

    # Validate file exists
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Validate file extension
    supported_formats = {'.wav', '.mp3'}
    if audio_path.suffix.lower() not in supported_formats:
        print(f"Error: Unsupported file format '{audio_path.suffix}'. Supported: {supported_formats}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribing: {audio_path}")
    print("Processing...")

    # Run inference
    model_output, midi_data, note_events = basic_pitch.inference.predict(str(audio_path))

    events = [
        NoteEvent(start=float(start), stop=float(end), midi=int(midi), amplitude=float(amp))
        for start, end, midi, amp, _pitch_bends in note_events
    ]
    raw_count = len(events)

    stats = {}
    if filter_settings is not None:
        events, stats = filter_notes(events, filter_settings)
    else:
        events = sorted(events, key=lambda n: n.start)

    # Print results
    print("\nDetected notes:")
    print("Pitch\t\tStart (s)\tStop (s)\tAmplitude")
    print("-" * 60)

    notes = []
    for event in events:
        pitch_name = midi_to_note_name(event.midi)
        print(f"{pitch_name}\t\t{event.start:.3f}\t\t{event.stop:.3f}\t\t{event.amplitude:.3f}")
        notes.append({
            "pitch": pitch_name,
            "midi": event.midi,
            "start": round(event.start, 3),
            "stop": round(event.stop, 3),
            "amplitude": round(event.amplitude, 3),
        })

    print(f"\nTotal notes detected: {len(notes)}")
    if filter_settings is not None:
        print(f"Raw notes from Basic Pitch: {raw_count}")
        for name, removed in stats.items():
            print(f"  removed {removed:>4}  {name}")

    if json_path:
        Path(json_path).write_text(json.dumps(notes, indent=2), encoding="utf-8")
        print(f"Wrote {len(notes)} notes to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file (.wav or .mp3) and output detected notes"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (.wav or .mp3)"
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="OUT.json",
        help="Also write the detected notes to this JSON file"
    )

    defaults = FilterSettings()
    filters = parser.add_argument_group("post-filters", "Set any value to 0 to turn that filter off.")
    filters.add_argument("--no-filter", action="store_true",
                         help="Skip all post-filters and print Basic Pitch's raw notes")
    filters.add_argument("--merge-gap", type=float, default=defaults.merge_gap, metavar="SEC",
                         help=f"Join same-pitch notes separated by at most this gap (default {defaults.merge_gap})")
    filters.add_argument("--min-duration", type=float, default=defaults.min_duration, metavar="SEC",
                         help=f"Drop notes shorter than this (default {defaults.min_duration})")
    filters.add_argument("--min-amplitude", type=float, default=defaults.min_amplitude, metavar="AMP",
                         help=f"Drop notes with amplitude below this, 0-1 (default {defaults.min_amplitude})")
    filters.add_argument("--relative-amplitude", type=float, default=defaults.relative_amplitude, metavar="RATIO",
                         help="Drop notes quieter than RATIO x an overlapping note "
                              f"(default {defaults.relative_amplitude})")
    filters.add_argument("--harmonic-ratio", type=float, default=defaults.harmonic_ratio, metavar="RATIO",
                         help="Drop notes 12/19/24/28 semitones above a note starting at the same time "
                              f"and quieter than RATIO x it (default {defaults.harmonic_ratio})")
    filters.add_argument("--onset-tolerance", type=float, default=defaults.onset_tolerance, metavar="SEC",
                         help=f"Max onset difference for 'same time' (default {defaults.onset_tolerance})")

    args = parser.parse_args()

    filter_settings = None if args.no_filter else FilterSettings(
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
        min_amplitude=args.min_amplitude,
        relative_amplitude=args.relative_amplitude,
        harmonic_ratio=args.harmonic_ratio,
        onset_tolerance=args.onset_tolerance,
    )

    try:
        transcribe_audio(args.audio_file, args.json_path, filter_settings)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
