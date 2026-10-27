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


def midi_to_note_name(midi_num):
    """Convert MIDI note number to note name (e.g., 60 -> C4)."""
    return pretty_midi.note_number_to_name(midi_num)


def transcribe_audio(audio_path, json_path=None):
    """
    Transcribe an audio file and print detected notes.

    Args:
        audio_path: Path to .wav or .mp3 file
        json_path: Optional path to also write the notes as JSON
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


    # Sort by start time
    note_events = sorted(note_events, key=lambda x: x[0])

    # Print results
    print("\nDetected notes:")
    print("Pitch\t\tStart (s)\tStop (s)")
    print("-" * 50)

    notes = []
    for start_time, end_time, midi_pitch, amplitude, pitch_shift in note_events:
        pitch_name = midi_to_note_name(int(midi_pitch))
        print(f"{pitch_name}\t\t{start_time:.3f}\t\t{end_time:.3f}")
        notes.append({
            "pitch": pitch_name,
            "midi": int(midi_pitch),
            "start": round(float(start_time), 3),
            "stop": round(float(end_time), 3),
        })

    print(f"\nTotal notes detected: {len(note_events)}")

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

    args = parser.parse_args()

    try:
        transcribe_audio(args.audio_file, args.json_path)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
