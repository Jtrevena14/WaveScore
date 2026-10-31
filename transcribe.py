#!/usr/bin/env python3
"""
Transcribe an audio file to notes (pitch, start time, stop time) and chord events.

Backends: Spotify Basic Pitch (any instrument) or ByteDance piano transcription.
Phantom notes are removed by the filter chosen in pipeline.py.
"""

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import pretty_midi

from backends import BACKENDS, transcribe
from chords import find_chords, is_chord
from pipeline import FILTER_MODES, default_settings, load_settings, make_filter, tuned_settings_path
from postprocess import FilterSettings

# CLI flag -> FilterSettings field, for the hand-filter overrides.
SETTING_FLAGS = {f.name: "--" + f.name.replace("_", "-") for f in fields(FilterSettings)}


def midi_to_note_name(midi_num):
    """Convert MIDI note number to note name (e.g., 60 -> C4)."""
    return pretty_midi.note_number_to_name(midi_num)


def transcribe_audio(audio_path, backend="basic-pitch", filter_mode="auto", settings=None,
                     model_path=None, json_path=None, chords_json_path=None):
    """
    Transcribe an audio file and print detected notes and chords.

    Args:
        audio_path: Path to .wav or .mp3 file
        backend: one of backends.BACKENDS
        filter_mode: one of pipeline.FILTER_MODES
        settings: optional FilterSettings for the hand filter (default/tuned modes)
        model_path: optional model file for the learned or cnn filter
        json_path: optional path to also write the notes as JSON (read by the visualizer)
        chords_json_path: optional path to also write the chord events as JSON
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

    apply_filter, filter_description = make_filter(filter_mode, backend, settings, model_path)

    print(f"Transcribing: {audio_path} (model: {backend})")
    print("Processing...")

    raw = transcribe(audio_path, backend)
    events = apply_filter(raw, audio_path)

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

    print(f"\nTotal notes detected: {len(notes)} (raw from model: {len(raw)})")
    print(f"Filter: {filter_description}")

    chord_events = find_chords(events)
    print(f"\nChords ({sum(is_chord(c) for c in chord_events)} chords, {len(chord_events)} onset groups):")
    for chord in chord_events:
        pitches = " ".join(midi_to_note_name(n.midi) for n in chord.notes)
        print(f"  {chord.start:7.3f} s  {chord.name:<10} {pitches}")

    if json_path:
        # The model is recorded so the visualizer can pick that model's filter defaults.
        Path(json_path).write_text(json.dumps({"model": backend, "notes": notes}, indent=2), encoding="utf-8")
        print(f"\nWrote {len(notes)} notes to {json_path}")

    if chords_json_path:
        data = [
            {"start": round(c.start, 3), "name": c.name, "midi": [n.midi for n in c.notes],
             "pitches": [midi_to_note_name(n.midi) for n in c.notes]}
            for c in chord_events
        ]
        Path(chords_json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {len(data)} chord events to {chords_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file (.wav or .mp3) and output detected notes and chords"
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
    parser.add_argument("--chords-json", metavar="OUT.json",
                        help="Also write the chord events to this JSON file")
    parser.add_argument("--model", choices=BACKENDS, default="basic-pitch",
                        help="Transcription model: basic-pitch (any instrument) or piano (solo piano)")

    filters = parser.add_argument_group(
        "post-filters",
        "--filter auto uses a trained CNN for the model if one exists in models/, "
        "then a trained classifier, then tuned settings, then the hand defaults.")
    filters.add_argument("--filter", choices=FILTER_MODES, default="auto",
                         help="How to remove phantom notes (default auto)")
    filters.add_argument("--no-filter", action="store_true", help="Same as --filter none")
    filters.add_argument("--learned-filter", metavar="MODEL.joblib",
                         help="Use this trained classifier (implies --filter learned)")
    filters.add_argument("--cnn-filter", metavar="MODEL.pt",
                         help="Use this trained CNN (implies --filter cnn)")
    filters.add_argument("--filter-settings", metavar="SETTINGS.json",
                         help="Use hand-filter settings from this file (from eval/tune_filters.py)")

    defaults = FilterSettings()
    overrides = parser.add_argument_group(
        "hand-filter overrides",
        "Each one overrides a single hand-filter setting (0 turns that filter off). "
        "Giving any of these switches to the hand filter.")
    for name, flag in SETTING_FLAGS.items():
        overrides.add_argument(flag, dest=name, type=float, default=None, metavar="X",
                               help=f"(hand default {getattr(defaults, name)})")

    args = parser.parse_args()

    mode = "none" if args.no_filter else args.filter
    if args.learned_filter and args.cnn_filter:
        parser.error("choose one of --learned-filter and --cnn-filter")
    if args.learned_filter:
        mode = "learned"
    if args.cnn_filter:
        mode = "cnn"

    settings = None
    given = {name: getattr(args, name) for name in SETTING_FLAGS if getattr(args, name) is not None}
    if args.filter_settings or given:
        if mode in ("learned", "cnn"):
            parser.error(f"hand-filter settings can't be combined with the {mode} filter")
        if args.filter_settings:
            settings = load_settings(args.filter_settings)
        elif mode in ("auto", "tuned") and tuned_settings_path(args.model).exists():
            settings = load_settings(tuned_settings_path(args.model))
        else:
            settings = default_settings(args.model)
        settings = replace(settings, **given)
        if mode == "auto":
            mode = "default"

    try:
        transcribe_audio(args.audio_file, args.model, mode, settings, args.learned_filter or args.cnn_filter,
                         args.json_path, args.chords_json)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
