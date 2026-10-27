# Plan: Audio → notes script (pitch, start_time, stop_time)

## Context
Starting fresh on SheetCreator. The only goal for now: take a .wav or .mp3 and print the notes played, each with pitch, start time and stop time. No sheet-music rendering yet.

## Library choice: Spotify Basic Pitch (`basic-pitch`)
- Polyphonic automatic music transcription (handles chords/multiple instruments), accepts .wav and .mp3 directly.
- `predict()` returns `note_events`: tuples of `(start_time_s, end_time_s, midi_pitch, amplitude, pitch_bends)`, which maps straight onto the requested output.
- Alternatives considered: `librosa.pyin` / CREPE are monophonic only, so they fail on chords. Basic Pitch is the best fit.
- Caveat: on Windows, use Python 3.10 or 3.11 if the default TensorFlow backend gives install trouble. Recent `basic-pitch` versions also bundle an ONNX/CoreML runtime fallback.

## Implementation
Single file: `C:\Users\Jesse Trevena\Desktop\SheetCreator\transcribe.py`
1. `argparse` takes one positional arg: the audio file path (.wav/.mp3). Validate that the file exists and the extension is supported.
2. Call `basic_pitch.inference.predict(path)`, which returns `(model_output, midi_data, note_events)`.
3. Convert each MIDI number to a note name (e.g. 60 → `C4`) with a small helper (`pretty_midi.note_number_to_name`, already a Basic Pitch dependency).
4. Sort by start time and print one line per note: `pitch  start_time  stop_time` (seconds, 3 decimals). Example: `C4  0.512  0.934`.
5. Include a `requirements.txt` containing `basic-pitch`.

## Verification
1. `pip install -r requirements.txt`
2. Run `python transcribe.py sample.wav` on a short known clip (e.g. a few piano notes or a C major scale) and check that the printed pitches and times match what was played.
3. Run it on an .mp3 to confirm both formats work.
