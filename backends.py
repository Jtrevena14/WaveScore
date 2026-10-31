"""
Transcription backends. Each turns an audio file into raw NoteEvents.

- basic-pitch: Spotify Basic Pitch, any instrument.
- piano: ByteDance high-resolution piano transcription (solo piano only).
"""

from pathlib import Path

import numpy as np

from postprocess import NoteEvent

BACKENDS = ("basic-pitch", "piano")

PIANO_CHECKPOINT = Path(__file__).parent / "models" / "piano_note_pedal.pth"
PIANO_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)

LOWEST_MIDI = 21  # A0, the first row of both models' 88-key outputs

_piano_model = None


def transcribe(audio_path, backend):
    """Return the backend's raw notes for `audio_path`, sorted by start time."""
    if backend == "basic-pitch":
        notes = _basic_pitch(audio_path)
    elif backend == "piano":
        notes = _piano(audio_path)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {BACKENDS}")
    return sorted(notes, key=lambda n: (n.start, n.midi))


def _onset_strength(onsets, frames_per_second, start, midi):
    """Peak onset activation within two frames of `start` for this pitch."""
    frame = int(round(start * frames_per_second))
    lo, hi = max(0, frame - 2), min(len(onsets), frame + 3)
    row = midi - LOWEST_MIDI
    if lo >= hi or not 0 <= row < onsets.shape[1]:
        return 0.0
    return float(np.max(onsets[lo:hi, row]))


def _basic_pitch(audio_path):
    import basic_pitch.inference
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

    model_output, _midi_data, note_events = basic_pitch.inference.predict(str(audio_path))
    onsets = model_output["onset"]
    fps = AUDIO_SAMPLE_RATE / FFT_HOP

    return [
        NoteEvent(
            start=float(start),
            stop=float(end),
            midi=int(midi),
            amplitude=float(amp),
            onset_strength=_onset_strength(onsets, fps, float(start), int(midi)),
        )
        for start, end, midi, amp, _pitch_bends in note_events
    ]


def _piano(audio_path):
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    global _piano_model
    if _piano_model is None:
        if not PIANO_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Piano model checkpoint not found at {PIANO_CHECKPOINT}.\n"
                f"Download it with:\n  curl -L -o \"{PIANO_CHECKPOINT}\" \"{PIANO_CHECKPOINT_URL}\""
            )
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _piano_model = PianoTranscription(device=device, checkpoint_path=str(PIANO_CHECKPOINT))

    # librosa reads mp3 without ffmpeg; the package's own loader needs ffmpeg.
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    result = _piano_model.transcribe(audio, None)
    onsets = result["output_dict"]["reg_onset_output"]
    fps = _piano_model.frames_per_second

    notes = []
    for event in result["est_note_events"]:
        start, stop, midi = float(event["onset_time"]), float(event["offset_time"]), int(event["midi_note"])
        notes.append(NoteEvent(
            start=start,
            stop=max(stop, start + 0.001),
            midi=midi,
            amplitude=event["velocity"] / 127,
            onset_strength=_onset_strength(onsets, fps, start, midi),
        ))
    return notes
