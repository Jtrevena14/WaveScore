"""
CNN phantom-note filter: looks at a constant-Q spectrogram patch around each candidate note,
plus the per-note features from features.py, and scores how likely the note was really played.

Shared by eval/train_cnn.py (training) and pipeline.py (inference), so both cut exactly the
same patches. The patch is pitch-relative: row 0 is always 13 semitones below the note, so the
same filters see "an octave above", "a fifth above" etc. whatever the note is.
"""

import numpy as np

from features import FEATURE_NAMES, note_features

SAMPLE_RATE = 22050
HOP = 256                      # about 11.6 ms per frame
BINS_PER_SEMITONE = 3
LOWEST_MIDI = 21               # A0, CQT bin 0
CQT_SEMITONES = 100            # A0 up to about C9
DB_RANGE = 80.0                # log magnitude is mapped from [-80 dB, 0 dB] to [0, 1]

BELOW, ABOVE = 13, 29          # semitones of context below/above the note (covers 12/19/24/28 overtones)
BEFORE, AFTER = 8, 24          # frames of context before/after the onset
PITCH_BINS = (BELOW + ABOVE) * BINS_PER_SEMITONE + 3   # 129
FRAMES = BEFORE + AFTER                                 # 32

# Training cuts patches this much larger so it can jitter them; inference takes the centre.
PITCH_JITTER, TIME_JITTER = 1, 2

PATCH_CONFIG = {
    "sample_rate": SAMPLE_RATE, "hop": HOP, "bins_per_semitone": BINS_PER_SEMITONE,
    "below": BELOW, "above": ABOVE, "before": BEFORE, "after": AFTER,
}


def compute_cqt(audio_path):
    """Normalised log-magnitude CQT, shape (CQT_SEMITONES * BINS_PER_SEMITONE, frames), values in [0, 1]."""
    import librosa

    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    cqt = np.abs(librosa.cqt(
        audio, sr=SAMPLE_RATE, hop_length=HOP, fmin=librosa.midi_to_hz(LOWEST_MIDI),
        n_bins=CQT_SEMITONES * BINS_PER_SEMITONE, bins_per_octave=12 * BINS_PER_SEMITONE,
    ))
    db = librosa.amplitude_to_db(cqt, ref=np.max(cqt) if cqt.size and cqt.max() > 0 else 1.0, top_db=DB_RANGE)
    return ((db + DB_RANGE) / DB_RANGE).astype(np.float32)


def cut_patches(cqt, notes, pitch_margin=PITCH_JITTER, time_margin=TIME_JITTER):
    """
    One patch per note, shape (n, PITCH_BINS + 2*pitch_margin, FRAMES + 2*time_margin), float16.
    Anything outside the spectrogram is 0 (silence).
    """
    height = PITCH_BINS + 2 * pitch_margin
    width = FRAMES + 2 * time_margin
    padded = np.pad(cqt, ((height, height), (width, width)))
    out = np.zeros((len(notes), height, width), dtype=np.float16)
    for i, note in enumerate(notes):
        # The middle bin of a semitone sits on the note, so one extra bin each side of the range.
        row = (note.midi - LOWEST_MIDI - BELOW) * BINS_PER_SEMITONE - 1 - pitch_margin
        col = int(round(note.start * SAMPLE_RATE / HOP)) - BEFORE - time_margin
        out[i] = padded[row + height:row + 2 * height, col + width:col + 2 * width]
    return out


def centre(patches):
    """Crop cached (jitter-margin) patches to the size the model takes."""
    return patches[:, PITCH_JITTER:PITCH_JITTER + PITCH_BINS, TIME_JITTER:TIME_JITTER + FRAMES]


def build_model():
    import torch
    from torch import nn

    def block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    class NoteCNN(nn.Module):
        def __init__(self, n_features=len(FEATURE_NAMES)):
            super().__init__()
            self.conv = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(
                nn.Linear(64 + n_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1),
            )

        def forward(self, patches, features):
            """patches (B, PITCH_BINS, FRAMES), features (B, n_features) standardised -> logits (B,)"""
            x = self.conv(patches.unsqueeze(1)).flatten(1)
            return self.head(torch.cat([x, features], dim=1)).squeeze(1)

    return NoteCNN()


def load_cnn(path):
    import torch

    bundle = torch.load(path, map_location="cpu")
    if bundle["features"] != FEATURE_NAMES or bundle["patch"] != PATCH_CONFIG:
        raise ValueError(f"{path} was trained with different features or patches; re-run eval/train_cnn.py")
    model = build_model()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    bundle["model"] = model
    return bundle


def score_notes(model, patches, features, mean, std, device="cpu", batch_size=512):
    """Probability that each note is real. `patches` are model-sized (already centred)."""
    import torch

    feats = ((features - np.asarray(mean)) / np.asarray(std)).astype(np.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            p = torch.from_numpy(np.asarray(patches[i:i + batch_size], dtype=np.float32)).to(device)
            f = torch.from_numpy(feats[i:i + batch_size]).to(device)
            scores.append(torch.sigmoid(model(p, f)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def apply_cnn_filter(notes, audio_path, bundle):
    """Keep the notes the CNN scores at or above its threshold."""
    if not notes:
        return notes
    patches = centre(cut_patches(compute_cqt(audio_path), notes))
    scores = score_notes(bundle["model"], patches, note_features(notes), bundle["mean"], bundle["std"])
    return [n for n, s in zip(notes, scores) if s >= bundle["threshold"]]
