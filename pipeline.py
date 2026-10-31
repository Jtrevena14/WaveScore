"""
Chooses and applies the phantom-note filter, so transcribe.py and the eval
scripts always process notes the same way.

Filter modes:
  none     raw backend output
  default  hand-set FilterSettings (see postprocess.py)
  tuned    FilterSettings found by eval/tune_filters.py
  learned  merge fragments, then the classifier from eval/train_classifier.py
  cnn      merge fragments, then the spectrogram CNN from eval/train_cnn.py (needs the audio)
  auto     the first that exists for the backend: cnn, learned, tuned, else default

make_filter returns apply(notes, audio_path=None); only the cnn mode uses the audio.
"""

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from postprocess import FilterSettings, filter_notes, merge_fragments

MODELS_DIR = Path(__file__).parent / "models"
FILTER_MODES = ("auto", "none", "default", "tuned", "learned", "cnn")


def tuned_settings_path(backend):
    return MODELS_DIR / f"filter_settings_{backend}.json"


def learned_model_path(backend):
    return MODELS_DIR / f"note_filter_{backend}.joblib"


def cnn_model_path(backend):
    return MODELS_DIR / f"note_filter_{backend}.pt"


def default_settings(backend):
    """
    Hand defaults. They were set on Basic Pitch output. The piano model skips the amplitude
    thresholds (its amplitude is velocity, a different scale) and merging (it detects onsets
    itself and leaves small gaps between real re-strikes, which merging would join).
    web/src/app/filters.ts (defaultFilterSettings) mirrors these values; keep them in sync.
    """
    if backend == "piano":
        return replace(FilterSettings(), merge_gap=0, min_amplitude=0, relative_amplitude=0)
    return FilterSettings()


def learned_merge_gap(backend):
    """The learned filter merges fragments first, the same way in training and at inference."""
    return default_settings(backend).merge_gap


def load_settings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {f.name for f in fields(FilterSettings)}
    return FilterSettings(**{k: v for k, v in data["settings"].items() if k in names})


def save_settings(path, settings, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"settings": asdict(settings), **metadata}, indent=2), encoding="utf-8")


def load_learned(path):
    import joblib
    from features import FEATURE_NAMES

    bundle = joblib.load(path)
    if bundle["features"] != FEATURE_NAMES:
        raise ValueError(f"{path} was trained with different features; re-run eval/train_classifier.py")
    return bundle


def resolve_mode(mode, backend):
    """Turn 'auto' into the best filter available for this backend."""
    if mode != "auto":
        return mode
    if cnn_model_path(backend).exists():
        return "cnn"
    if learned_model_path(backend).exists():
        return "learned"
    if tuned_settings_path(backend).exists():
        return "tuned"
    return "default"


def make_filter(mode, backend, settings=None, model_path=None):
    """
    Return (apply, description), where apply(notes) -> filtered notes.

    `settings` overrides the hand settings for 'default'/'tuned'; `model_path` overrides
    the model file for 'learned' and 'cnn'.
    """
    mode = resolve_mode(mode, backend)

    if mode == "none":
        return (lambda notes, audio_path=None: sorted(notes, key=lambda n: (n.start, n.midi))), "none (raw notes)"

    if mode == "cnn":
        from cnn_filter import apply_cnn_filter, load_cnn

        path = Path(model_path) if model_path else cnn_model_path(backend)
        if not path.exists():
            raise FileNotFoundError(f"No CNN filter at {path}; run eval/train_cnn.py first")
        bundle = load_cnn(path)
        if bundle.get("backend") not in (None, backend):
            print(f"Warning: {path.name} was trained on '{bundle['backend']}' output, not '{backend}'")

        def apply(notes, audio_path=None):
            if audio_path is None:
                raise ValueError("The cnn filter needs the audio file")
            merged = merge_fragments(notes, bundle["merge_gap"]) if bundle["merge_gap"] > 0 else list(notes)
            return sorted(apply_cnn_filter(merged, audio_path, bundle), key=lambda n: (n.start, n.midi))

        return apply, f"cnn ({path.name}, threshold {bundle['threshold']:.2f})"

    if mode == "learned":
        from features import apply_learned_filter

        path = Path(model_path) if model_path else learned_model_path(backend)
        bundle = load_learned(path)
        if bundle.get("backend") not in (None, backend):
            print(f"Warning: {path.name} was trained on '{bundle['backend']}' output, not '{backend}'")

        merge_gap = bundle.get("merge_gap", learned_merge_gap(backend))

        def apply(notes, audio_path=None):
            merged = merge_fragments(notes, merge_gap) if merge_gap > 0 else list(notes)
            return sorted(apply_learned_filter(merged, bundle), key=lambda n: (n.start, n.midi))

        return apply, f"learned ({path.name}, threshold {bundle['threshold']:.2f})"

    if settings is None:
        if mode == "tuned":
            path = tuned_settings_path(backend)
            if not path.exists():
                raise FileNotFoundError(f"No tuned settings at {path}; run eval/tune_filters.py first")
            settings = load_settings(path)
        else:
            settings = default_settings(backend)

    return (lambda notes, audio_path=None: filter_notes(notes, settings)[0]), f"{mode} {asdict(settings)}"
