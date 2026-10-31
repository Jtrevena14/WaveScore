"""
Caches each backend's raw notes per audio file, so tuning and training never re-run a model.
"""

import json
from dataclasses import asdict
from pathlib import Path

from backends import transcribe
from postprocess import NoteEvent

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def cached_notes(audio_path, backend, cache_dir=DEFAULT_CACHE):
    """Raw notes for `audio_path`, from the cache if present, else from the model (then cached)."""
    path = Path(cache_dir) / backend / f"{Path(audio_path).stem}.json"
    if path.exists():
        return [NoteEvent(**n) for n in json.loads(path.read_text(encoding="utf-8"))]

    notes = transcribe(audio_path, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(n) for n in notes]), encoding="utf-8")
    return notes


def load_dataset(files, backend, cache_dir=DEFAULT_CACHE, verbose=True):
    """Return [(MapsFile, raw_notes, reference_notes)], transcribing uncached files with progress."""
    from eval.maps import load_labels

    data = []
    for i, f in enumerate(files, 1):
        cached = (Path(cache_dir) / backend / f"{f.stem}.json").exists()
        if verbose and not cached:
            print(f"  [{i}/{len(files)}] transcribing {f.stem} with {backend}", flush=True)
        data.append((f, cached_notes(f.audio, backend, cache_dir), load_labels(f.labels)))
    return data
