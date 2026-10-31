import { Note } from './note';

/*
 * TypeScript port of postprocess.py so the filters can be tuned live in the browser.
 * Keep the two in sync: same settings, same defaults, same steps in the same order.
 */

/** Semitone offsets of the strongest overtones: octave, octave + fifth, two octaves, two octaves + major third. */
const HARMONIC_INTERVALS = new Set([12, 19, 24, 28]);

/** Gaps at or below this many seconds count as touching (a re-strike), not as a fragment gap. */
const TOUCH_EPSILON = 0.001;

/** Setting any value to 0 turns that filter off. Names match the transcribe.py flags. */
export interface FilterSettings {
  mergeGap: number;
  minDuration: number;
  minAmplitude: number;
  relativeAmplitude: number;
  harmonicRatio: number;
  onsetTolerance: number;
}

/** Hand defaults, set on Basic Pitch output. */
export const DEFAULT_FILTER_SETTINGS: FilterSettings = {
  mergeGap: 0.03,
  minDuration: 0.06,
  minAmplitude: 0.35,
  relativeAmplitude: 0.25,
  harmonicRatio: 0.8,
  onsetTolerance: 0.05,
};

/**
 * Defaults for the model that made the notes; mirrors pipeline.default_settings in Python.
 * The piano model's amplitude is velocity (a different scale), and it leaves small gaps
 * between real re-strikes, so its amplitude thresholds and merging are off.
 */
export function defaultFilterSettings(model: string | null): FilterSettings {
  if (model === 'piano') {
    return { ...DEFAULT_FILTER_SETTINGS, mergeGap: 0, minAmplitude: 0, relativeAmplitude: 0 };
  }
  return DEFAULT_FILTER_SETTINGS;
}

/** Human-readable name of the defaults in use, for the filter panel. */
export function defaultsLabel(model: string | null): string {
  if (model === 'piano') {
    return 'piano model';
  }
  if (model === 'basic-pitch') {
    return 'Basic Pitch';
  }
  return model === null ? 'unknown model (using Basic Pitch defaults)' : `${model} (using Basic Pitch defaults)`;
}

export interface RemovedNote {
  note: Note;
  reason: string;
}

export interface FilterResult {
  kept: Note[];
  removed: RemovedNote[];
}

const byStart = (a: Note, b: Note) => a.start - b.start || a.midi - b.midi;

/** Joins same-pitch notes separated by a real gap of at most `gap` seconds. Touching notes stay separate. */
function mergeFragments(notes: Note[], gap: number, removed: RemovedNote[]): Note[] {
  const byPitch = new Map<number, Note[]>();
  for (const note of [...notes].sort((a, b) => a.midi - b.midi || a.start - b.start)) {
    const list = byPitch.get(note.midi) ?? [];
    list.push(note);
    byPitch.set(note.midi, list);
  }

  const merged: Note[] = [];
  for (const pitchNotes of byPitch.values()) {
    let current = pitchNotes[0];
    let pieces = [current];

    const flush = () => {
      merged.push(current);
      if (pieces.length > 1) {
        for (const piece of pieces) {
          removed.push({ note: piece, reason: 'merged into longer note' });
        }
      }
    };

    for (const note of pitchNotes.slice(1)) {
      const noteGap = note.start - current.stop;
      if (noteGap > TOUCH_EPSILON && noteGap <= gap) {
        current = {
          ...current,
          stop: Math.max(current.stop, note.stop),
          amplitude: weightedAmplitude(current, note),
        };
        pieces.push(note);
      } else {
        flush();
        current = note;
        pieces = [note];
      }
    }
    flush();
  }

  return merged.sort(byStart);
}

/** Duration-weighted amplitude so a short blip doesn't dominate. */
function weightedAmplitude(a: Note, b: Note): number | undefined {
  if (a.amplitude === undefined || b.amplitude === undefined) {
    return a.amplitude ?? b.amplitude;
  }
  const da = a.stop - a.start;
  const db = b.stop - b.start;
  const total = da + db;
  return total > 0 ? (a.amplitude * da + b.amplitude * db) / total : Math.max(a.amplitude, b.amplitude);
}

/** Drops notes much quieter than another note sounding at the same time. */
function dropRelativelyQuiet(notes: Note[], ratio: number): (note: Note) => boolean {
  const sorted = [...notes].sort(byStart);
  return (note) => {
    if (note.amplitude === undefined) {
      return false;
    }
    for (const other of sorted) {
      if (other.start >= note.stop) {
        break;
      }
      if (
        other !== note &&
        other.amplitude !== undefined &&
        other.stop > note.start &&
        note.amplitude < ratio * other.amplitude
      ) {
        return true;
      }
    }
    return false;
  };
}

/** Drops a note a harmonic interval above a louder note starting at about the same time. */
function isOvertone(notes: Note[], ratio: number, tolerance: number): (note: Note) => boolean {
  const sorted = [...notes].sort(byStart);
  return (note) => {
    if (note.amplitude === undefined) {
      return false;
    }
    for (const other of sorted) {
      if (other.start > note.start + tolerance) {
        break;
      }
      if (
        other.amplitude !== undefined &&
        Math.abs(other.start - note.start) <= tolerance &&
        HARMONIC_INTERVALS.has(note.midi - other.midi) &&
        note.amplitude < ratio * other.amplitude
      ) {
        return true;
      }
    }
    return false;
  };
}

/** Runs all enabled filters in order. Notes without `amplitude` skip the amplitude-based steps. */
export function filterNotes(notes: Note[], settings: FilterSettings): FilterResult {
  const removed: RemovedNote[] = [];
  let kept = [...notes].sort(byStart);

  const drop = (shouldDrop: (note: Note) => boolean, reason: string) => {
    const next: Note[] = [];
    for (const note of kept) {
      if (shouldDrop(note)) {
        removed.push({ note, reason });
      } else {
        next.push(note);
      }
    }
    kept = next;
  };

  if (settings.mergeGap > 0) {
    kept = mergeFragments(kept, settings.mergeGap, removed);
  }
  if (settings.minDuration > 0) {
    drop((note) => note.stop - note.start < settings.minDuration, 'too short');
  }
  if (settings.minAmplitude > 0) {
    drop((note) => note.amplitude !== undefined && note.amplitude < settings.minAmplitude, 'too quiet');
  }
  if (settings.relativeAmplitude > 0) {
    drop(dropRelativelyQuiet(kept, settings.relativeAmplitude), 'quieter than a simultaneous note');
  }
  if (settings.harmonicRatio > 0) {
    drop(isOvertone(kept, settings.harmonicRatio, settings.onsetTolerance), 'likely overtone');
  }

  return { kept, removed: removed.sort((a, b) => byStart(a.note, b.note)) };
}

/** Builds the transcribe.py command that applies these settings. */
export function transcribeCommand(audioFile: string, model: string | null, settings: FilterSettings): string {
  return [
    `python transcribe.py ${audioFile} --model ${model ?? 'basic-pitch'} --json notes.json`,
    `--merge-gap ${settings.mergeGap}`,
    `--min-duration ${settings.minDuration}`,
    `--min-amplitude ${settings.minAmplitude}`,
    `--relative-amplitude ${settings.relativeAmplitude}`,
    `--harmonic-ratio ${settings.harmonicRatio}`,
    `--onset-tolerance ${settings.onsetTolerance}`,
  ].join(' ');
}
