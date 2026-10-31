/** One note as written by `transcribe.py --json`. */
export interface Note {
  pitch: string;
  midi: number;
  start: number;
  stop: number;
  /** Model confidence or velocity, 0-1. Missing in files written before it was added. */
  amplitude?: number;
}

/** A loaded notes file. `model` is the transcribe.py --model that made it, or null for old files. */
export interface NotesFile {
  model: string | null;
  notes: Note[];
}

/**
 * Accepts both shapes transcribe.py has written: `{ model, notes: [...] }` (current)
 * and a bare array of notes (older files, model unknown).
 */
export function parseNotesFile(data: unknown): NotesFile {
  if (Array.isArray(data)) {
    return { model: null, notes: parseNotes(data) };
  }
  if (data !== null && typeof data === 'object' && 'notes' in data) {
    const file = data as { model?: unknown; notes: unknown };
    return {
      model: typeof file.model === 'string' ? file.model : null,
      notes: parseNotes(file.notes),
    };
  }
  throw new Error('Expected a notes file from transcribe.py --json');
}

/** Checks the parsed JSON shape and returns it as typed notes. Throws with a readable message if invalid. */
export function parseNotes(data: unknown): Note[] {
  if (!Array.isArray(data)) {
    throw new Error('Expected a JSON array of notes');
  }

  return data.map((item, index) => {
    const note = (item ?? {}) as Partial<Note>;
    if (
      typeof note.pitch !== 'string' ||
      typeof note.midi !== 'number' ||
      typeof note.start !== 'number' ||
      typeof note.stop !== 'number'
    ) {
      throw new Error(`Note ${index} is missing pitch, midi, start or stop`);
    }
    const parsed: Note = { pitch: note.pitch, midi: note.midi, start: note.start, stop: note.stop };
    if (typeof note.amplitude === 'number') {
      parsed.amplitude = note.amplitude;
    }
    return parsed;
  });
}
