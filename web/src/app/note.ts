/** One note as written by `transcribe.py --json`. */
export interface Note {
  pitch: string;
  midi: number;
  start: number;
  stop: number;
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
    return { pitch: note.pitch, midi: note.midi, start: note.start, stop: note.stop };
  });
}
