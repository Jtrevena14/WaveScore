import { DEFAULT_FILTER_SETTINGS, defaultFilterSettings, filterNotes } from './filters';
import { Note } from './note';

function note(midi: number, start: number, stop: number, amplitude?: number): Note {
  return { pitch: String(midi), midi, start, stop, amplitude };
}

describe('filterNotes', () => {
  it('matches postprocess.py on the synthetic chord case', () => {
    const notes = [
      note(48, 0.0, 1.0, 0.9), // C3 fundamental
      note(60, 0.01, 0.8, 0.5), // C4 overtone, quieter -> removed
      note(52, 0.0, 1.0, 0.8), // E3 real chord note -> kept
      note(70, 0.0, 0.03, 0.6), // blip -> too short
      note(55, 1.2, 1.5, 0.6), // G3 fragments -> merged
      note(55, 1.51, 1.9, 0.6),
      note(65, 0.2, 0.6, 0.15), // quiet -> removed
    ];

    const { kept, removed } = filterNotes(notes, DEFAULT_FILTER_SETTINGS);

    expect(kept.map((n) => [n.midi, n.start, n.stop])).toEqual([
      [48, 0.0, 1.0],
      [52, 0.0, 1.0],
      [55, 1.2, 1.9],
    ]);
    const reasons = removed.map((r) => r.reason).sort();
    expect(reasons).toEqual([
      'likely overtone',
      'merged into longer note',
      'merged into longer note',
      'too quiet',
      'too short',
    ]);
  });

  it('keeps touching same-pitch notes separate (re-strikes)', () => {
    const notes = [note(64, 0.0, 0.9, 0.77), note(64, 0.9, 1.8, 0.77)];
    expect(filterNotes(notes, DEFAULT_FILTER_SETTINGS).kept.length).toBe(2);
  });

  it('skips amplitude filters for notes without amplitude', () => {
    const notes = [note(48, 0.0, 1.0), note(60, 0.0, 1.0)];
    expect(filterNotes(notes, DEFAULT_FILTER_SETTINGS).kept.length).toBe(2);
  });

  it('piano defaults keep near re-strikes and quiet chord tones (JazzLick case)', () => {
    const notes = [
      note(74, 1.789, 2.03, 0.528), // D5
      note(74, 2.045, 2.11, 0.22), // D5 re-strike 15 ms later: merging would swallow it
      note(50, 0.176, 1.202, 0.346), // soft chord tones below the Basic Pitch 0.35 floor
      note(55, 2.717, 3.633, 0.244),
    ];
    expect(filterNotes(notes, defaultFilterSettings('piano')).kept.length).toBe(4);
    expect(filterNotes(notes, DEFAULT_FILTER_SETTINGS).kept.length).toBeLessThan(4);
  });

  it('uses Basic Pitch defaults when the model is unknown', () => {
    expect(defaultFilterSettings(null)).toEqual(DEFAULT_FILTER_SETTINGS);
    expect(defaultFilterSettings('basic-pitch')).toEqual(DEFAULT_FILTER_SETTINGS);
  });

  it('turns every filter off when all settings are 0', () => {
    const off = { mergeGap: 0, minDuration: 0, minAmplitude: 0, relativeAmplitude: 0, harmonicRatio: 0, onsetTolerance: 0 };
    const notes = [note(70, 0.0, 0.01, 0.01), note(82, 0.0, 0.01, 0.005)];
    expect(filterNotes(notes, off).kept.length).toBe(2);
  });
});
