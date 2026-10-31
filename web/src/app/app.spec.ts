import { TestBed } from '@angular/core/testing';
import { App } from './app';
import { parseNotes, parseNotesFile } from './note';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('should render the heading', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Piano Visualizer');
  });

  it('should render all 88 piano keys', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelectorAll('.key').length).toBe(88);
  });
});

describe('parseNotes', () => {
  it('accepts the output of transcribe.py --json', () => {
    const notes = parseNotes([{ pitch: 'C4', midi: 60, start: 0.512, stop: 0.934 }]);
    expect(notes).toEqual([{ pitch: 'C4', midi: 60, start: 0.512, stop: 0.934 }]);
  });

  it('rejects data that is not an array', () => {
    expect(() => parseNotes({ midi: 60 })).toThrowError('Expected a JSON array of notes');
  });

  it('rejects a note with missing fields', () => {
    expect(() => parseNotes([{ pitch: 'C4', midi: 60 }])).toThrowError(
      'Note 0 is missing pitch, midi, start or stop',
    );
  });
});

describe('parseNotesFile', () => {
  const note = { pitch: 'C4', midi: 60, start: 0.5, stop: 0.9, amplitude: 0.4 };

  it('reads the model from the current file shape', () => {
    expect(parseNotesFile({ model: 'piano', notes: [note] })).toEqual({ model: 'piano', notes: [note] });
  });

  it('accepts older files that are a bare array, with the model unknown', () => {
    expect(parseNotesFile([note])).toEqual({ model: null, notes: [note] });
  });

  it('rejects anything else', () => {
    expect(() => parseNotesFile({ midi: 60 })).toThrowError('Expected a notes file from transcribe.py --json');
  });
});
