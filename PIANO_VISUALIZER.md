# Feature: Piano Visualizer (Angular web app)

## Goal
Show the notes found by `transcribe.py` on a virtual piano so transcription results can be checked quickly by eye.

## Non-goals
- No sheet-music rendering.
- No audio upload or transcription inside the browser.
- No backend server. The app is fully static.

## Data hand-off
Add an optional `--json <out.json>` flag to `transcribe.py`. When it is given, the script also writes:

```json
[
  { "pitch": "C4", "midi": 60, "start": 0.512, "stop": 0.934 }
]
```

- `pitch` is the note name, `midi` is the MIDI number, `start` and `stop` are in seconds.
- The existing text output stays exactly as it is. The flag only adds a file.
- The `midi` field lets the UI work with numbers and never parse note names.
- The web app reads this file through a file picker. Drag-and-drop is optional.

## Stack
- Angular (current stable), standalone components, no NgModules.
- TypeScript in strict mode.
- Plain CSS. No UI library.

## Location
A new `web/` folder next to `transcribe.py`, created with `ng new`.

## Features
1. **Piano keyboard**: 88 keys, A0 to C8 (MIDI 21 to 108). White and black keys are drawn in CSS.
2. **Load JSON**: a control that loads a notes file produced by `--json`.
3. **Playback**: a Play/Pause button and a time slider for the current time `t`. A key is highlighted while `start <= t < stop` for one of its notes.
4. **Note list**: a scrolling list or timeline of all notes under the piano, as a quick sanity view.
5. **Stretch goal**: clicking a note in the list seeks playback to that note's start.

## Components and types
- `Note` interface: `{ pitch: string; midi: number; start: number; stop: number }`.
- `PianoComponent`: takes the set of currently active MIDI numbers and draws the keys.
- `PlayerService`: a `requestAnimationFrame` clock that publishes the current time.
- `App` shell: holds the loaded notes, the list and the controls.

## Verification
1. `python transcribe.py Mary_Had_a_Little_Lamb.mp3 --json notes.json`
2. Check that the text output of `transcribe.py` is unchanged compared with before the flag was added.
3. `cd web` then `ng serve`, open the local URL, and load `notes.json`.
4. Press Play. The highlighted keys should match the melody as it plays.

## Open questions and future work
- Playing the audio in sync with the highlights.
- A falling-notes view.
- Sheet-music rendering, later.
