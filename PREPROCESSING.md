# Preprocessing: Reducing Phantom Notes from Basic Pitch

## Problem
Basic Pitch reports notes that were not played. We call these **phantom notes**. They are most common in chords, where several notes sound at once. Earlier attempts hit the same issue, so this doc collects ideas for fixing it before we pick one.

This is a brainstorm. Nothing here is implemented yet.

## 1. Why phantom notes appear
- **Harmonics and overtones.** A note at pitch *p* has energy at *p+12, p+19, p+24, p+28* (octave, octave+fifth, two octaves, two octaves+major third). In a chord these partials land on other real notes, or on pitches that are not in the chord at all.
- **Neighbour leakage.** A strong note can produce weaker detections one semitone away, or an octave below.
- **Fragmentation.** One sustained note is reported as several short notes of the same pitch.
- **Short spurious blips.** Very short, low-amplitude detections from noise, reverb, pedal resonance, or mp3 compression artefacts.
- **Sympathetic resonance and decay.** Strings or room resonance keep a pitch sounding after it was released, or excite pitches that were never struck.

## 2. Ideas

Cost is the effort to build. Payoff is the expected reduction in phantom notes.

### 2.1 Measure first: an evaluation harness
- **What:** Make a set of test clips with known ground truth. Start with single notes, intervals, triads, seventh chords, and a short melody. Render them to audio from MIDI using `pretty_midi` with FluidSynth, or a simple synth. Transcribe each one and score the output with `mir_eval` note-level precision, recall and F1 (50 ms onset tolerance).
- **Why:** Phantom notes show up as lost precision. Without numbers, every other change is guesswork.
- **Cost:** Medium. **Payoff:** Enables everything else.

### 2.2 Tune Basic Pitch's own parameters
- **What:** `predict()` exposes settings we can change: `onset_threshold`, `frame_threshold`, `minimum_note_length`, `minimum_frequency`, `maximum_frequency`. Raising the thresholds, lengthening the minimum note, and limiting the frequency range to the instrument should remove many blips. Expose them as CLI flags on `transcribe.py`.
- **Cost:** Low. **Payoff:** Medium. Check the exact parameter names and defaults against the installed version before relying on them.

### 2.3 Post-filter the note events
- **Amplitude floor.** Drop notes below an absolute threshold, and notes much quieter than a simultaneous note.
- **Minimum duration.** Drop notes shorter than a set length.
- **Merge fragments.** Join same-pitch notes separated by a very small gap.
- **Harmonic suppression.** If a quieter note sits 12, 19, 24 or 28 semitones above a louder note, and starts at the same time, treat it as an overtone and drop it. This targets the chord case directly.
- **Cost:** Low. **Payoff:** High. Needs amplitude in the output (see 2.8).

### 2.4 Work from the raw model output
- **What:** `predict()` returns `model_output`, which holds the note, onset and contour matrices. We could write our own note extraction. It would require a clear onset peak for each note, use per-pitch adaptive thresholds, and reject any note that has no onset support.
- **Cost:** High. **Payoff:** High, but only if 2.3 is not enough.

### 2.5 Clean the audio before the model
- **Format:** Convert mp3 to lossless wav first (ffmpeg), so the decoder's artefacts don't add detections.
- **Basic cleanup:** Mono mixdown, loudness normalisation, trimming silence.
- **Noise reduction:** `noisereduce`, if recordings have hiss or room noise.
- **Percussive separation:** `librosa.effects.hpss` can remove key clicks and hammer noise that create false onsets.
- **Source separation:** Demucs, only if the input is a full band mix. It is slow and can damage the notes we want.
- **Cost:** Low to medium. **Payoff:** Varies by recording. Adopt only if the harness shows a gain.

### 2.6 Consensus and ensembles
- **Multiple runs:** Transcribe several parameter settings, or lightly shifted copies of the audio, and keep only notes that appear in most runs.
- **A second model:** Intersect Basic Pitch with another transcriber. For solo piano, ByteDance's `piano_transcription_inference` is generally more accurate than Basic Pitch. Check this on our own harness before adopting it.
- **Cost:** Medium to high. **Payoff:** High for precision, but it can also lose real notes.

### 2.7 Music-theory priors (later, riskier)
- **What:** Estimate the key, and use chord templates to penalise isolated out-of-key notes with low confidence.
- **Risk:** Music is not always in key, so this can remove real notes.
- **Cost:** Medium. **Payoff:** Uncertain.

### 2.8 Human in the loop through the visualizer
- **What:** Add `amplitude` to the `--json` output. Add a confidence slider to the web app, so thresholds can be tuned by looking at the piano roll.
- **Cost:** Low. **Payoff:** Useful for tuning, and it does not fix the model itself.

## 3. Suggested order
1. **Evaluation harness** (2.1) with a small chord test set.
2. **Parameter flags** (2.2) and `amplitude` in the JSON (2.8).
3. **Post-filters with harmonic suppression** (2.3), scored against the harness.
4. **Audio cleanup** (2.5), only if the numbers show it helps.
5. **Ensembles or a second model** (2.6), if filtering stops improving results.

## 4. Open questions
- What instrument and recording type are we targeting: solo piano, a mixed band, or phone recordings?
- Do we have any recordings with known sheet music or MIDI that we can use as ground truth?
- Which error matters more: a phantom note, or a missed note?
- Are heavier dependencies acceptable (FluidSynth, ffmpeg, a second model)?

## 5. Related files
- `transcribe.py`: the script that would change.
- `PLAN.md`, `PIANO_VISUALIZER.md`: earlier specs.
- `web/src/app/note.ts`: the note shape the visualizer reads.
