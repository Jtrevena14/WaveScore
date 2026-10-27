import { Injectable } from '@angular/core';
import { Note } from './note';

const MASTER_GAIN = 0.25;
const ATTACK = 0.005;
const DECAY = 0.4;
const SUSTAIN = 0.35;
const RELEASE = 0.08;
const DEFAULT_AMPLITUDE = 0.7;

interface Voice {
  oscillators: OscillatorNode[];
  envelope: GainNode;
}

/** Small Web Audio synth that sounds whichever notes are passed to `sync()`. */
@Injectable({ providedIn: 'root' })
export class SynthService {
  private context: AudioContext | null = null;
  private output: AudioNode | null = null;
  private readonly voices = new Map<Note, Voice>();

  /** Creates or resumes the audio context. Call from a user gesture such as a click. */
  unlock(): void {
    if (!this.context) {
      this.context = new AudioContext();
      const master = this.context.createGain();
      master.gain.value = MASTER_GAIN;
      const compressor = this.context.createDynamicsCompressor();
      master.connect(compressor).connect(this.context.destination);
      this.output = master;
    }
    if (this.context.state === 'suspended') {
      void this.context.resume();
    }
  }

  /** Starts voices for new notes and releases voices for notes no longer in the list. */
  sync(notes: Note[]): void {
    const wanted = new Set(notes);
    for (const [note, voice] of this.voices) {
      if (!wanted.has(note)) {
        this.release(voice);
        this.voices.delete(note);
      }
    }

    if (!this.context || !this.output) {
      return;
    }
    for (const note of wanted) {
      if (!this.voices.has(note)) {
        this.voices.set(note, this.start(this.context, this.output, note));
      }
    }
  }

  private start(context: AudioContext, output: AudioNode, note: Note): Voice {
    const now = context.currentTime;
    const frequency = 440 * 2 ** ((note.midi - 69) / 12);
    const peak = note.amplitude ?? DEFAULT_AMPLITUDE;

    const envelope = context.createGain();
    envelope.gain.setValueAtTime(0, now);
    envelope.gain.linearRampToValueAtTime(peak, now + ATTACK);
    envelope.gain.setTargetAtTime(peak * SUSTAIN, now + ATTACK, DECAY / 3);
    envelope.connect(output);

    // Triangle fundamental plus a quiet sine an octave up for a soft, piano-like tone.
    const fundamental = context.createOscillator();
    fundamental.type = 'triangle';
    fundamental.frequency.value = frequency;
    fundamental.connect(envelope);

    const overtone = context.createOscillator();
    overtone.type = 'sine';
    overtone.frequency.value = frequency * 2;
    const overtoneGain = context.createGain();
    overtoneGain.gain.value = 0.2;
    overtone.connect(overtoneGain).connect(envelope);

    fundamental.start(now);
    overtone.start(now);
    return { oscillators: [fundamental, overtone], envelope };
  }

  private release(voice: Voice): void {
    const context = this.context;
    if (!context) {
      return;
    }
    const now = context.currentTime;
    const gain = voice.envelope.gain;
    gain.cancelScheduledValues(now);
    gain.setValueAtTime(gain.value, now);
    gain.setTargetAtTime(0, now, RELEASE / 3);

    for (const oscillator of voice.oscillators) {
      oscillator.stop(now + RELEASE);
    }
    voice.oscillators[0].onended = () => voice.envelope.disconnect();
  }
}
