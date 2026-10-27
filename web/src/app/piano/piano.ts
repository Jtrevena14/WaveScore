import { Component, computed, input } from '@angular/core';

const FIRST_MIDI = 21; // A0
const LAST_MIDI = 108; // C8
const BLACK_PITCH_CLASSES = new Set([1, 3, 6, 8, 10]);

const WHITE_KEY_COUNT = 52;
const WHITE_WIDTH = 100 / WHITE_KEY_COUNT; // percent of keyboard width
const BLACK_WIDTH = WHITE_WIDTH * 0.6;

interface PianoKey {
  midi: number;
  /** Left edge in percent, used by black keys only. */
  left: number;
}

function buildKeys(): { white: PianoKey[]; black: PianoKey[] } {
  const white: PianoKey[] = [];
  const black: PianoKey[] = [];
  let whitesBefore = 0;

  for (let midi = FIRST_MIDI; midi <= LAST_MIDI; midi++) {
    if (BLACK_PITCH_CLASSES.has(midi % 12)) {
      // A black key sits on the boundary after the white keys already placed.
      black.push({ midi, left: whitesBefore * WHITE_WIDTH - BLACK_WIDTH / 2 });
    } else {
      white.push({ midi, left: 0 });
      whitesBefore++;
    }
  }

  return { white, black };
}

const KEYS = buildKeys();

@Component({
  selector: 'app-piano',
  templateUrl: './piano.html',
  styleUrl: './piano.css',
})
export class PianoComponent {
  /** MIDI numbers of the notes that should be lit right now. */
  readonly active = input<readonly number[]>([]);

  protected readonly whiteKeys = KEYS.white;
  protected readonly blackKeys = KEYS.black;
  protected readonly activeSet = computed(() => new Set(this.active()));

  protected isActive(midi: number): boolean {
    return this.activeSet().has(midi);
  }
}
