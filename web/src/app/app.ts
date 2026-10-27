import { Component, computed, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { PianoComponent } from './piano/piano';
import { Note, parseNotes } from './note';
import { PlayerService } from './player.service';

@Component({
  selector: 'app-root',
  imports: [PianoComponent, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly player = inject(PlayerService);

  protected readonly fileName = signal<string | null>(null);
  protected readonly notes = signal<Note[]>([]);
  protected readonly error = signal<string | null>(null);

  protected readonly duration = computed(() =>
    this.notes().reduce((max, note) => Math.max(max, note.stop), 0),
  );

  /** Notes sounding at the current playback time. */
  protected readonly activeNotes = computed(() => {
    const t = this.player.time();
    return this.notes().filter((note) => note.start <= t && t < note.stop);
  });

  protected readonly activeMidi = computed(() => this.activeNotes().map((note) => note.midi));

  protected async onFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    try {
      const parsed: unknown = JSON.parse(await file.text());
      this.notes.set(parseNotes(parsed));
      this.fileName.set(file.name);
      this.error.set(null);
      this.player.reset();
    } catch (err) {
      this.notes.set([]);
      this.fileName.set(null);
      this.error.set(err instanceof Error ? err.message : 'Could not read the file');
    }
  }

  protected togglePlay(): void {
    if (this.player.playing()) {
      this.player.pause();
    } else {
      this.player.play(this.duration());
    }
  }

  protected onSeek(event: Event): void {
    this.player.seek(Number((event.target as HTMLInputElement).value));
  }

  protected isPlaying(note: Note): boolean {
    const t = this.player.time();
    return note.start <= t && t < note.stop;
  }
}
