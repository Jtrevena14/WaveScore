import { Component, computed, effect, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { PianoComponent } from './piano/piano';
import { FilterPanelComponent } from './filter-panel/filter-panel';
import { defaultFilterSettings, defaultsLabel, filterNotes, transcribeCommand } from './filters';
import { Note, parseNotesFile } from './note';
import { PlayerService } from './player.service';
import { SynthService } from './synth.service';

/** A row in the note list; `reason` is set when the filters removed the note. */
interface ListItem {
  note: Note;
  reason?: string;
}

@Component({
  selector: 'app-root',
  imports: [PianoComponent, FilterPanelComponent, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly player = inject(PlayerService);
  private readonly synth = inject(SynthService);

  protected readonly fileName = signal<string | null>(null);
  /** Notes exactly as loaded from the file. */
  protected readonly rawNotes = signal<Note[]>([]);
  /** The transcribe.py model that made the file, or null for older files that don't say. */
  protected readonly model = signal<string | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly soundOn = signal(true);

  protected readonly modelDefaults = computed(() => defaultFilterSettings(this.model()));
  protected readonly defaultsLabel = computed(() => defaultsLabel(this.model()));
  protected readonly filterSettings = signal(defaultFilterSettings(null));
  protected readonly filtersOn = signal(true);
  protected readonly showRemoved = signal(false);

  protected readonly filterResult = computed(() => filterNotes(this.rawNotes(), this.filterSettings()));

  /** The notes that are highlighted and played. */
  protected readonly notes = computed(() =>
    this.filtersOn() ? this.filterResult().kept : this.rawNotes(),
  );

  protected readonly listItems = computed<ListItem[]>(() => {
    const kept: ListItem[] = this.notes().map((note) => ({ note }));
    if (!this.filtersOn() || !this.showRemoved()) {
      return kept;
    }
    return [...kept, ...this.filterResult().removed].sort(
      (a, b) => a.note.start - b.note.start || a.note.midi - b.note.midi,
    );
  });

  protected readonly hasAmplitude = computed(() =>
    this.rawNotes().every((note) => note.amplitude !== undefined),
  );

  protected readonly command = computed(() => {
    const name = this.fileName() ?? 'audio.json';
    const audio = name.replace(/(\.raw)?\.json$/i, '') + '.mp3';
    return transcribeCommand(audio, this.model(), this.filterSettings());
  });

  protected readonly duration = computed(() =>
    this.notes().reduce((max, note) => Math.max(max, note.stop), 0),
  );

  /** Notes sounding at the current playback time. */
  protected readonly activeNotes = computed(() => {
    const t = this.player.time();
    return this.notes().filter((note) => note.start <= t && t < note.stop);
  });

  protected readonly activeMidi = computed(() => this.activeNotes().map((note) => note.midi));

  constructor() {
    // Sound exactly the highlighted notes; pausing, muting or loading a new file silences them.
    effect(() => {
      this.synth.sync(this.player.playing() && this.soundOn() ? this.activeNotes() : []);
    });
  }

  protected async onFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    try {
      const parsed: unknown = JSON.parse(await file.text());
      const notesFile = parseNotesFile(parsed);
      this.rawNotes.set(notesFile.notes);
      this.model.set(notesFile.model);
      this.filterSettings.set(defaultFilterSettings(notesFile.model));
      this.fileName.set(file.name);
      this.error.set(null);
      this.player.reset();
    } catch (err) {
      this.rawNotes.set([]);
      this.model.set(null);
      this.fileName.set(null);
      this.error.set(err instanceof Error ? err.message : 'Could not read the file');
    }
  }

  protected togglePlay(): void {
    if (this.player.playing()) {
      this.player.pause();
    } else {
      this.synth.unlock();
      this.player.play(this.duration());
    }
  }

  protected onSoundToggle(event: Event): void {
    this.soundOn.set((event.target as HTMLInputElement).checked);
  }

  protected onSeek(event: Event): void {
    this.player.seek(Number((event.target as HTMLInputElement).value));
  }

  protected isPlaying(note: Note): boolean {
    const t = this.player.time();
    return note.start <= t && t < note.stop;
  }
}
