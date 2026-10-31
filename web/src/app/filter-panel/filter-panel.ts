import { Component, input, output, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FilterSettings } from '../filters';

interface SettingRow {
  key: keyof FilterSettings;
  label: string;
  hint: string;
  max: number;
  step: number;
  /** Disabled when the loaded notes have no amplitude. */
  needsAmplitude: boolean;
}

const ROWS: SettingRow[] = [
  { key: 'mergeGap', label: 'Merge gap (s)', hint: 'Joins same-pitch pieces split by a gap up to this long', max: 0.2, step: 0.005, needsAmplitude: false },
  { key: 'minDuration', label: 'Min duration (s)', hint: 'Drops blips shorter than this', max: 0.3, step: 0.01, needsAmplitude: false },
  { key: 'minAmplitude', label: 'Min amplitude', hint: 'Drops notes quieter than this', max: 1, step: 0.01, needsAmplitude: true },
  { key: 'relativeAmplitude', label: 'Relative amplitude', hint: 'Drops notes quieter than this × a note sounding with them', max: 1, step: 0.01, needsAmplitude: true },
  { key: 'harmonicRatio', label: 'Harmonic ratio', hint: 'Drops notes 12/19/24/28 semitones above a louder note, if quieter than this × it', max: 1.5, step: 0.01, needsAmplitude: true },
  { key: 'onsetTolerance', label: 'Onset tolerance (s)', hint: 'How close starts must be to count as "the same time" for overtones', max: 0.15, step: 0.005, needsAmplitude: false },
];

/** Sliders for the post-filters, so they can be tuned while listening. */
@Component({
  selector: 'app-filter-panel',
  imports: [DecimalPipe],
  templateUrl: './filter-panel.html',
  styleUrl: './filter-panel.css',
})
export class FilterPanelComponent {
  readonly settings = input.required<FilterSettings>();
  /** Defaults for the model that made the notes; "Reset to defaults" restores these. */
  readonly defaults = input.required<FilterSettings>();
  readonly defaultsLabel = input.required<string>();
  readonly enabled = input.required<boolean>();
  readonly showRemoved = input.required<boolean>();
  readonly keptCount = input.required<number>();
  readonly totalCount = input.required<number>();
  readonly hasAmplitude = input.required<boolean>();
  readonly command = input.required<string>();

  readonly settingsChange = output<FilterSettings>();
  readonly enabledChange = output<boolean>();
  readonly showRemovedChange = output<boolean>();

  protected readonly rows = ROWS;
  protected readonly copied = signal(false);

  protected onSlider(key: keyof FilterSettings, event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    this.settingsChange.emit({ ...this.settings(), [key]: value });
  }

  protected reset(): void {
    this.settingsChange.emit(this.defaults());
  }

  protected onEnabled(event: Event): void {
    this.enabledChange.emit((event.target as HTMLInputElement).checked);
  }

  protected onShowRemoved(event: Event): void {
    this.showRemovedChange.emit((event.target as HTMLInputElement).checked);
  }

  protected async copyCommand(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.command());
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 1500);
    } catch {
      // Clipboard can be blocked; the command is still shown in the text field to copy by hand.
    }
  }
}
