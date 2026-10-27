import { Injectable, signal } from '@angular/core';

/** Playback clock. Advances `time` in seconds using requestAnimationFrame. */
@Injectable({ providedIn: 'root' })
export class PlayerService {
  readonly time = signal(0);
  readonly playing = signal(false);

  private duration = 0;
  private frameId: number | null = null;
  private lastFrame = 0;

  play(duration: number): void {
    if (this.playing()) {
      return;
    }
    this.duration = duration;
    if (this.time() >= duration) {
      this.time.set(0);
    }
    this.playing.set(true);
    this.lastFrame = performance.now();
    this.frameId = requestAnimationFrame(this.tick);
  }

  pause(): void {
    this.playing.set(false);
    if (this.frameId !== null) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }
  }

  seek(seconds: number): void {
    this.time.set(Math.max(0, seconds));
  }

  reset(): void {
    this.pause();
    this.time.set(0);
  }

  private readonly tick = (now: number): void => {
    const next = this.time() + (now - this.lastFrame) / 1000;
    this.lastFrame = now;

    if (next >= this.duration) {
      this.time.set(this.duration);
      this.pause();
      return;
    }

    this.time.set(next);
    this.frameId = requestAnimationFrame(this.tick);
  };
}
