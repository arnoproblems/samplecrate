"""Audio playback with a sample-accurate position, so visuals stay in sync with what you hear.

A single output stream stays open and is fed by a callback, which lets playback start
instantly and lets loops repeat seamlessly. Switching samples fades the old one out
quickly to avoid a click, without softening the attack of the new one.

Opening an audio device can take seconds, so it never happens on the caller's thread:
play() always returns at once, and a sound that has to wait for the device starts as
soon as it's ready.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

CHANNELS = 2
IDLE_RELEASE_SECONDS = 30.0
SILENT_BLOCK = 1024
DEVICE_LATENCY_SECONDS = 0.06
DEVICE_BLOCK_FRAMES = 1024
MAX_INTERPOLATION_SECONDS = 0.1
WARM_UP_RATE = 44100


@dataclass(slots=True)
class _Voice:
    data: np.ndarray
    sr: int
    pos: int = 0
    loop: bool = False
    seek_to: int | None = None


class SilentStream:
    """An output stream with no sound: it runs the callback in real time for `--no-audio`, SSH and tests."""
    latency = 0.0

    def __init__(self, samplerate: int, callback):
        self._sr = samplerate
        self._callback = callback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        out = np.zeros((SILENT_BLOCK, CHANNELS), dtype="float32")
        interval = SILENT_BLOCK / self._sr
        next_at = time.monotonic()
        while not self._stop.is_set():
            self._callback(out, SILENT_BLOCK, None, None)
            next_at += interval
            self._stop.wait(max(0.0, next_at - time.monotonic()))

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


def _silent_factory(samplerate: int, callback) -> SilentStream:
    return SilentStream(samplerate, callback)


def _device_factory(samplerate: int, callback):
    import sounddevice as sd
    # The callback is Python and competes with the UI for the interpreter, so it gets a
    # generous buffer. The displayed position compensates, so visuals stay in sync with the sound.
    return sd.OutputStream(samplerate=samplerate, channels=CHANNELS, dtype="float32", callback=callback,
                           latency=DEVICE_LATENCY_SECONDS, blocksize=DEVICE_BLOCK_FRAMES)


class Player:
    def __init__(self, stream_factory=_silent_factory, volume: float = 0.8):
        self._factory = stream_factory
        self._stream = None
        self._stream_sr: int | None = None
        self._stream_lock = threading.Lock()
        self._voice: _Voice | None = None
        self._fading: _Voice | None = None
        self._waiting: _Voice | None = None     # wants to play, but the device isn't ready
        self._opening_sr: int | None = None
        self._volume = float(volume)
        self._gain = float(volume)
        self._held_seconds = 0.0
        self._last_active = time.monotonic()
        self._closed = False
        self._callback_at: float | None = None
        self.error: str | None = None
        self.underruns = 0

    # ------------------------------------------------------------------ audio thread

    def _callback(self, out, frames, time_info, status) -> None:
        self._callback_at = time.monotonic()
        if status and getattr(status, "output_underflow", False):
            self.underruns += 1
        out.fill(0)
        fading, self._fading = self._fading, None
        if fading is not None:
            self._mix(fading, out, frames, np.linspace(self._gain, 0.0, frames, dtype="float32"))
        voice = self._voice
        if voice is None:
            return
        target = self._volume
        ramp = np.linspace(self._gain, target, frames, dtype="float32")
        self._gain = target
        if not self._mix(voice, out, frames, ramp) and self._voice is voice:
            self._voice = None
            self._held_seconds = 0.0

    @staticmethod
    def _mix(voice: _Voice, out, frames: int, gains: np.ndarray) -> bool:
        """Add a block of a voice to the output. Returns False when a non-looping voice has ended."""
        if voice.seek_to is not None:
            voice.pos, voice.seek_to = voice.seek_to, None
        data, pos, total, written = voice.data, voice.pos, len(voice.data), 0
        alive = True
        while written < frames:
            take = min(frames - written, total - pos)
            if take > 0:
                out[written:written + take] += data[pos:pos + take] * gains[written:written + take, None]
                written += take
                pos += take
            if pos >= total:
                if not voice.loop:
                    alive = False
                    break
                pos = 0
        voice.pos = pos
        return alive

    # ------------------------------------------------------------------ the output device

    def warm_up(self, sr: int = WARM_UP_RATE) -> None:
        """Open the device in the background, so the first play is instant."""
        if self._stream is None:
            self._open_in_background(sr)

    def _open_in_background(self, sr: int) -> None:
        if self._opening_sr == sr:
            return
        self._opening_sr = sr
        threading.Thread(target=self._open_stream, args=(sr,), daemon=True).start()

    def _open_stream(self, sr: int) -> None:
        with self._stream_lock:
            if self._closed:
                return
            self._close_stream()
            try:
                stream = self._factory(sr, self._callback)
                stream.start()
            except Exception as exc:
                # No usable device: carry on silently, so the visuals keep working
                self.error = str(exc)
                self._factory = _silent_factory
                stream = self._factory(sr, self._callback)
                stream.start()
            self._stream, self._stream_sr = stream, sr
            self._opening_sr = None
            self._last_active = time.monotonic()
        waiting = self._waiting
        if waiting is not None:
            if waiting.sr == sr:
                self._waiting, self._voice = None, waiting
                self._gain = self._volume
            else:
                self._open_in_background(waiting.sr)

    def _close_stream(self) -> None:
        stream, self._stream, self._stream_sr = self._stream, None, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def release_if_idle(self) -> None:
        """Close the device after a period of silence, so we don't hold the audio hardware open forever."""
        if self.is_playing or self._stream is None or self._opening_sr is not None:
            return
        if time.monotonic() - self._last_active > IDLE_RELEASE_SECONDS:
            threading.Thread(target=self._release, daemon=True).start()
            self._last_active = time.monotonic()

    def _release(self) -> None:
        with self._stream_lock:
            if not self.is_playing:
                self._close_stream()

    def close(self) -> None:
        self._closed = True
        self._voice = self._waiting = None
        with self._stream_lock:
            self._close_stream()

    # ------------------------------------------------------------------ control

    def play(self, audio: np.ndarray, sr: int, *, start: float = 0.0, loop: bool = False) -> None:
        """Start a clip. Returns immediately, even if the device still has to be opened."""
        total = len(audio)
        if total == 0:
            return
        begin = min(max(int(start * sr), 0), total - 1)
        voice = _Voice(audio, sr, pos=begin, loop=loop)
        self._fading = self._voice
        self._voice = None
        self._last_active = time.monotonic()
        if self._stream is not None and self._stream_sr == sr and self._opening_sr is None:
            self._waiting = None
            self._gain = self._volume
            self._voice = voice
        else:
            self._waiting = voice
            self._open_in_background(sr)

    def pause(self) -> None:
        """Stop and remember the position, so a later play can resume from it."""
        voice = self._voice or self._waiting
        if voice is None:
            return
        self._held_seconds = voice.pos / voice.sr
        self._waiting = None
        self._fading, self._voice = self._voice, None
        self._last_active = time.monotonic()

    def stop(self) -> None:
        self.pause()
        self._held_seconds = 0.0

    def seek(self, seconds: float) -> None:
        voice = self._voice or self._waiting
        if voice is None:
            self._held_seconds = max(seconds, 0.0)
            return
        voice.seek_to = min(max(int(seconds * voice.sr), 0), len(voice.data) - 1)

    def set_loop(self, loop: bool) -> None:
        voice = self._voice or self._waiting
        if voice is not None:
            voice.loop = loop

    @property
    def is_playing(self) -> bool:
        """True from the moment play is requested, including while waiting for the device."""
        return self._voice is not None or self._waiting is not None

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = min(max(float(value), 0.0), 1.0)

    @property
    def position(self) -> float:
        """Seconds into the clip, corrected for output latency: what is reaching the speakers right now."""
        waiting = self._waiting
        if waiting is not None:
            return waiting.pos / waiting.sr
        voice = self._voice
        if voice is None:
            return self._held_seconds
        latency = getattr(self._stream, "latency", 0.0) or 0.0
        # Callbacks arrive in bursts, so fill in the time since the last one for smooth motion
        since = 0.0 if self._callback_at is None else min(time.monotonic() - self._callback_at, MAX_INTERPOLATION_SECONDS)
        duration = len(voice.data) / voice.sr
        seconds = voice.pos / voice.sr - latency + since
        if voice.loop:
            # Just after a loop wraps, the speakers are still playing the end of the clip
            return seconds % duration
        return min(max(seconds, 0.0), duration)


def create_player(*, no_audio: bool = False, volume: float = 0.8) -> tuple[Player, str | None]:
    """Build a player for the real audio device, or a silent one when there isn't a usable device."""
    if no_audio:
        return Player(_silent_factory, volume), None
    try:
        import sounddevice as sd
        sd.query_devices(kind="output")
    except Exception as exc:
        return Player(_silent_factory, volume), f"no audio output ({exc}); playing silently"
    player = Player(_device_factory, volume)
    player.warm_up()
    return player, None
