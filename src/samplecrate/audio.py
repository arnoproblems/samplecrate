"""Decoding audio files and pre-computing what the visualizer needs."""

from __future__ import annotations

import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

PEAK_BUCKETS = 4096
GRAM_FRAMES = 768
GRAM_BANDS = 96
GRAM_WINDOW = 2048
CACHE_BYTES = 768 * 1024 * 1024
FFMPEG_SR = 44100


@dataclass(frozen=True, slots=True, eq=False)
class Clip:
    """A decoded sample plus its precomputed overview data."""
    path: Path
    audio: np.ndarray       # frames x 2, float32, ready for playback
    mono: np.ndarray        # frames, float32
    sr: int
    channels: int           # as stored in the file
    format: str
    mins: np.ndarray
    maxs: np.ndarray
    rms: np.ndarray
    peak: float
    rms_total: float
    gram: np.ndarray        # bands x frames, 0..1, low frequencies first
    mean_spectrum: np.ndarray  # one 0..1 value per band

    @property
    def duration(self) -> float:
        return len(self.mono) / self.sr if self.sr else 0.0

    @property
    def nbytes(self) -> int:
        return self.audio.nbytes + self.mono.nbytes

    @property
    def peak_db(self) -> float:
        return to_db(self.peak)

    @property
    def rms_db(self) -> float:
        return to_db(self.rms_total)


def to_db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-10)))


def probe_duration(path: Path) -> float:
    """Length in seconds, read from the file header where possible."""
    try:
        return float(sf.info(str(path)).duration)
    except (RuntimeError, sf.LibsndfileError):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())


def _read_pcm(path: Path) -> tuple[np.ndarray, int, int, str]:
    try:
        info = sf.info(str(path))
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, sr, info.channels, f"{info.format} {info.subtype}"
    except (RuntimeError, sf.LibsndfileError):
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(FFMPEG_SR), "-"],
            capture_output=True, timeout=300)
        if out.returncode or not out.stdout:
            raise ValueError(f"can't decode {path.name}") from None
        return np.frombuffer(out.stdout, dtype="float32").reshape(-1, 2).copy(), FFMPEG_SR, 2, "ffmpeg"


def load_clip(path: Path) -> Clip:
    data, sr, channels, fmt = _read_pcm(path)
    if len(data) == 0:
        raise ValueError(f"{path.name} contains no audio")
    if data.shape[1] == 1:
        stereo = np.repeat(data, 2, axis=1)
    else:
        stereo = data[:, :2]
    mono = data.mean(axis=1).astype("float32")
    mins, maxs, rms = compute_peaks(mono)
    gram, mean_spectrum = compute_spectrogram(mono, sr)
    return Clip(path, stereo, mono, sr, channels, fmt, mins, maxs, rms,
                float(np.abs(mono).max()), float(np.sqrt(np.mean(mono ** 2))), gram, mean_spectrum)


def compute_peaks(mono: np.ndarray, buckets: int = PEAK_BUCKETS):
    n = min(buckets, len(mono))
    edges = np.linspace(0, len(mono), n + 1).astype(int)
    starts = edges[:-1]
    squares = np.add.reduceat(mono.astype("float64") ** 2, starts)
    return (np.minimum.reduceat(mono, starts), np.maximum.reduceat(mono, starts),
            np.sqrt(squares / np.maximum(np.diff(edges), 1)).astype("float32"))


def band_edges(bands: int, sr: int, fft_size: int, fmin: float = 30.0, fmax: float = 18000.0) -> np.ndarray:
    """FFT bin boundaries for log-spaced bands.

    Low bands are narrower than one bin, so several may share a bin. They must not be pushed
    apart to get a bin each: that would shift every band above them and falsify the frequency axis.
    """
    top = min(fmax, sr / 2)
    hz = np.geomspace(fmin, top, bands + 1)
    bins = np.round(hz * fft_size / sr).astype(int)
    return np.clip(bins, 1, fft_size // 2)


def band_power(power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean power per band along the last axis, safe for bands that land on the same bin."""
    starts, ends = edges[:-1], np.maximum(edges[1:], edges[:-1] + 1)
    cumulative = np.cumsum(power, axis=-1)
    cumulative = np.concatenate([np.zeros_like(cumulative[..., :1]), cumulative], axis=-1)
    limit = cumulative.shape[-1] - 1
    ends = np.minimum(ends, limit)
    starts = np.minimum(starts, limit - 1)
    return (cumulative[..., ends] - cumulative[..., starts]) / (ends - starts)


def compute_spectrogram(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    size = GRAM_WINDOW
    usable = np.pad(mono, (0, size)) if len(mono) < size else mono
    count = min(GRAM_FRAMES, max(1, (len(usable) - size) // (size // 4) + 1))
    positions = np.linspace(0, len(usable) - size, count).astype(int)
    frames = usable[positions[:, None] + np.arange(size)[None, :]] * np.hanning(size).astype("float32")
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    bands = band_power(power, band_edges(GRAM_BANDS, sr, size))
    # Tilt about +3 dB per octave so highs are visible, as they would be to the ear
    tilt = np.geomspace(1.0, 18000 / 30, GRAM_BANDS)
    db = 10 * np.log10(bands * tilt + 1e-12)
    top = db.max()
    gram = np.clip((db - (top - 72)) / 72, 0, 1).T
    mean_db = 10 * np.log10(np.mean(bands * tilt, axis=0) + 1e-12)
    mean_spectrum = np.clip((mean_db - (top - 72)) / 72, 0, 1)
    return gram.astype("float32"), mean_spectrum.astype("float32")


class ClipCache:
    """Recently decoded clips, so moving back and forth through the list stays instant."""

    def __init__(self, max_bytes: int = CACHE_BYTES):
        self.max_bytes = max_bytes
        self._clips: OrderedDict[str, Clip] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path) -> Clip:
        key = str(path)
        with self._lock:
            clip = self._clips.get(key)
            if clip is not None:
                self._clips.move_to_end(key)
                return clip
        clip = load_clip(path)
        with self._lock:
            self._clips[key] = clip
            total = sum(c.nbytes for c in self._clips.values())
            while total > self.max_bytes and len(self._clips) > 1:
                _, old = self._clips.popitem(last=False)
                total -= old.nbytes
        return clip

    def forget(self, path: Path) -> None:
        with self._lock:
            self._clips.pop(str(path), None)
