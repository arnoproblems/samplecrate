"""Measuring what a sample sounds like: the numbers behind the sound maps and the Finder.

Results are cached in <library>/.crate/features.json, keyed by path, size and modified time.
They are never written to filenames, so the naming convention is unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_FILE = "features.json"
CACHE_VERSION = 3
WORKERS = 4
FFMPEG_SR = 44100
FLOAT_SUBTYPES = {"FLOAT", "DOUBLE"}

MAX_SPECTRUM_SECONDS = 90
SPECTRUM_WINDOW = 8192
MAX_SPECTRUM_FRAMES = 200
ONSET_BANDS = 24
ONSET_LAG = 3
ONSET_MEDIAN_SPAN = 31
ONSET_RISE_DB = 2.5
ONSET_MIN_GAP_SECONDS = 0.05
AUDIBLE_FLOOR_DB = -50.0        # a sound counts as over once it's this far below its own peak
DECAY_DROP_DB = -20.0
DECAY_SECONDS = 30
ONSET_SECONDS = 60
SIGNATURE_SIZE = 32
SIGNATURE_FLOOR = 1e-6
NEAR_DUPLICATE_CENTER = 0.02    # about a third of a semitone

LOW_MAX_HZ = 250.0
HIGH_MIN_HZ = 4000.0
LOW, MID, HIGH = "Low", "Mid", "High"


@dataclass(frozen=True, slots=True)
class Features:
    center_hz: float            # center of the sound's energy
    low_hz: float               # 10% of the energy is below this
    high_hz: float              # 90% of the energy is below this
    audible: float              # seconds from first to last sound above the audible floor
    decay: float                # seconds to fall 20 dB from the peak
    busy: float                 # onsets per second of audible sound
    movement: float             # how much the sound's spectrum changes over time
    signature: tuple[float, ...]  # compact shape of spectrum and envelope, for spotting near-duplicates
    audio_hash: str = ""          # identical for files holding the same sound, whatever their metadata or format

    @property
    def range(self) -> str:
        return range_of(self.center_hz)


def range_of(hz: float) -> str:
    return LOW if hz < LOW_MAX_HZ else MID if hz < HIGH_MIN_HZ else HIGH


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    return _read_audio(path)[:2]


def _read_audio(path: Path) -> tuple[np.ndarray, int, str]:
    """Decode to (mono, sample rate, hash of the sound).

    Integers are read because they're bit-exact across formats, while floats aren't: WAV and FLAC
    scale differently, so the same sound would hash differently. Keeping the top 16 bits lets a
    24-bit file match its 16-bit copy. Channels are mixed first, so a dual-mono file matches a mono one.
    """
    try:
        with sf.SoundFile(str(path)) as fh:
            sr = fh.samplerate
            if fh.subtype in FLOAT_SUBTYPES:
                # Asking a float file for integers truncates every sample to nearly zero, so scale it ourselves
                pcm = (np.clip(fh.read(dtype="float64", always_2d=True), -1.0, 1.0 - 1e-9) * 2147483648.0).astype("int32")
            else:
                pcm = fh.read(dtype="int32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s32le", "-ac", "2", "-ar", str(FFMPEG_SR), "-"],
            capture_output=True, timeout=300)
        if out.returncode or not out.stdout:
            raise ValueError(f"can't decode {path.name}") from None
        pcm, sr = np.frombuffer(out.stdout, dtype="int32").reshape(-1, 2), FFMPEG_SR
    if len(pcm) == 0:
        raise ValueError(f"{path.name} contains no audio")
    mixed = (pcm >> 16).astype("int32").sum(axis=1) // pcm.shape[1]
    code = hashlib.md5(mixed.astype("int16").tobytes()).hexdigest()
    mono = (pcm.astype("float64").mean(axis=1) / 2147483648.0).astype("float32")
    return mono, sr, code


def _rms_envelope(mono: np.ndarray, window: int, hop: int) -> np.ndarray:
    """RMS level per frame, in one pass over a running sum of squares."""
    if len(mono) < window:
        return np.array([float(np.sqrt(np.mean(mono.astype("float64") ** 2)))])
    energy = np.concatenate(([0.0], np.cumsum(mono.astype("float64") ** 2)))
    starts = np.arange(0, len(mono) - window + 1, hop)
    return np.sqrt(np.maximum(energy[starts + window] - energy[starts], 0.0) / window)


def _to_db(level: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(level, 1e-10))


def _audible_slice(mono: np.ndarray) -> np.ndarray:
    """The part of a sample with sound in it, without leading or trailing silence."""
    window, hop = 1024, 256
    db = _to_db(_rms_envelope(mono, window, hop))
    above = np.where(db > db.max() + AUDIBLE_FLOOR_DB)[0]
    if len(above) == 0:
        return mono
    # A window that only just crosses the floor holds its sound at its far edge, so don't count the whole window
    first, last = int(above[0]), int(above[-1])
    start = 0 if first == 0 else first * hop + window - hop
    end = len(mono) if last == len(db) - 1 else last * hop + hop
    return mono[start:end] if end > start else mono


def _decay_seconds(mono: np.ndarray, sr: int) -> float:
    # A window of about 23 ms follows the body of the sound. Anything shorter measures only the initial click.
    window, hop = 1024, 64
    db = _to_db(_rms_envelope(mono[:sr * DECAY_SECONDS], window, hop))
    peak = int(db.argmax())
    below = np.where(db[peak:] < db[peak] + DECAY_DROP_DB)[0]
    frames = int(below[0]) if len(below) else len(db) - peak
    return frames * hop / sr


def _spectrum_frames(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Power spectra of evenly spaced windows, leaving out silent ones. Returns (power by frame, frequencies)."""
    clip = mono[:sr * MAX_SPECTRUM_SECONDS]
    # The same duration at every sample rate, so a resampled copy measures the same as its original
    size = (round(SPECTRUM_WINDOW * sr / 44100) + 15) // 16 * 16
    if len(clip) < size:
        clip = np.pad(clip, (0, size - len(clip)))
    count = min(MAX_SPECTRUM_FRAMES, max(1, (len(clip) - size) // (size // 2) + 1))
    positions = np.linspace(0, len(clip) - size, count).astype(int)
    frames = clip[positions[:, None] + np.arange(size)[None, :]] * np.hanning(size).astype("float32")
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(size, 1 / sr)
    audible_band = (freqs >= 20) & (freqs <= 20000)
    power, freqs = power[:, audible_band], freqs[audible_band]
    frame_power = power.sum(axis=1)
    if frame_power.max() > 0:
        power = power[frame_power > frame_power.max() * 1e-5]
    return power, freqs


def _spectral_shape(power: np.ndarray, freqs: np.ndarray) -> tuple[float, float, float]:
    mean = power.mean(axis=0)
    total = float(mean.sum())
    if total <= 0:
        return 0.0, 0.0, 0.0
    running = np.cumsum(mean) / total
    return (float((freqs * mean).sum() / total),
            float(freqs[int(np.searchsorted(running, 0.1))]),
            float(freqs[int(np.searchsorted(running, 0.9))]))


def _band_energy(power: np.ndarray, freqs: np.ndarray, bands: int) -> np.ndarray:
    """Energy in log-spaced bands, from one power spectrum."""
    edges = np.geomspace(max(freqs[0], 20.0), freqs[-1] + 1, bands + 1)
    index = np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, bands - 1)
    return np.bincount(index, weights=power, minlength=bands)


def _movement(power: np.ndarray, freqs: np.ndarray) -> float:
    """How far the center of the sound travels, in octaves. A steady chord scores 0 and a filter sweep several.

    Comparing neighboring moments instead would miss exactly the slow evolution this is meant to catch:
    it rated risers as stiller than pads.
    """
    if power.shape[0] < 3:
        return 0.0
    totals = np.maximum(power.sum(axis=1), 1e-12)
    centers = np.log2(np.maximum((power * freqs).sum(axis=1) / totals, 20.0))
    return float(np.percentile(centers, 90) - np.percentile(centers, 10))


def onset_strength(mono: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """How sharply the level rises at each moment, in dB across the spectrum. Returns (strength, hop in samples).

    Each band is compared with about 30 ms earlier. Comparing adjacent frames would read
    the small fluctuations in a sustained tone as a stream of hits.
    """
    clip = mono[:sr * ONSET_SECONDS]
    size, hop = 1024, max(sr // 100, 1)
    if len(clip) < size * 2:
        return np.zeros(0), hop
    count = (len(clip) - size) // hop + 1
    window = np.hanning(size).astype("float32")
    edges = np.unique(np.geomspace(2, size // 2, ONSET_BANDS + 1).astype(int))
    chunks = []
    for start in range(0, count, 512):
        positions = np.arange(start, min(start + 512, count)) * hop
        power = np.abs(np.fft.rfft(clip[positions[:, None] + np.arange(size)[None, :]] * window, axis=1)) ** 2
        chunks.append(np.add.reduceat(power[:, :size // 2], edges[:-1], axis=1))
    bands = np.concatenate(chunks)
    floor = bands.max() * 1e-7 + 1e-12      # rises out of near-silence don't count
    db = 10 * np.log10(bands + floor)
    if len(db) <= ONSET_LAG:
        return np.zeros(len(db)), hop
    return np.concatenate([np.zeros(ONSET_LAG), np.maximum(db[ONSET_LAG:] - db[:-ONSET_LAG], 0).mean(axis=1)]), hop


def _onsets_per_second(mono: np.ndarray, sr: int, audible: float) -> float:
    """Count note and hit starts: moments when the level jumps across the spectrum."""
    if audible <= 0:
        return 0.0
    rise, hop = onset_strength(mono, sr)
    if len(rise) == 0:
        return 0.0
    if len(rise) <= ONSET_LAG + ONSET_MEDIAN_SPAN:
        return 1 / max(audible, 0.01)
    padded = np.pad(rise, ONSET_MEDIAN_SPAN // 2, mode="edge")
    threshold = np.median(np.lib.stride_tricks.sliding_window_view(padded, ONSET_MEDIAN_SPAN), axis=1) + ONSET_RISE_DB
    peaks = np.where((rise[1:-1] > rise[:-2]) & (rise[1:-1] >= rise[2:]) & (rise[1:-1] > threshold[1:-1]))[0] + 1
    kept, last = 0, -10**9
    gap = max(int(ONSET_MIN_GAP_SECONDS * sr / hop), 1)
    for peak in peaks:
        if peak - last >= gap:
            kept += 1
            last = peak
    return max(kept, 1) / max(min(audible, ONSET_SECONDS), 0.01)


def _signature(mono: np.ndarray, power: np.ndarray, freqs: np.ndarray) -> tuple[float, ...]:
    energy = _band_energy(power.mean(axis=0), freqs, SIGNATURE_SIZE)
    # Relative to the sound's own peak, so a quieter copy of a sound has the same signature
    bands = np.log10(energy / max(float(energy.max()), 1e-20) + SIGNATURE_FLOOR)
    envelope = _rms_envelope(mono, max(len(mono) // SIGNATURE_SIZE, 1), max(len(mono) // SIGNATURE_SIZE, 1))
    envelope = np.interp(np.linspace(0, len(envelope) - 1, SIGNATURE_SIZE), np.arange(len(envelope)), envelope)
    envelope = envelope / max(float(envelope.max()), 1e-10)
    return tuple(round(float(v), 3) for v in np.concatenate([bands, envelope]))


def measure(path: Path) -> Features:
    mono, sr, code = _read_audio(Path(path))
    # Measured on the audible part only, so padding a file with silence doesn't change what it "sounds like"
    sound = _audible_slice(mono)
    power, freqs = _spectrum_frames(sound, sr)
    center, low, high = _spectral_shape(power, freqs)
    audible = len(sound) / sr
    return Features(
        center_hz=round(center, 2),
        low_hz=round(low, 2),
        high_hz=round(high, 2),
        audible=round(audible, 4),
        decay=round(_decay_seconds(mono, sr), 4),
        busy=round(_onsets_per_second(mono, sr, audible), 3),
        movement=round(_movement(power, freqs), 4),
        signature=_signature(sound, power, freqs),
        audio_hash=code,
    )


def are_near_duplicates(a: Features, b: Features, tolerance: float = 0.015) -> bool:
    """True when two samples have effectively the same spectrum, envelope, length and pitch center: most likely the same sound."""
    if abs(a.audible - b.audible) > 0.02 + 0.01 * max(a.audible, b.audible):
        return False
    # The signature's bands are too coarse to tell neighboring notes apart, so compare centers directly.
    if abs(a.center_hz - b.center_hz) > NEAR_DUPLICATE_CENTER * max(a.center_hz, b.center_hz, 1.0):
        return False
    return float(np.abs(np.array(a.signature) - np.array(b.signature)).mean()) <= tolerance


# ---------------------------------------------------------------------- the cache

def load_cache(state_dir: Path) -> dict[str, dict]:
    try:
        data = json.loads((state_dir / CACHE_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return data.get("items", {}) if data.get("version") == CACHE_VERSION else {}


def save_cache(state_dir: Path, items: dict[str, dict]) -> None:
    state_dir.mkdir(exist_ok=True)
    path = state_dir / CACHE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "items": items}, separators=(",", ":")))
    os.replace(tmp, path)


def from_entry(entry: dict) -> Features | None:
    try:
        fields = {k: entry[k] for k in Features.__slots__}
    except KeyError:
        return None
    fields["signature"] = tuple(fields["signature"])
    return Features(**fields)


def to_entry(features: Features, stat: os.stat_result) -> dict:
    entry = asdict(features)
    entry["size"], entry["mtime"] = stat.st_size, stat.st_mtime_ns
    return entry


class Analyzer:
    """Measures every sample the cache doesn't already cover, on background threads."""

    def __init__(self, library, *, workers: int = WORKERS):
        self.library = library
        self.workers = workers
        self.done = 0
        self.total = 0
        self.failed: list[str] = []
        self.measured: list[str] = []
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self, on_progress=None) -> int:
        """Measure what's missing and save the results. Returns how many samples were measured."""
        library = self.library
        todo = [s for s in library.samples if s.features is None]
        self.total, self.done = len(todo), 0
        fresh: dict[str, dict] = {}

        def work(sample):
            if self._stop.is_set():
                return
            try:
                stat = sample.path.stat()
                features = measure(sample.path)
            except Exception:
                self.failed.append(sample.rel)
            else:
                sample.features = features
                fresh[sample.rel] = to_entry(features, stat)
                self.measured.append(sample.rel)
            self.done += 1
            if on_progress:
                on_progress(self.done, self.total)

        if todo:
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(work, todo))
        if not library.read_only and (fresh or len(library.feature_cache) != len(library.samples)):
            kept = {s.rel: library.feature_cache[s.rel] for s in library.samples if s.rel in library.feature_cache}
            kept.update(fresh)
            library.feature_cache = kept
            save_cache(library.state_dir, kept)
        elif fresh:
            library.feature_cache.update(fresh)
        return len(fresh)
