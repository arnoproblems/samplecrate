"""Fixtures. Everything runs in a temporary library, with a temporary Trash and silent audio."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from samplecrate.library import Library
from samplecrate.naming import parse_name

SR = 22050
LOOP_NOISE_SEED = 7

CONFORMING = [
    "Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav",
    "Drums__Kick__OneShot__NA__F__0.30s__dark-tuned.wav",
    "Drums__Hat-Open__OneShot__NA__NA__0.40s__bright.wav",
    "Drums__Full__Loop__120bpm__NA__2.00s__busy-break.wav",
    "Drums__Full__Loop__174bpm__NA__1.38s__jungle.wav",
    "Bass__808__OneShot__NA__F__0.50s__long.wav",
    "Bass__Reese__Loop__087bpm__Amin__2.76s__wide.wav",
    "Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav",
    "Instruments__Piano__Loop__122bpm__Cmaj__1.97s__soft.wav",
    "Vocals__Phrase__Loop__090bpm__Emin__2.67s__hello.wav",
    "FX__Riser__OneShot__120bpm__NA__1.00s__whoosh.wav",
]
UNSORTED = "unsorted-stuff/my new kick.wav"
INDEX_FIELDS = ["path", "main", "sub", "type", "bpm", "bpm_source", "key", "key_source", "key_guess_low_confidence",
                "length_s", "description", "original_name", "source_paths", "duplicate_copies", "md5",
                "sample_rate", "format", "channels", "category_source", "type_source"]


def tone(seconds: float, hz: float = 110.0, sr: int = SR, decay: bool = True) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    wave = 0.8 * np.sin(2 * np.pi * hz * t)
    if decay:
        wave *= np.exp(-3 * t)
    return np.stack([wave, wave * 0.9], axis=1).astype("float32")


def write_wav(path: Path, seconds: float, subtype: str = "PCM_16", **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, tone(seconds, **kwargs), kwargs.get("sr", SR), subtype=subtype)
    return path


@pytest.fixture
def library_root(tmp_path) -> Path:
    root = tmp_path / "lib"
    root.mkdir()
    rows = []
    for number, filename in enumerate(CONFORMING):
        name = parse_name(filename)
        write_wav(root / name.relpath, name.length, hz=80 + number * 37)    # each a distinct sound
        rows.append({
            "path": name.relpath, "main": name.main, "sub": name.sub, "type": name.type, "bpm": name.bpm or "",
            "bpm_source": "name" if name.bpm else "", "key": name.key or "", "key_source": "name" if name.key else "",
            "key_guess_low_confidence": "", "length_s": f"{name.length:.2f}", "description": name.desc,
            "original_name": f"ORIG_{name.desc}.wav", "source_paths": f"ToSort/ORIG_{name.desc}.wav",
            "duplicate_copies": 0, "md5": "0" * 32, "sample_rate": SR, "format": "PCM_16", "channels": 2,
            "category_source": "name", "type_source": "name"})
    write_wav(root / UNSORTED, 0.25, hz=620)
    with open(root / "_index.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return root


@pytest.fixture
def library(library_root) -> Library:
    return Library(library_root).scan()


@pytest.fixture(autouse=True)
def isolate_environment(tmp_path, monkeypatch):
    """No test may touch the real Trash or the real library."""
    monkeypatch.setenv("CRATE_TRASH_DIR", str(tmp_path / "trash"))
    monkeypatch.delenv("CRATE_LIBRARY", raising=False)
    yield


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under a folder, with its contents, for proving nothing changed."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class ManualStream:
    """An output stream that does nothing on its own; tests drive the callback themselves."""
    latency = 0.0

    def __init__(self, samplerate, callback):
        self.samplerate, self.callback = samplerate, callback
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True
