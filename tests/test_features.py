import os
import time

import numpy as np
import pytest
import soundfile as sf

from conftest import UNSORTED, write_wav
from samplecrate import features as fx
from samplecrate.library import Library

SR = 44100


def write(path, signal, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal.astype("float32"), sr)
    return path


def sine(hz, seconds, level=0.5):
    return level * np.sin(2 * np.pi * hz * np.arange(int(SR * seconds)) / SR)


def clicks(rate, seconds=8):
    out = np.zeros(int(SR * seconds))
    n = int(0.01 * SR)
    for i in range(int(seconds * rate)):
        start = int(i / rate * SR)
        out[start:start + n] += np.random.default_rng(i).normal(0, 0.5, n) * np.exp(-np.arange(n) / 80)
    return out


@pytest.mark.parametrize("hz,band", [(60, fx.LOW), (200, fx.LOW), (1000, fx.MID), (3000, fx.MID), (6000, fx.HIGH), (12000, fx.HIGH)])
def test_center_and_range_of_pure_tones(tmp_path, hz, band):
    result = fx.measure(write(tmp_path / "a.wav", sine(hz, 1)))
    assert result.center_hz == pytest.approx(hz, rel=0.05)
    assert result.range == band


def test_range_boundaries():
    assert fx.range_of(249.9) == fx.LOW and fx.range_of(250) == fx.MID
    assert fx.range_of(3999) == fx.MID and fx.range_of(4000) == fx.HIGH


def test_energy_span_reaches_both_tones(tmp_path):
    signal = sine(100, 2) + sine(5000, 2)
    result = fx.measure(write(tmp_path / "a.wav", signal))
    assert result.low_hz < 150 and result.high_hz > 4500


def test_audible_length_ignores_silence(tmp_path):
    signal = np.concatenate([np.zeros(SR), sine(440, 0.5), np.zeros(SR * 3)])
    result = fx.measure(write(tmp_path / "a.wav", signal))
    assert result.audible == pytest.approx(0.5, abs=0.06)


def test_audible_length_of_a_full_sound(tmp_path):
    assert fx.measure(write(tmp_path / "a.wav", sine(440, 2))).audible == pytest.approx(2, abs=0.05)


def test_decay_orders_short_and_long(tmp_path):
    t = np.arange(SR * 2) / SR
    fast = sine(100, 2) * np.exp(-40 * t)
    slow = sine(100, 2) * np.exp(-4 * t)
    a, b = fx.measure(write(tmp_path / "a.wav", fast)), fx.measure(write(tmp_path / "b.wav", slow))
    assert a.decay < 0.15 < 0.4 < b.decay


@pytest.mark.parametrize("rate", [1, 2, 4, 8, 12])
def test_busy_counts_real_onsets(tmp_path, rate):
    assert fx.measure(write(tmp_path / "a.wav", clicks(rate))).busy == pytest.approx(rate, abs=rate * 0.2 + 0.2)


def test_sustained_tone_is_not_busy(tmp_path):
    """Regression: small fluctuations in a held tone used to count as a stream of hits."""
    beating = sine(220, 8, 0.4) + sine(223, 8, 0.2)
    assert fx.measure(write(tmp_path / "a.wav", beating)).busy < 0.5


def test_movement_separates_steady_from_sweeping(tmp_path):
    chord = sum(sine(h, 8, 0.1) for h in (110, 220, 330, 440, 660))
    freq = np.geomspace(100, 8000, SR * 8)
    sweep = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    still, moving = fx.measure(write(tmp_path / "a.wav", chord)), fx.measure(write(tmp_path / "b.wav", sweep))
    assert still.movement < 0.05
    assert moving.movement > 2


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"])
def test_every_bit_depth_measures_alike(tmp_path, subtype):
    """Regression: reading a float file as integers truncated it to silence, and measured nonsense."""
    path = tmp_path / "a.wav"
    sf.write(path, (sine(440, 1) * np.exp(-2 * np.arange(SR) / SR)).astype("float32"), SR, subtype=subtype)
    result = fx.measure(path)
    assert result.center_hz == pytest.approx(440, rel=0.03)
    assert fx._read_mono(path)[0].max() > 0.3
    assert result.audio_hash == fx.measure(path).audio_hash


def test_padding_with_silence_changes_nothing_about_the_sound(tmp_path):
    t = np.arange(int(SR * 0.8)) / SR
    hit = np.convolve(np.random.default_rng(3).normal(0, 0.5, len(t)), np.ones(30) / 30, mode="same") * np.exp(-5 * t) + sine(90, 0.8) * np.exp(-5 * t)
    padded = np.concatenate([np.zeros(4410), hit, np.zeros(SR)])
    a, b = fx.measure(write(tmp_path / "a.wav", hit)), fx.measure(write(tmp_path / "b.wav", padded))
    assert b.center_hz == pytest.approx(a.center_hz, rel=0.03)
    assert b.audible == pytest.approx(a.audible, abs=0.02)
    assert np.abs(np.array(a.signature) - np.array(b.signature)).mean() < 0.1


def test_silence_and_tiny_files_dont_crash(tmp_path):
    for frames in (1, 50, 2000, SR):
        result = fx.measure(write(tmp_path / f"s{frames}.wav", np.zeros(frames)))
        assert result.center_hz == 0
    assert fx.measure(write(tmp_path / "t.wav", sine(440, 0.01))).audible > 0


def test_empty_and_garbage_files_raise(tmp_path):
    empty = write(tmp_path / "e.wav", np.zeros(0))
    with pytest.raises(ValueError):
        fx.measure(empty)
    junk = tmp_path / "j.wav"
    junk.write_bytes(b"nope")
    with pytest.raises(Exception):
        fx.measure(junk)


def test_near_duplicates_catch_copies_and_spare_different_sounds(tmp_path):
    t = np.arange(SR) / SR
    kick = sine(60, 1) * np.exp(-6 * t)
    a = fx.measure(write(tmp_path / "a.wav", kick))
    quieter = fx.measure(write(tmp_path / "b.wav", kick * 0.5))
    other = fx.measure(write(tmp_path / "c.wav", sine(120, 1) * np.exp(-6 * t)))
    shorter = fx.measure(write(tmp_path / "d.wav", sine(60, 0.3) * np.exp(-20 * np.arange(int(SR * 0.3)) / SR)))
    assert fx.are_near_duplicates(a, quieter)
    assert not fx.are_near_duplicates(a, other)
    assert not fx.are_near_duplicates(a, shorter)


def test_neighboring_notes_are_not_near_duplicates(tmp_path):
    """Regression: notes close enough to share a signature band still sound different."""
    t = np.arange(SR) / SR
    envelope = np.exp(-6 * t)
    c, d = fx.measure(write(tmp_path / "c.wav", sine(65.4, 1) * envelope)), fx.measure(write(tmp_path / "d.wav", sine(73.4, 1) * envelope))
    assert not fx.are_near_duplicates(c, d)


# ---------------------------------------------------------------------- cache and analyzer

def test_analyzer_measures_and_caches(library_root):
    lib = Library(library_root).scan()
    analyzer = fx.Analyzer(lib)
    assert analyzer.run() == len(lib.samples)
    assert all(s.features for s in lib.samples)
    fresh = Library(library_root).scan()
    assert all(s.features for s in fresh.samples)
    assert fx.Analyzer(fresh).run() == 0


def test_changed_file_is_measured_again(library_root):
    lib = Library(library_root).scan()
    fx.Analyzer(lib).run()
    target = lib.samples[0]
    write_wav(target.path, 0.9, hz=500)
    os.utime(target.path, (time.time() + 5, time.time() + 5))
    fresh = Library(library_root).scan()
    assert fresh.by_rel[target.rel].features is None
    assert fx.Analyzer(fresh).run() == 1


def test_version_change_discards_the_cache(library_root):
    lib = Library(library_root).scan()
    fx.Analyzer(lib).run()
    path = library_root / ".crate" / fx.CACHE_FILE
    path.write_text(path.read_text().replace(f'"version":{fx.CACHE_VERSION}', '"version":-1'))
    assert all(s.features is None for s in Library(library_root).scan().samples)


def test_corrupt_cache_is_ignored(library_root):
    (library_root / ".crate").mkdir()
    (library_root / ".crate" / fx.CACHE_FILE).write_text("{ not json")
    assert all(s.features is None for s in Library(library_root).scan().samples)


def test_read_only_measures_in_memory_and_writes_nothing(library_root):
    lib = Library(library_root, read_only=True).scan()
    fx.Analyzer(lib).run()
    assert all(s.features for s in lib.samples)
    assert not (library_root / ".crate").exists()
    lib.scan()
    assert all(s.features for s in lib.samples)    # still known after a rescan in the same session


def test_unreadable_file_is_reported_not_fatal(library_root):
    bad = library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.10s__bad.wav"
    bad.write_bytes(b"not audio")
    lib = Library(library_root).scan()
    analyzer = fx.Analyzer(lib)
    analyzer.run()
    assert analyzer.failed == [bad.relative_to(library_root).as_posix()]
    assert lib.by_rel[analyzer.failed[0]].features is None


def test_stop_ends_the_work_early(library_root):
    lib = Library(library_root).scan()
    analyzer = fx.Analyzer(lib, workers=1)
    analyzer.stop()
    assert analyzer.run() == 0


def test_progress_reaches_the_total(library_root):
    lib = Library(library_root).scan()
    seen = []
    fx.Analyzer(lib, workers=1).run(lambda done, total: seen.append((done, total)))
    assert seen[-1] == (len(lib.samples), len(lib.samples))


def test_measurements_follow_a_rename_and_an_undo(library_root):
    from samplecrate import organize
    lib = Library(library_root).scan()
    fx.Analyzer(lib).run()
    rel = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav"
    before = lib.by_rel[rel].features
    moves = organize.retag(lib, [lib.by_rel[rel]], organize.Changes(desc="other"))
    assert lib.by_rel[moves[0].dst].features == before
    assert Library(library_root).scan().by_rel[moves[0].dst].features == before
    organize.undo(lib)
    assert lib.by_rel[rel].features == before
