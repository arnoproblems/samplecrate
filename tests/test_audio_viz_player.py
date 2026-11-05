import time

import numpy as np
import pytest
import soundfile as sf

from conftest import ManualStream, tone, write_wav
from samplecrate import audio, viz
from samplecrate.player import Player


# ---------------------------------------------------------------------- decoding

@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_load_clip_formats(tmp_path, subtype):
    clip = audio.load_clip(write_wav(tmp_path / "a.wav", 0.5, subtype=subtype))
    assert clip.audio.shape[1] == 2 and clip.sr == 22050
    assert abs(clip.duration - 0.5) < 0.001
    assert clip.peak > 0.5


def test_load_mono_is_duplicated_for_playback(tmp_path):
    path = tmp_path / "m.wav"
    sf.write(path, tone(0.2)[:, 0], 22050)
    clip = audio.load_clip(path)
    assert clip.channels == 1 and clip.audio.shape[1] == 2


def test_load_multichannel_keeps_first_two(tmp_path):
    path = tmp_path / "s.wav"
    sf.write(path, np.zeros((1000, 6), dtype="float32"), 22050)
    assert audio.load_clip(path).audio.shape[1] == 2


def test_tiny_and_silent_clips_load(tmp_path):
    for frames in (1, 3, 100):
        path = tmp_path / f"t{frames}.wav"
        sf.write(path, np.zeros((frames, 2), dtype="float32"), 44100)
        clip = audio.load_clip(path)
        assert clip.peak == 0 and clip.gram.shape[0] == audio.GRAM_BANDS


def test_empty_and_garbage_files_raise_cleanly(tmp_path):
    empty = tmp_path / "e.wav"
    sf.write(empty, np.zeros((0, 2), dtype="float32"), 44100)
    with pytest.raises(ValueError):
        audio.load_clip(empty)
    junk = tmp_path / "j.wav"
    junk.write_bytes(b"this is not audio at all")
    with pytest.raises((ValueError, RuntimeError, sf.LibsndfileError)):
        audio.load_clip(junk)


def test_peaks_bound_the_signal():
    mono = tone(1.0)[:, 0]
    mins, maxs, rms = audio.compute_peaks(mono)
    assert mins.min() == mono.min() and maxs.max() == mono.max()
    assert (rms >= 0).all() and len(mins) == audio.PEAK_BUCKETS


def test_spectrogram_finds_the_right_frequency():
    sr = 44100
    t = np.arange(sr) / sr
    gram, mean = audio.compute_spectrogram((0.8 * np.sin(2 * np.pi * 1000 * t)).astype("float32"), sr)
    hz = np.geomspace(30, 18000, audio.GRAM_BANDS + 1)
    peak_band = int(mean.argmax())
    # One band's width of slack, since the tone sits close to a bin edge
    assert hz[peak_band - 1] <= 1000 <= hz[peak_band + 2]
    assert gram.min() >= 0 and gram.max() <= 1


def test_cache_returns_same_clip_and_evicts(tmp_path):
    a, b = write_wav(tmp_path / "a.wav", 0.5), write_wav(tmp_path / "b.wav", 0.5)
    cache = audio.ClipCache(max_bytes=1)
    clip_a = cache.get(a)
    assert cache.get(a) is clip_a
    cache.get(b)
    assert cache.get(a) is not clip_a


def test_cache_forget_rereads_changed_file(tmp_path):
    a = write_wav(tmp_path / "a.wav", 0.5)
    cache = audio.ClipCache()
    first = cache.get(a)
    cache.forget(a)
    assert cache.get(a) is not first


# ---------------------------------------------------------------------- visualisation

def clip_arrays(seconds=1.0):
    mono = tone(seconds)[:, 0]
    return audio.compute_peaks(mono) + (mono,)


def test_braille_bit_layout():
    dots = np.zeros((4, 2), dtype=bool)
    assert viz.braille_rows(dots) == ["⠀"]
    dots[0, 0] = True
    assert viz.braille_rows(dots) == ["⠁"]
    dots[:] = True
    assert viz.braille_rows(dots) == ["⣿"]


@pytest.mark.parametrize("cols,rows", [(1, 1), (2, 1), (1, 3), (80, 6), (80, 7), (300, 40)])
def test_wave_cells_shapes(cols, rows):
    mins, maxs, rms, _ = clip_arrays()
    lines, invert, core = viz.wave_cells(mins, maxs, rms, cols, rows)
    assert len(lines) == rows and all(len(line) == cols for line in lines)
    assert invert.shape == (rows, cols) and core.shape == (rows, cols)


def test_wave_cells_loud_signal_fills_the_height():
    mins, maxs = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    lines, invert, _ = viz.wave_cells(mins, maxs, np.array([1.0, 1.0]), 2, 4)
    assert lines == ["██"] * 4 and not invert.any()


def test_wave_cells_silence_is_a_centre_line_not_a_full_wave():
    zero = np.zeros(10)
    for rows in (4, 5):
        lines, _, _ = viz.wave_cells(zero, zero, zero, 10, rows)
        assert any(line.strip() for line in lines)
        assert not any("█" in line for line in lines)


def test_wave_cells_positive_only_is_top_half():
    mins, maxs = np.zeros(4), np.ones(4)
    lines, _, _ = viz.wave_cells(mins, maxs, maxs, 4, 4)
    assert lines[0] == "████" and lines[1] == "████"
    assert not lines[3].strip()


def test_wave_cells_empty_input():
    lines, _, _ = viz.wave_cells(np.zeros(0), np.zeros(0), np.zeros(0), 5, 2)
    assert len(lines) == 2


def test_scope_window_starts_at_rising_zero_crossing():
    sr = 44100
    t = np.arange(sr) / sr
    mono = np.sin(2 * np.pi * 220 * t).astype("float32")
    window = viz.scope_window(mono, 137, 1024)
    assert len(window) == 1024
    assert abs(window[0]) < 0.05 and window[1] > window[0]


@pytest.mark.parametrize("position", [-5, 0, 10, 10**9])
def test_scope_window_handles_any_position(position):
    assert len(viz.scope_window(np.zeros(100, dtype="float32"), position, 256)) == 256


def test_scope_dots_are_connected_and_centred():
    window = np.sin(np.linspace(0, 12, 512))
    dots = viz.scope_dots(window, 40, 5, 1.0)
    assert dots.shape == (20, 80)
    assert dots.any(axis=0).all()      # every column has a trace, with no gaps
    flat = viz.scope_dots(np.zeros(100), 10, 2, 1.0)
    assert flat.sum() == 20            # silence is one flat line


def test_spectrum_levels_peak_at_the_right_bar():
    sr = 44100
    t = np.arange(4096) / sr
    levels = viz.spectrum_levels((np.sin(2 * np.pi * 1000 * t)).astype("float32"), sr, 40)
    assert len(levels) == 40 and levels.max() <= 1
    edges = np.geomspace(30, 18000, 41)
    bar = int(levels.argmax())
    assert edges[bar - 1] <= 1000 <= edges[bar + 2]


@pytest.mark.parametrize("hz", [60, 200, 1000, 5000, 12000])
def test_spectrum_axis_is_truly_logarithmic(hz):
    """The labels on the spectrum assume log spacing, so a tone must land where its label is."""
    sr = 44100
    t = np.arange(4096) / sr
    levels = viz.spectrum_levels(np.sin(2 * np.pi * hz * t).astype("float32"), sr, 64)
    expected = np.log(hz / 30) / np.log(18000 / 30) * 64
    assert abs(int(levels.argmax()) - expected) <= 2.5


def test_spectrum_levels_silence_and_degenerate():
    assert viz.spectrum_levels(np.zeros(4096), 44100, 10).max() == 0
    assert len(viz.spectrum_levels(np.zeros(4), 44100, 10)) == 10
    assert len(viz.spectrum_levels(np.zeros(4096), 44100, 0)) == 0


def test_low_sample_rate_spectrum_is_safe():
    assert len(viz.spectrum_levels(np.zeros(4096), 8000, 60)) == 60


def test_bar_glyph():
    assert viz.bar_glyph(1.0, 0, 4) == "█" and viz.bar_glyph(1.0, 3, 4) == "█"
    assert viz.bar_glyph(0.0, 0, 4) == " "
    assert viz.bar_glyph(0.5, 2, 4) == " "
    assert viz.bar_glyph(0.5, 1, 4) == "█"
    assert viz.bar_glyph(0.125, 0, 4) == "▄"


def test_gram_pixels_puts_high_frequencies_on_top():
    gram = np.zeros((96, 100))
    gram[-1, :] = 1.0
    pixels = viz.gram_pixels(gram, 20, 3)
    assert pixels.shape == (6, 20)
    assert pixels[0].max() == 1.0 and pixels[-1].max() == 0.0


def test_ruler_uses_bars_when_tempo_is_known():
    line, unit = viz.ruler(8.0, 80, 120)    # 4 bars at 120 BPM
    assert unit == "bars" and len(line) == 80
    assert "╵1" in line and "╵4" in line


def test_ruler_uses_time_without_tempo():
    line, unit = viz.ruler(2.0, 80, None)
    assert unit == "sec" and len(line) == 80 and "╵0" in line


@pytest.mark.parametrize("duration,width,bpm", [(0.01, 10, None), (0, 10, None), (300, 5, 60), (5, 1, 120), (1, 0, None)])
def test_ruler_never_overflows(duration, width, bpm):
    assert len(viz.ruler(duration, width, bpm)[0]) == width


def test_format_time():
    assert viz.format_time(0) == "0:00.0"
    assert viz.format_time(65.5) == "1:05.5"


# ---------------------------------------------------------------------- playback

def make_player(volume=1.0):
    player = Player(ManualStream, volume=volume)
    return player


def ready(player):
    deadline = time.monotonic() + 2
    while player._waiting is not None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert player._waiting is None


def pull(player, frames=512):
    out = np.zeros((frames, 2), dtype="float32")
    player._callback(out, frames, None, None)
    return out


def ramp(frames):
    return np.tile(np.linspace(0, 1, frames, dtype="float32")[:, None], (1, 2))


def test_play_never_blocks_on_a_slow_device():
    def slow(sr, cb):
        time.sleep(0.3)
        return ManualStream(sr, cb)
    player = Player(slow)
    started = time.perf_counter()
    player.play(np.zeros((1000, 2), dtype="float32"), 44100)
    assert time.perf_counter() - started < 0.05
    assert player.is_playing       # counts as playing while waiting for the device
    ready(player)
    assert player._voice is not None
    player.close()


def test_output_matches_input_at_full_volume():
    player = make_player(1.0)
    data = ramp(2048)
    player.play(data, 44100)
    ready(player)
    out = pull(player, 1024)
    np.testing.assert_array_almost_equal(out, data[:1024], decimal=5)
    player.close()


def test_attack_is_not_softened():
    """A new sound must start at full level: a fade-in would blunt every kick."""
    player = make_player(0.5)
    player.play(np.ones((1024, 2), dtype="float32"), 44100)
    ready(player)
    assert pull(player, 16)[0, 0] == pytest.approx(0.5)
    player.close()


def test_non_loop_ends_and_pads_with_silence():
    player = make_player()
    player.play(np.ones((100, 2), dtype="float32"), 44100)
    ready(player)
    out = pull(player, 512)
    assert out[99, 0] == pytest.approx(1.0) and out[100:].max() == 0
    assert not player.is_playing and player.position == 0
    player.close()


def test_loop_is_seamless():
    player = make_player()
    data = ramp(100)
    player.play(data, 44100, loop=True)
    ready(player)
    out = pull(player, 350)
    np.testing.assert_array_almost_equal(out[:100], data)
    np.testing.assert_array_almost_equal(out[100:200], data)
    np.testing.assert_array_almost_equal(out[300:350], data[:50])
    assert player.is_playing
    player.close()


def test_loop_shorter_than_a_block_keeps_going():
    player = make_player()
    player.play(np.ones((3, 2), dtype="float32"), 44100, loop=True)
    ready(player)
    assert pull(player, 512).min() == pytest.approx(1.0)
    player.close()


def test_switching_clips_fades_old_one_without_a_click():
    player = make_player()
    player.play(np.ones((5000, 2), dtype="float32"), 44100)
    ready(player)
    pull(player)
    player.play(np.zeros((5000, 2), dtype="float32"), 44100)
    out = pull(player, 512)
    assert out[0, 0] > 0.9 and out[-1, 0] < 0.01     # ramps down rather than cutting off
    assert np.all(np.diff(out[:, 0]) <= 1e-6)
    player.close()


def test_stop_fades_and_resets_position():
    player = make_player()
    player.play(np.ones((44100, 2), dtype="float32"), 44100)
    ready(player)
    pull(player, 4410)
    player.stop()
    assert not player.is_playing and player.position == 0
    out = pull(player, 512)
    assert out[0, 0] > 0.9 and out[-1, 0] < 0.01
    assert pull(player, 512).max() == 0
    player.close()


def test_pause_remembers_position():
    player = make_player()
    player.play(np.ones((44100, 2), dtype="float32"), 44100)
    ready(player)
    pull(player, 4410)
    player.pause()
    assert not player.is_playing and player.position == pytest.approx(0.1)
    player.close()


def test_pause_while_waiting_for_device_cancels_the_sound():
    def slow(sr, cb):
        time.sleep(0.1)
        return ManualStream(sr, cb)
    player = Player(slow)
    player.play(np.ones((1000, 2), dtype="float32"), 44100)
    player.pause()
    assert not player.is_playing
    time.sleep(0.2)
    assert player._voice is None    # the device opening later doesn't start anything
    player.close()


def test_seek_while_playing_and_while_stopped():
    player = make_player()
    data = ramp(44100)
    player.play(data, 44100)
    ready(player)
    pull(player, 512)
    player.seek(0.5)
    out = pull(player, 16)
    assert out[0, 0] == pytest.approx(data[22050, 0], abs=1e-3)
    player.stop()
    player.seek(0.25)
    assert player.position == pytest.approx(0.25)
    player.close()


def test_seek_is_clamped():
    player = make_player()
    player.play(np.ones((1000, 2), dtype="float32"), 44100)
    ready(player)
    player.seek(-5)
    assert pull(player, 2)[0, 0] == pytest.approx(1.0)
    player.seek(999)        # lands on the final frame, so the clip then ends normally
    out = pull(player, 4)
    assert out[0, 0] == pytest.approx(1.0) and out[1, 0] == 0
    assert not player.is_playing
    player.close()


def test_volume_clamps_and_ramps_smoothly():
    player = make_player(1.0)
    player.play(np.ones((44100, 2), dtype="float32"), 44100)
    ready(player)
    pull(player)
    player.volume = 5
    assert player.volume == 1
    player.volume = -1
    assert player.volume == 0
    out = pull(player, 512)
    assert out[0, 0] == pytest.approx(1.0) and out[-1, 0] == pytest.approx(0.0)
    assert np.all(np.diff(out[:, 0]) <= 1e-6)
    assert pull(player, 16).max() == 0
    player.close()


def test_position_accounts_for_output_latency():
    class Latent(ManualStream):
        latency = 0.1
    player = Player(Latent, volume=1.0)
    player.play(np.ones((44100, 2), dtype="float32"), 44100)
    ready(player)
    pull(player, 22050)
    assert player.position == pytest.approx(0.4, abs=0.02)
    player.close()


def test_position_wraps_just_after_a_loop_restarts():
    class Latent(ManualStream):
        latency = 0.1
    player = Player(Latent, volume=1.0)
    player.play(np.ones((44100, 2), dtype="float32"), 44100, loop=True)
    ready(player)
    pull(player, 44100 + 441)
    assert player.position == pytest.approx(0.91, abs=0.02)    # speakers are still on the end of the clip
    player.close()


def test_changing_sample_rate_reopens_the_stream():
    streams = []
    def factory(sr, cb):
        streams.append(ManualStream(sr, cb))
        return streams[-1]
    player = Player(factory)
    player.play(np.ones((1000, 2), dtype="float32"), 44100)
    ready(player)
    player.play(np.ones((1000, 2), dtype="float32"), 44100)
    assert len(streams) == 1
    player.play(np.ones((1000, 2), dtype="float32"), 48000)
    ready(player)
    assert [s.samplerate for s in streams] == [44100, 48000]
    assert streams[0].closed
    player.close()


def test_broken_device_falls_back_to_silent_and_reports():
    def broken(sr, cb):
        raise RuntimeError("no device here")
    player = Player(broken)
    player.play(np.zeros((44100, 2), dtype="float32"), 44100)
    ready(player)
    assert player.error == "no device here"
    assert player.is_playing
    time.sleep(0.15)
    assert player.position > 0    # the silent stream keeps time, so the visuals still run
    player.close()


def test_silent_stream_runs_in_real_time():
    player = Player(volume=0.0)
    player.play(np.zeros((44100, 2), dtype="float32"), 44100)
    ready(player)
    start = player.position
    time.sleep(0.3)
    assert 0.2 < player.position - start < 0.45
    player.close()


def test_empty_audio_is_ignored():
    player = make_player()
    player.play(np.zeros((0, 2), dtype="float32"), 44100)
    assert not player.is_playing
    player.close()
