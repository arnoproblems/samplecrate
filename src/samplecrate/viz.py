"""Sound visualisation as pure functions: audio data in, character grids and colour indexes out.

Nothing here knows about the terminal UI, which keeps it testable and lets `crate show`
draw the same waveform straight to stdout.
"""

from __future__ import annotations

import numpy as np

from samplecrate.audio import band_edges, band_power

# Braille cells are 2 dots wide and 4 tall; each dot has its own bit in the codepoint.
BRAILLE_BASE = 0x2800
_BRAILLE_BITS = np.array([[0x01, 0x08],
                          [0x02, 0x10],
                          [0x04, 0x20],
                          [0x40, 0x80]])
BAR_GLYPHS = " ▁▂▃▄▅▆▇█"
SPECTRUM_FLOOR_DB = -78.0


def braille_rows(dots: np.ndarray) -> list[str]:
    """Convert a boolean dot matrix (rows*4, cols*2) into rows of braille characters."""
    rows, cols = dots.shape[0] // 4, dots.shape[1] // 2
    if rows == 0 or cols == 0:
        return []
    cells = dots[:rows * 4, :cols * 2].reshape(rows, 4, cols, 2).transpose(0, 2, 1, 3)
    codes = (cells * _BRAILLE_BITS).sum(axis=(2, 3))
    return ["".join(map(chr, BRAILLE_BASE + row)) for row in codes]


def _rebin(values: np.ndarray, count: int, how) -> np.ndarray:
    """Reduce (or stretch) an array to `count` values using `how` across each span."""
    if len(values) == 0:
        return np.zeros(count)
    if len(values) <= count:
        index = np.floor(np.linspace(0, len(values), count, endpoint=False)).astype(int)
        return values[index]
    edges = np.linspace(0, len(values), count + 1).astype(int)
    return how.reduceat(values, edges[:-1])


def wave_dots(mins: np.ndarray, maxs: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """A mirrored min/max waveform as a dot matrix, scaled to the clip's own peak."""
    dot_cols, dot_rows = cols * 2, rows * 4
    low = _rebin(mins, dot_cols, np.minimum)
    high = _rebin(maxs, dot_cols, np.maximum)
    # Don't amplify near-silence into a full-height wave
    scale = max(float(np.abs(mins).max(initial=0)), float(np.abs(maxs).max(initial=0)), 0.02)
    centre = (dot_rows - 1) / 2
    top = np.round(centre - high / scale * centre).astype(int)
    bottom = np.round(centre - low / scale * centre).astype(int)
    grid = np.arange(dot_rows)[:, None]
    return (grid >= np.minimum(top, bottom)) & (grid <= np.maximum(top, bottom))


def wave_cells(mins: np.ndarray, maxs: np.ndarray, rms: np.ndarray, cols: int, rows: int):
    """A solid waveform using block characters, which give 8 vertical steps per cell.

    Returns (rows of text, invert, core). A True in `invert` means swap foreground and background
    for that cell: Unicode only has blocks that fill from the bottom, so the lower half of the wave
    draws the inverse of the shape it wants. `core` marks cells inside the RMS band, which the
    visualizer draws brighter, like the two-tone waveforms in a DAW.
    """
    low = _rebin(mins, cols, np.minimum)
    high = _rebin(maxs, cols, np.maximum)
    loud = _rebin(rms, cols, np.maximum)
    # Don't amplify near-silence into a full-height wave
    scale = max(float(np.abs(mins).max(initial=0)), float(np.abs(maxs).max(initial=0)), 0.02)
    eighths = rows * 8
    centre = eighths / 2
    top = np.floor(centre - np.clip(high, -scale, scale) / scale * centre)
    bottom = np.ceil(centre - np.clip(low, -scale, scale) / scale * centre)
    # Even pure silence draws a thin centre line
    top = np.minimum(top, np.floor(centre - 0.5))
    bottom = np.maximum(bottom, np.ceil(centre + 0.5))

    cell_top = (np.arange(rows)[:, None] * 8).astype(float)
    start = np.maximum(top[None, :], cell_top)
    end = np.minimum(bottom[None, :], cell_top + 8)
    filled = np.clip(end - start, 0, 8).astype(int)
    touches_bottom = end >= cell_top + 8
    touches_top = start <= cell_top
    invert = (filled > 0) & (filled < 8) & touches_top & ~touches_bottom
    floating = (filled > 0) & (filled < 8) & ~touches_top & ~touches_bottom

    lines = []
    for r in range(rows):
        chars = []
        for c in range(cols):
            f = int(filled[r, c])
            if floating[r, c]:
                chars.append("─" if f <= 3 else "━" if f <= 5 else "█")
            elif invert[r, c]:
                chars.append(BAR_GLYPHS[8 - f])
            else:
                chars.append(BAR_GLYPHS[f])
        lines.append("".join(chars))
    middle = cell_top + 4
    core = np.abs(middle - centre) <= (loud[None, :] / scale * centre)
    return lines, invert, core


def scope_window(mono: np.ndarray, position: int, span: int) -> np.ndarray:
    """A window of audio starting at a rising zero crossing, so a steady tone stands still on screen."""
    total = len(mono)
    if total == 0 or span <= 0:
        return np.zeros(max(span, 0), dtype="float32")
    start = min(max(position, 0), max(total - 1, 0))
    search = mono[start:start + span]
    crossings = np.where((search[:-1] <= 0) & (search[1:] > 0))[0]
    if len(crossings):
        start += int(crossings[0])
    window = mono[start:start + span]
    if len(window) < span:
        window = np.pad(window, (0, span - len(window)))
    return window


def scope_dots(window: np.ndarray, cols: int, rows: int, scale: float) -> np.ndarray:
    """An oscilloscope trace as a dot matrix, joined up so steep edges don't break the line."""
    dot_cols, dot_rows = cols * 2, rows * 4
    dots = np.zeros((dot_rows, dot_cols), dtype=bool)
    if dot_cols == 0 or dot_rows == 0 or len(window) == 0:
        return dots
    index = np.linspace(0, len(window) - 1, dot_cols).astype(int)
    centre = (dot_rows - 1) / 2
    y = np.round(centre - np.clip(window[index] / max(scale, 1e-3), -1, 1) * centre).astype(int)
    previous = np.concatenate([y[:1], y[:-1]])
    grid = np.arange(dot_rows)[:, None]
    return (grid >= np.minimum(y, previous)) & (grid <= np.maximum(y, previous))


def spectrum_levels(window: np.ndarray, sr: int, bars: int) -> np.ndarray:
    """Log-spaced spectrum bars from 0 to 1, for a window of audio."""
    if bars <= 0:
        return np.zeros(0)
    size = len(window)
    if size < 16:
        return np.zeros(bars)
    power = np.abs(np.fft.rfft(window * np.hanning(size))) ** 2 / (size ** 2 / 16)
    bands = band_power(power, band_edges(bars, sr, size))
    # +3 dB per octave to compensate for how music's energy falls with frequency
    tilt = np.geomspace(1.0, 18000 / 30, bars)
    db = 10 * np.log10(bands * tilt + 1e-12)
    return np.clip((db - SPECTRUM_FLOOR_DB) / -SPECTRUM_FLOOR_DB, 0, 1)


def resample_levels(values: np.ndarray, count: int) -> np.ndarray:
    """Stretch a band array to a new number of bars."""
    if count <= 0 or len(values) == 0:
        return np.zeros(max(count, 0))
    return np.interp(np.linspace(0, len(values) - 1, count), np.arange(len(values)), values)


def bar_glyph(level: float, row_from_bottom: int, rows: int) -> str:
    """The block character for one row of a vertical bar."""
    filled = level * rows
    if filled >= row_from_bottom + 1:
        return BAR_GLYPHS[8]
    fraction = filled - row_from_bottom
    return BAR_GLYPHS[int(fraction * 8)] if fraction > 0 else " "


def gram_pixels(gram: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Resample a spectrogram to rows*2 by cols, with high frequencies on top. Half-block cells show two pixels each."""
    pixel_rows = rows * 2
    if gram.size == 0 or cols == 0 or pixel_rows == 0:
        return np.zeros((pixel_rows, cols))
    band_index = np.linspace(gram.shape[0] - 1, 0, pixel_rows).astype(int)
    frame_index = np.floor(np.linspace(0, gram.shape[1], cols, endpoint=False)).astype(int)
    return gram[np.ix_(band_index, frame_index)]


def format_time(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}"


def ruler(duration: float, width: int, bpm: int | None) -> tuple[str, str]:
    """Tick marks and labels under a waveform: bars when the tempo is known, otherwise time. Returns (ruler, unit)."""
    if width <= 0:
        return "", ""
    line = [" "] * width
    if duration <= 0:
        return "".join(line), ""
    bar_seconds = 240.0 / bpm if bpm else None
    bars = duration / bar_seconds if bar_seconds else 0
    if bar_seconds and 1 <= bars <= 512:
        step = _nice_step(bars, width, (1, 2, 4, 8, 16, 32, 64, 128))
        unit = "bars"
        for n in np.arange(0, bars + 1e-6, step):
            _stamp(line, n * bar_seconds / duration, f"{int(n) + 1}", width)
        return "".join(line), unit
    step = _nice_step(duration, width, (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120))
    for t in np.arange(0, duration + 1e-9, step):
        label = f"{t:g}" if step < 1 or t < 60 else f"{int(t // 60)}:{int(t % 60):02d}"
        _stamp(line, t / duration, label, width)
    return "".join(line), "sec"


def _nice_step(span: float, width: int, steps: tuple) -> float:
    """The smallest step whose labels fit about every 8 columns."""
    wanted = span * 8 / max(width, 1)
    for step in steps:
        if step >= wanted:
            return step
    return steps[-1]


def _stamp(line: list[str], fraction: float, label: str, width: int) -> None:
    x = int(fraction * width)
    text = "╵" + label
    if x + len(text) > width or any(c != " " for c in line[max(x - 1, 0):x + len(text)]):
        return
    line[x:x + len(text)] = text
