"""Colours shared by the sample table, the details panel and the visualizer."""

from __future__ import annotations

import colorsys
import zlib
from functools import lru_cache

from rich.style import Style

from samplecrate import naming

CATEGORY_COLORS = {
    "Drums": "#ff9e64",
    "Percs": "#e0af68",
    "Bass": "#f7768e",
    "Synths": "#7dcfff",
    "Instruments": "#9ece6a",
    "Melodic": "#bb9af7",
    "Vocals": "#ff7eb6",
    "FX": "#73daca",
    "Unsorted": "#ff5f5f",
}
_EXTRA_COLORS = ("#c0caf5", "#2ac3de", "#b4f9f8", "#ffc777", "#c3e88d", "#fca7ea")
FAVORITE_COLOR = "#ffc777"
MARK_COLOR = "#7aa2f7"
WARNING_COLOR = "#ff5f5f"
MUTED_COLOR = "#7a809a"

# Black to white through purple and orange, for the spectrogram
_HEAT_STOPS = ("#0b0a1a", "#2b1163", "#6a1b8a", "#b5316f", "#e8593f", "#fca636", "#fdf0a5")
# Green at the bottom of a spectrum bar, red at the top
_METER_STOPS = ("#2ac3de", "#73daca", "#9ece6a", "#e0af68", "#ff9e64", "#f7768e")


def category_color(main: str) -> str:
    if main in CATEGORY_COLORS:
        return CATEGORY_COLORS[main]
    return _EXTRA_COLORS[zlib.crc32(main.encode()) % len(_EXTRA_COLORS)]


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def blend(a: str, b: str, t: float) -> str:
    """Mix two hex colours: t=0 gives a and t=1 gives b."""
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return f"#{round(ra + (rb - ra) * t):02x}{round(ga + (gb - ga) * t):02x}{round(ba + (bb - ba) * t):02x}"


def _ramp(stops: tuple[str, ...], size: int) -> list[str]:
    out = []
    for i in range(size):
        x = i / (size - 1) * (len(stops) - 1)
        low = min(int(x), len(stops) - 2)
        out.append(blend(stops[low], stops[low + 1], x - low))
    return out


HEAT_RAMP = _ramp(_HEAT_STOPS, 64)
METER_RAMP = _ramp(_METER_STOPS, 32)


@lru_cache(maxsize=4096)
def key_color(key: str) -> str:
    """Hue by position on the Camelot wheel, so keys that mix well look alike."""
    full = key if naming.key_mode(key) else key + "min"
    number = int(naming.camelot(full)[:-1])
    red, green, blue = colorsys.hls_to_rgb((number - 1) / 12, 0.72, 0.65)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


@lru_cache(maxsize=8192)
def compose(base: Style, color: str | None = None, bgcolor: str | None = None, bold: bool = False) -> Style:
    """Layer colours over a widget's base style. Cached, because the visualizer needs thousands per frame."""
    return base + Style(color=color, bgcolor=bgcolor, bold=bold or None)
