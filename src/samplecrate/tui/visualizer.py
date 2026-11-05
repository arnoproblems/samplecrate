"""The sound visualizer panel: waveform, live spectrum, oscilloscope and spectrogram."""

from __future__ import annotations

import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget

from samplecrate import viz
from samplecrate.audio import Clip
from samplecrate.tui import plane as plane_view
from samplecrate.tui import theme
from samplecrate.tui.plane import PlaneFrame

AUDIO_MODES = ("wave", "spectrum", "scope", "gram")
PLANE_MODES = ("character", "footprint")
MODES = AUDIO_MODES + PLANE_MODES
MODE_LABELS = {"wave": "Waveform", "spectrum": "Spectrum", "scope": "Scope", "gram": "Spectrogram",
               "character": "Character", "footprint": "Footprint"}
SPECTRUM_WINDOW = 4096
SCOPE_SECONDS = 0.035
MAX_SCOPE_GAIN = 12
FALL_PER_FRAME = 0.045
PEAK_FALL_PER_FRAME = 0.012


class Visualizer(Widget):
    can_focus = False

    class Seek(Message):
        def __init__(self, fraction: float):
            super().__init__()
            self.fraction = fraction

    class PlaneClicked(Message):
        def __init__(self, point: tuple[float, float], double: bool):
            super().__init__()
            self.point = point
            self.double = double

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "wave"
        self.clip: Clip | None = None
        self.color = "#7aa2f7"
        self.bpm: int | None = None
        self.position = 0.0
        self.playing = False
        self.message = "Select a sample"
        self.plane: PlaneFrame | None = None
        self.plane_message = "Measuring your library…"
        self._frame: list[Strip] | None = None
        self._cache_key = None
        self._wave_rows: list[str] = []
        self._wave_invert = np.zeros((0, 0), dtype=bool)
        self._wave_core = np.zeros((0, 0), dtype=bool)
        self._last_play_x: int | None = None
        self._ruler = ""
        self._gram_rows: list[list[Segment]] = []
        self._levels = np.zeros(0)
        self._peaks = np.zeros(0)

    # ------------------------------------------------------------------ inputs

    def set_clip(self, clip: Clip | None, color: str = "#7aa2f7", bpm: int | None = None, message: str = "") -> None:
        self.clip, self.color, self.bpm = clip, color, bpm
        self.message = message
        self.position = 0.0
        self._levels = np.zeros(0)
        self._peaks = np.zeros(0)
        self._cache_key = None
        self._invalidate()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._levels = np.zeros(0)
        self._peaks = np.zeros(0)
        self._cache_key = None
        self._invalidate()

    def set_plane(self, frame: PlaneFrame | None, message: str = "") -> None:
        self.plane = frame
        self.plane_message = message
        if self.mode in PLANE_MODES:
            self._invalidate()

    def set_transport(self, position: float, playing: bool) -> None:
        was_playing = self.playing
        self.position, self.playing = position, playing
        if self.mode in PLANE_MODES:
            return
        if self.mode in ("spectrum", "scope"):
            if playing or was_playing:
                self._invalidate()
            return
        # The waveform and spectrogram only change when the playhead reaches a new column
        play_x = self._play_x(self.size.width) if self.size.width else None
        if playing != was_playing or play_x != self._last_play_x:
            self._last_play_x = play_x
            self._invalidate()

    def _invalidate(self) -> None:
        self._frame = None
        self.refresh()

    def on_resize(self) -> None:
        self._cache_key = None
        self._frame = None

    def on_click(self, event) -> None:
        if self.mode in PLANE_MODES:
            if self.plane is not None:
                area = plane_view.plane_area(self.size.width, self.size.height, self.plane.readout is not None)
                self.post_message(self.PlaneClicked(plane_view.to_plane(area, event.x, event.y), event.chain >= 2))
            return
        if self.clip is not None and self.size.width > 0:
            self.post_message(self.Seek(min(max(event.x / self.size.width, 0.0), 1.0)))

    # ------------------------------------------------------------------ drawing

    def render_line(self, y: int) -> Strip:
        if self._frame is None or len(self._frame) != self.size.height:
            self._frame = self._draw(self.size.width, self.size.height)
        return self._frame[y] if 0 <= y < len(self._frame) else Strip.blank(self.size.width)

    def _draw(self, width: int, height: int) -> list[Strip]:
        if width <= 0 or height <= 0:
            return []
        if self.mode in PLANE_MODES:
            if self.plane is None:
                return self._draw_message(width, height, self.plane_message)
            return plane_view.draw(self.plane, width, height, self.rich_style, self._background)
        if self.clip is None:
            return self._draw_message(width, height)
        return {"wave": self._draw_wave, "spectrum": self._draw_spectrum,
                "scope": self._draw_scope, "gram": self._draw_gram}[self.mode](width, height)

    def _draw_message(self, width: int, height: int, message: str | None = None) -> list[Strip]:
        base = self.rich_style
        blank = Strip([Segment(" " * width, base)], width)
        lines = [blank] * height
        text = (self.message if message is None else message).center(width)[:width]
        lines[height // 2] = Strip([Segment(text, theme.compose(base, theme.MUTED_COLOR))], width)
        return lines

    @property
    def _background(self) -> str:
        return self.background_colors[1].hex6

    def _play_x(self, width: int) -> int | None:
        if not self.clip or not (self.playing or self.position > 0) or self.clip.duration <= 0:
            return None
        return min(int(self.position / self.clip.duration * width), width - 1)

    def _ruler_strip(self, width: int, base: Style) -> Strip:
        style = theme.compose(base, theme.MUTED_COLOR)
        return Strip([Segment(self._ruler, style)], width).adjust_cell_length(width, base)

    # waveform -----------------------------------------------------------

    def _draw_wave(self, width: int, height: int) -> list[Strip]:
        rows = max(height - 1, 1)
        key = ("wave", width, rows, id(self.clip))
        if self._cache_key != key:
            clip = self.clip
            self._wave_rows, self._wave_invert, self._wave_core = viz.wave_cells(clip.mins, clip.maxs, clip.rms, width, rows)
            self._ruler = viz.ruler(clip.duration, width, self.bpm)[0]
            self._cache_key = key
        base = self.rich_style
        bg = self._background
        play_x = self._play_x(width)
        # What's been played is lit in the sample's colour, and the rest is a dimmer shade.
        # Within each, the RMS core is brighter than the peaks around it.
        colors = {
            (True, True): theme.blend(self.color, "#ffffff", 0.3),
            (True, False): self.color,
            (False, True): theme.blend(bg, self.color, 0.8),
            (False, False): theme.blend(bg, self.color, 0.5),
        }
        styles = {}
        for (is_lit, is_core), color in colors.items():
            styles[(is_lit, is_core, False)] = theme.compose(base, color)
            styles[(is_lit, is_core, True)] = theme.compose(base, bg, color)
        head = theme.compose(base, "#ffffff", bold=True)
        lines = []
        for row, chars in enumerate(self._wave_rows):
            invert, core = self._wave_invert[row], self._wave_core[row]
            segments, x = [], 0
            while x < width:
                if x == play_x:
                    segments.append(Segment("│", head))
                    x += 1
                    continue
                is_lit = play_x is not None and x < play_x
                state = (is_lit, bool(core[x]), bool(invert[x]))
                end = x + 1
                while (end < width and end != play_x and bool(core[end]) == state[1]
                       and bool(invert[end]) == state[2]):
                    end += 1
                segments.append(Segment(chars[x:end], styles[state]))
                x = end
            lines.append(Strip(segments, width))
        if height > 1:
            lines.append(self._ruler_strip(width, base))
        return lines

    # spectrum -----------------------------------------------------------

    def _live_window(self, size: int) -> np.ndarray:
        clip = self.clip
        position = int(self.position * clip.sr)
        window = clip.mono[position:position + size]
        return np.pad(window, (0, size - len(window))) if len(window) < size else window

    def _draw_spectrum(self, width: int, height: int) -> list[Strip]:
        rows = max(height - 1, 1)
        bar_width = 2 if width >= 90 else 1
        bars = max(width // (bar_width + 1), 1)
        clip = self.clip
        if self.playing:
            target = viz.spectrum_levels(self._live_window(SPECTRUM_WINDOW), clip.sr, bars)
        else:
            target = viz.resample_levels(clip.mean_spectrum, bars)
        if len(self._levels) != bars or not self.playing:
            self._levels, self._peaks = target.copy(), target.copy()
        # Bars jump up immediately and fall gently, like a hardware meter
        self._levels = np.maximum(target, self._levels - FALL_PER_FRAME)
        self._peaks = np.maximum(self._levels, self._peaks - PEAK_FALL_PER_FRAME)
        base = self.rich_style
        ramp = theme.METER_RAMP
        lines = []
        for row in range(rows):
            from_bottom = rows - 1 - row
            color = ramp[min(int(from_bottom / max(rows - 1, 1) * (len(ramp) - 1)), len(ramp) - 1)]
            style = theme.compose(base, color)
            peak_style = theme.compose(base, "#ffffff")
            segments = []
            for bar in range(bars):
                level, peak = self._levels[bar], self._peaks[bar]
                glyph = viz.bar_glyph(level, from_bottom, rows)
                # A held peak shows as a thin line above the bar
                if glyph == " " and self.playing and int(peak * rows) == from_bottom and peak > level:
                    segments.append(Segment("▔" * bar_width, peak_style))
                else:
                    segments.append(Segment(glyph * bar_width, style))
                segments.append(Segment(" ", base))
            lines.append(Strip(segments).adjust_cell_length(width, base))
        if height > 1:
            lines.append(self._frequency_axis(width, bars, bar_width, base))
        return lines

    def _frequency_axis(self, width: int, bars: int, bar_width: int, base: Style) -> Strip:
        line = [" "] * width
        top = min(18000, self.clip.sr / 2)
        step = bar_width + 1
        for hz, label in ((50, "50"), (100, "100"), (200, "200"), (500, "500"), (1000, "1k"),
                          (2000, "2k"), (5000, "5k"), (10000, "10k")):
            if hz >= top:
                continue
            x = int(np.log(hz / 30) / np.log(top / 30) * bars) * step
            if x + len(label) <= width:
                line[x:x + len(label)] = label
        return Strip([Segment("".join(line), theme.compose(base, theme.MUTED_COLOR))], width)

    # scope --------------------------------------------------------------

    def _draw_scope(self, width: int, height: int) -> list[Strip]:
        clip = self.clip
        span = max(int(clip.sr * SCOPE_SECONDS), 16)
        if self.playing:
            start = int(self.position * clip.sr)
        else:
            # Stopped: show the loudest moment rather than an empty start
            start = int(np.argmax(clip.rms) / max(len(clip.rms), 1) * len(clip.mono))
        window = viz.scope_window(clip.mono, start, span)
        # Auto-gain so decay tails stay visible, but limited so silence doesn't turn into a wall of noise
        scale = max(float(np.abs(window).max(initial=0)), clip.peak / MAX_SCOPE_GAIN)
        rows = viz.braille_rows(viz.scope_dots(window, width, height, scale))
        base = self.rich_style
        bg = self._background
        lines = []
        for index, chars in enumerate(rows):
            distance = abs(index - (len(rows) - 1) / 2) / max((len(rows) - 1) / 2, 1)
            color = theme.blend(self.color, "#ffffff", distance * 0.5)
            if not self.playing:
                color = theme.blend(bg, color, 0.65)
            lines.append(Strip([Segment(chars, theme.compose(base, color))], width))
        return lines

    # spectrogram --------------------------------------------------------

    def _draw_gram(self, width: int, height: int) -> list[Strip]:
        rows = max(height - 1, 1)
        key = ("gram", width, rows, id(self.clip))
        base = self.rich_style
        if self._cache_key != key:
            clip = self.clip
            pixels = viz.gram_pixels(clip.gram, width, rows)
            ramp = theme.HEAT_RAMP
            indexes = (pixels * (len(ramp) - 1)).astype(int)
            self._gram_rows = []
            for row in range(rows):
                top, bottom = indexes[row * 2], indexes[row * 2 + 1]
                self._gram_rows.append([
                    Segment("▀", theme.compose(base, ramp[top[x]], ramp[bottom[x]])) for x in range(width)])
            self._ruler = viz.ruler(clip.duration, width, self.bpm)[0]
            self._cache_key = key
        play_x = self._play_x(width)
        lines = []
        for segments in self._gram_rows:
            if play_x is None:
                lines.append(Strip(segments, width))
            else:
                cells = list(segments)
                cells[play_x] = Segment("│", theme.compose(base, "#ffffff", cells[play_x].style.bgcolor.get_truecolor().hex
                                                           if cells[play_x].style.bgcolor else None, bold=True))
                lines.append(Strip(cells, width))
        if height > 1:
            lines.append(self._ruler_strip(width, base))
        return lines
