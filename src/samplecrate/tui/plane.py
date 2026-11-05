"""Drawing the sound maps. The same renderer serves the visualizer panel and the Finder's blank plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from samplecrate import planes
from samplecrate.planes import PlaneSpec, Point
from samplecrate.tui import theme

GUTTER = 7
MIN_WIDTH, MIN_HEIGHT = 24, 4
SPARSE_SHARE = 0.25
_DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
_X_TICKS = ((0.05, ".05s"), (0.1, ".1s"), (0.5, ".5s"), (1, "1s"), (4, "4s"), (10, "10s"), (50, "50s"))
_REGIONS = (("HITS", planes.MIN_SECONDS, planes.HIT_SECONDS), ("NOTES", planes.HIT_SECONDS, planes.NOTE_SECONDS),
            ("PHRASES & LOOPS", planes.NOTE_SECONDS, planes.MAX_SECONDS))


@dataclass(slots=True)
class Dot:
    x: float
    y: float
    color: str | None = None    # None draws it faint, as background context
    rel: str = ""


@dataclass(slots=True)
class Marker:
    x: float
    y: float
    label: str
    active: bool = False


@dataclass
class PlaneFrame:
    spec: PlaneSpec
    dots: list[Dot] = field(default_factory=list)
    selected: Point | None = None
    span: tuple[float, float] | None = None     # how far the selected sample's energy reaches, on the y axis
    markers: list[Marker] = field(default_factory=list)
    target: Point | None = None                 # the dot you placed in the Finder
    cursor: Point | None = None                 # where the dot would go
    readout: str | None = None
    accent: str = "#7aa2f7"


@dataclass(frozen=True, slots=True)
class Area:
    """The part of the widget that is the plane itself, in cells."""
    left: int
    top: int
    width: int
    height: int


def plane_area(width: int, height: int, has_readout: bool) -> Area:
    top = 1 if has_readout else 0
    return Area(GUTTER, top, max(width - GUTTER - 1, 1), max(height - top - 1, 1))


def to_plane(area: Area, x: int, y: int) -> Point:
    """A clicked cell, as a position on the plane from 0 to 1."""
    px = (x - area.left) / max(area.width - 1, 1)
    py = 1 - (y - area.top) / max(area.height - 1, 1)
    return min(max(px, 0.0), 1.0), min(max(py, 0.0), 1.0)


def to_cell(area: Area, point: Point) -> tuple[int, int]:
    return round(point[0] * (area.width - 1)), round((1 - point[1]) * (area.height - 1))


def seconds_x(seconds: float) -> float:
    span = math.log(planes.MAX_SECONDS) - math.log(planes.MIN_SECONDS)
    return (math.log(seconds) - math.log(planes.MIN_SECONDS)) / span


class _Grid:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.chars = [[" "] * width for _ in range(height)]
        self.fg: list[list[str | None]] = [[None] * width for _ in range(height)]
        self.bg: list[list[str | None]] = [[None] * width for _ in range(height)]
        self.bold = [[False] * width for _ in range(height)]

    def put(self, x: int, y: int, char: str, fg: str | None = None, bg: str | None = None, bold: bool = False) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.chars[y][x], self.fg[y][x], self.bg[y][x], self.bold[y][x] = char, fg, bg, bold

    def text(self, x: int, y: int, text: str, fg: str | None = None, bold: bool = False) -> None:
        for offset, char in enumerate(text):
            self.put(x + offset, y, char, fg, None, bold)

    def strip(self, base: Style, prefix: list[Segment], y: int) -> list[Segment]:
        segments = list(prefix)
        x = 0
        chars, fg, bg, bold = self.chars[y], self.fg[y], self.bg[y], self.bold[y]
        while x < self.width:
            end = x + 1
            while end < self.width and fg[end] == fg[x] and bg[end] == bg[x] and bold[end] == bold[x]:
                end += 1
            segments.append(Segment("".join(chars[x:end]), theme.compose(base, fg[x], bg[x], bold[x])))
            x = end
        return segments


def _draw_guides(grid: _Grid, spec: PlaneSpec, color: str) -> None:
    w, h = grid.width, grid.height
    if spec.kind == "character":
        rows, cols = [h // 2], [w // 2]
    else:
        rows = [round((1 - 2 / 3) * (h - 1)), round((1 - 1 / 3) * (h - 1))]
        cols = [round(seconds_x(planes.HIT_SECONDS) * (w - 1)), round(seconds_x(planes.NOTE_SECONDS) * (w - 1))]
    for row in rows:
        for x in range(w):
            grid.put(x, row, "─", color)
    for col in cols:
        for y in range(h):
            grid.put(col, y, "┼" if y in rows else ("│" if spec.kind == "character" else "┊"), color)


def _draw_dots(grid: _Grid, dots: list[Dot], faint: str) -> None:
    w, h = grid.width, grid.height
    codes: dict[tuple[int, int], int] = {}
    colors: dict[tuple[int, int], str] = {}
    for dot in dots:
        sx = round(dot.x * (w * 2 - 1))
        sy = round((1 - dot.y) * (h * 4 - 1))
        cell = (sx // 2, sy // 4)
        codes[cell] = codes.get(cell, 0) | _DOT_BITS[sy % 4][sx % 2]
        if dot.color or cell not in colors:
            if dot.color or colors.get(cell, faint) == faint:
                colors[cell] = dot.color or faint
    # Braille packs a crowded map with detail, but a lone braille dot is hard to see on a sparse one
    sparse = len(codes) < w * h * SPARSE_SHARE
    for (x, y), code in codes.items():
        grid.put(x, y, "•" if sparse else chr(0x2800 + code), colors[(x, y)])


def _draw_labels(grid: _Grid, frame: PlaneFrame, muted: str) -> None:
    spec = frame.spec
    w, h = grid.width, grid.height
    if spec.kind == "footprint":
        for name, low, high in _REGIONS:
            start, end = seconds_x(low) * (w - 1), seconds_x(high) * (w - 1)
            if end - start > len(name) + 1:
                grid.text(round((start + end - len(name)) / 2), 0, name, muted)
    elif spec.corners:
        dark_low, bright_low, dark_high, bright_high = spec.corners
        accent = frame.accent
        grid.text(1, 0, bright_low.upper(), accent, True)
        grid.text(w - len(bright_high) - 1, 0, bright_high.upper(), accent, True)
        grid.text(1, h - 1, dark_low.upper(), accent, True)
        grid.text(w - len(dark_high) - 1, h - 1, dark_high.upper(), accent, True)


def _gutter_labels(spec: PlaneSpec, height: int) -> dict[int, str]:
    if spec.kind == "character":
        return {0: spec.y_high, height - 1: spec.y_low}
    labels = {}
    for band, name in enumerate(planes.BAND_NAMES):
        labels[round((1 - (band + 0.5) / 3) * (height - 1))] = name.upper()
    return labels


def _axis_line(spec: PlaneSpec, width: int) -> str:
    line = [" "] * width
    if spec.kind == "character":
        left, right = f"◄ {spec.x_low}", f"{spec.x_high} ►"
        line[0:len(left)] = left
        line[width - len(right):width] = right
    else:
        for seconds, label in _X_TICKS:
            x = round(seconds_x(seconds) * (width - 1))
            if x + len(label) <= width and all(c == " " for c in line[max(x - 1, 0):x + len(label)]):
                line[x:x + len(label)] = label
    return "".join(line)


def draw(frame: PlaneFrame, width: int, height: int, base: Style, background: str) -> list[Strip]:
    """Render a whole plane to strips, one per row."""
    blank = Strip([Segment(" " * width, base)], width)
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        lines = [blank] * height
        if height:
            text = "Make the window bigger to see this map"[:width].center(width)
            lines[height // 2] = Strip([Segment(text, theme.compose(base, theme.MUTED_COLOR))], width)
        return lines

    area = plane_area(width, height, frame.readout is not None)
    grid = _Grid(area.width, area.height)
    muted = theme.MUTED_COLOR
    guide = theme.blend(background, muted, 0.4)
    _draw_guides(grid, frame.spec, guide)
    _draw_dots(grid, frame.dots, theme.blend(background, muted, 0.75))
    _draw_labels(grid, frame, theme.blend(background, muted, 0.8))

    if frame.selected is not None:
        x, y = to_cell(area, frame.selected)
        if frame.span:
            top = to_cell(area, (0, frame.span[1]))[1]
            bottom = to_cell(area, (0, frame.span[0]))[1]
            for row in range(top, bottom + 1):
                grid.put(x, row, "┃", frame.accent)
        grid.put(x, y, "◉", "#ffffff", None, True)
    if frame.target is not None:
        x, y = to_cell(area, frame.target)
        grid.put(x, y, "◎", "#ffffff", None, True)
    for marker in frame.markers:
        x, y = to_cell(area, (marker.x, marker.y))
        if marker.active:
            grid.put(x, y, marker.label, background, "#ffffff", True)
        else:
            grid.put(x, y, marker.label, background, frame.accent, True)
    if frame.cursor is not None:
        x, y = to_cell(area, frame.cursor)
        grid.put(x, y, "╋", "#ffffff", None, True)

    labels = _gutter_labels(frame.spec, area.height)
    lines: list[Strip] = []
    if frame.readout is not None:
        text = frame.readout[:width - 1]
        lines.append(Strip([Segment(" " + text, theme.compose(base, "#ffffff", bold=True))]).adjust_cell_length(width, base))
    for row in range(area.height):
        gutter = labels.get(row, "").rjust(GUTTER - 1)[:GUTTER - 1] + " "
        prefix = [Segment(gutter, theme.compose(base, muted))]
        lines.append(Strip(grid.strip(base, prefix, row)).adjust_cell_length(width, base))
    axis = " " * GUTTER + _axis_line(frame.spec, area.width)
    lines.append(Strip([Segment(axis, theme.compose(base, muted))]).adjust_cell_length(width, base))
    return lines
