"""A virtual sample table: it only ever draws the rows on screen, so filtering stays instant at any library size."""

from __future__ import annotations

from dataclasses import dataclass

from rich.cells import set_cell_size
from rich.segment import Segment
from rich.style import Style
from textual.binding import Binding
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip

from samplecrate import naming
from samplecrate.library import Sample
from samplecrate.tui import theme

FLAGS_WIDTH = 3
ARROWS = {False: "▲", True: "▼"}


@dataclass(frozen=True, slots=True)
class Column:
    id: str
    label: str
    width: int
    right: bool = False


def layout(width: int, show_genres: bool = False) -> list[Column]:
    """Columns for a given table width. The description takes what's left, and narrow tables drop columns.

    Genres appear when there's room, or whenever you're sorting by them.
    """
    fixed = [Column("main", "Main", 11), Column("sub", "Sub", 11), Column("type", "Type", 5),
             Column("bpm", "BPM", 4, right=True), Column("key", "Key", 6), Column("len", "Length", 7, right=True)]
    if width >= 100 or show_genres:
        fixed.append(Column("genres", "Genres", 16))
    if width < (74 if not show_genres else 96):
        fixed = [c for c in fixed if c.id != "main"]
    if width < 58:
        fixed = [c for c in fixed if c.id not in ("type", "len")]
    used = FLAGS_WIDTH + sum(c.width + 1 for c in fixed) + 1
    return [*fixed, Column("desc", "Description", max(width - used, 8))]


class SampleTable(ScrollView, can_focus=True):
    COMPONENT_CLASSES = {
        "sample-table--header",
        "sample-table--cursor",
        "sample-table--cursor-blur",
        "sample-table--marked",
        "sample-table--muted",
    }
    BINDINGS = [
        Binding("up,k", "move(-1)", "Up", show=False),
        Binding("down,j", "move(1)", "Down", show=False),
        Binding("pageup,ctrl+b", "page(-1)", "Page up", show=False),
        Binding("pagedown,ctrl+f", "page(1)", "Page down", show=False),
        Binding("home,g", "jump(0)", "Top", show=False),
        Binding("end,G", "jump(-1)", "Bottom", show=False),
    ]

    cursor = reactive(0, repaint=False)

    class Highlighted(Message):
        """The sample under the cursor changed. by_user is False when a filter or sort moved it, not you."""
        def __init__(self, sample: Sample | None, by_user: bool):
            super().__init__()
            self.sample = sample
            self.by_user = by_user

    class Activated(Message):
        """Enter or double-click on a row."""
        def __init__(self, sample: Sample):
            super().__init__()
            self.sample = sample

    class SortRequested(Message):
        def __init__(self, column: str):
            super().__init__()
            self.column = column

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows: list[Sample] = []
        self.marked: set[str] = set()
        self.sort_column = "main"
        self.sort_reverse = False
        self.anchor: str | None = None      # the sample you last chose, which filters try to stay on
        self._announced: Sample | None = None

    @property
    def current(self) -> Sample | None:
        return self.rows[self.cursor] if 0 <= self.cursor < len(self.rows) else None

    def set_rows(self, rows: list[Sample], keep_rel: str | None = None) -> None:
        """Replace the rows, keeping the cursor on your sample whenever it's listed.

        The anchor survives filters that hide it. Typing "t" on the way to "type:loop" can drop
        your sample for a moment, and it comes back as soon as the filter lets it.
        """
        if keep_rel is not None:
            self.anchor = keep_rel
        elif self.anchor is None and self.current is not None:
            self.anchor = self.current.rel
        index = next((i for i, s in enumerate(rows) if s.rel == self.anchor), None)
        previous = self.cursor
        self.rows = rows
        self.virtual_size = Size(0, len(rows) + 1)
        target = index if index is not None else min(previous, max(len(rows) - 1, 0))
        self.set_reactive(SampleTable.cursor, target)
        self.scroll_to(y=max(target - self._page_rows // 2, 0), animate=False, force=True)
        self.refresh()
        self._announce(by_user=False)

    def _announce(self, *, by_user: bool) -> None:
        current = self.current
        if current is not self._announced:
            self._announced = current
            self.post_message(self.Highlighted(current, by_user))

    def _columns(self, width: int) -> list[Column]:
        return layout(width, self.sort_column == "genres")

    @property
    def _page_rows(self) -> int:
        return max(self.size.height - 1, 1)

    # ------------------------------------------------------------------ navigation

    def watch_cursor(self) -> None:
        self.anchor = self.current.rel if self.current else self.anchor
        top = int(self.scroll_y)
        if self.cursor < top:
            self.scroll_to(y=self.cursor, animate=False, force=True)
        elif self.cursor >= top + self._page_rows:
            self.scroll_to(y=self.cursor - self._page_rows + 1, animate=False, force=True)
        self.refresh()
        self._announce(by_user=True)

    def _clamp(self, index: int) -> int:
        return min(max(index, 0), max(len(self.rows) - 1, 0))

    def action_move(self, delta: int) -> None:
        self.cursor = self._clamp(self.cursor + delta)

    def action_page(self, direction: int) -> None:
        self.cursor = self._clamp(self.cursor + direction * self._page_rows)

    def action_jump(self, index: int) -> None:
        self.cursor = self._clamp(index if index >= 0 else len(self.rows) - 1)

    def on_click(self, event) -> None:
        if event.y == 0:
            column = self._column_at(event.x + int(self.scroll_x))
            if column:
                self.post_message(self.SortRequested(column))
            return
        index = int(self.scroll_y) + event.y - 1
        if 0 <= index < len(self.rows):
            self.cursor = index
            if event.chain >= 2:
                self.post_message(self.Activated(self.rows[index]))

    def key_enter(self) -> None:
        if self.current:
            self.post_message(self.Activated(self.current))

    def _column_at(self, x: int) -> str | None:
        pos = FLAGS_WIDTH
        for column in self._columns(self.scrollable_content_region.width):
            if pos <= x < pos + column.width + 1:
                return column.id
            pos += column.width + 1
        return None

    def on_resize(self) -> None:
        self.virtual_size = Size(0, len(self.rows) + 1)

    # ------------------------------------------------------------------ drawing

    def render_line(self, y: int) -> Strip:
        width = self.scrollable_content_region.width
        if y == 0:
            return self._header_strip(width)
        index = int(self.scroll_y) + y - 1
        if 0 <= index < len(self.rows):
            return self._row_strip(self.rows[index], index == self.cursor, width)
        return Strip([Segment(" " * width, self.rich_style)], width)

    def _header_strip(self, width: int) -> Strip:
        style = self.get_component_rich_style("sample-table--header")
        segments = [Segment(" " * FLAGS_WIDTH, style)]
        for column in self._columns(width):
            label = column.label
            if column.id == self.sort_column:
                label = f"{label}{ARROWS[self.sort_reverse]}"
            segments.append(Segment(_fit(label, column.width, column.right) + " ", style))
        return Strip(segments).adjust_cell_length(width, style)

    def _cell_values(self, sample: Sample) -> dict[str, tuple[str, str | None]]:
        muted = theme.MUTED_COLOR
        cat = theme.category_color(sample.main)
        is_loop = sample.type == naming.LOOP
        return {
            "main": (sample.main, cat),
            "sub": (sample.sub, cat),
            "type": ("Loop" if is_loop else ("Shot" if sample.type else ""), None if is_loop else muted),
            "bpm": (str(sample.bpm) if sample.bpm else "·", None if sample.bpm else muted),
            "key": (sample.key or "·", theme.key_color(sample.key) if sample.key else muted),
            "len": (f"{sample.length:.2f}s" if sample.length is not None else "·", muted),
            "genres": (", ".join(sample.genres) or "·", None if sample.genres else muted),
            "desc": (sample.desc, theme.WARNING_COLOR if sample.problems else None),
        }

    def _row_strip(self, sample: Sample, is_cursor: bool, width: int) -> Strip:
        marked = sample.rel in self.marked
        if is_cursor:
            base = self.get_component_rich_style("sample-table--cursor" if self.has_focus else "sample-table--cursor-blur")
        elif marked:
            base = self.get_component_rich_style("sample-table--marked")
        else:
            base = self.rich_style
        segments = [
            Segment(" ", base),
            Segment("★" if sample.favorite else " ", theme.compose(base, theme.FAVORITE_COLOR)),
            _flag_segment(sample, marked, base),
        ]
        values = self._cell_values(sample)
        for column in self._columns(width):
            text, color = values[column.id]
            style = theme.compose(base, color) if color and not is_cursor else base
            segments.append(Segment(_fit(text, column.width, column.right) + " ", style))
        return Strip(segments).adjust_cell_length(width, base)


def _flag_segment(sample: Sample, marked: bool, base: Style) -> Segment:
    if marked:
        return Segment("●", theme.compose(base, theme.MARK_COLOR, bold=True))
    if sample.problems:
        return Segment("!", theme.compose(base, theme.WARNING_COLOR, bold=True))
    return Segment(" ", base)


def _fit(text: str, width: int, right: bool = False) -> str:
    if len(text) > width:
        text = text[:max(width - 1, 0)] + "…"
    fitted = set_cell_size(text, width)
    return fitted.strip().rjust(width) if right else fitted
