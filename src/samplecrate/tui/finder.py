"""The Finder: pick a category, place a dot on a blank plane, get the closest sounds."""

from __future__ import annotations

from rich.segment import Segment
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Footer, Input, OptionList, Static, Tree
from textual.widgets.option_list import Option

from samplecrate import planes, query
from samplecrate.library import Sample
from samplecrate.planes import Category, Match, Point
from samplecrate.tui import plane as plane_view
from samplecrate.tui import theme
from samplecrate.tui.plane import Marker, PlaneFrame
from samplecrate.tui.transport import Control, TransportBar
from samplecrate.tui.visualizer import Visualizer

COUNTS = (3, 5, 10)
ANY = "__any__"
CLOSENESS_COLORS = {planes.CLOSE: "#9ece6a", planes.FAIR: "#e0af68", planes.FAR: "#f7768e"}
_CATEGORY_TERMS = ("in", "main", "sub", "type", "is")


class PlanePicker(Widget, can_focus=True):
    """A blank plane. Aim with the arrow keys or the mouse, then submit a dot."""

    BINDINGS = [
        Binding("left,h", "aim(-1, 0)", "Aim", show=False),
        Binding("right,l", "aim(1, 0)", show=False),
        Binding("up,k", "aim(0, 1)", show=False),
        Binding("down,j", "aim(0, -1)", show=False),
        Binding("enter,space", "submit", "Place dot"),
    ]

    class Submitted(Message):
        def __init__(self, point: Point):
            super().__init__()
            self.point = point

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec: planes.PlaneSpec | None = None
        self.cursor: Point = (0.5, 0.5)
        self.target: Point | None = None
        self.markers: list[Marker] = []
        self.accent = "#7aa2f7"
        self.message = "Measuring your library…"

    def _frame(self) -> PlaneFrame | None:
        if self.spec is None:
            return None
        hint = "Aim with the arrow keys or mouse. Enter or click places the dot."
        return PlaneFrame(self.spec, target=self.target, markers=self.markers,
                          cursor=None if self.target == self.cursor else self.cursor, readout=hint, accent=self.accent)

    def render_line(self, y: int) -> Strip:
        width, height = self.size.width, self.size.height
        frame = self._frame()
        if frame is None:
            line = self.message.center(width)[:width] if y == height // 2 else " " * width
            return Strip([Segment(line, self.rich_style)], width)
        lines = plane_view.draw(frame, width, height, self.rich_style, self.background_colors[1].hex6)
        return lines[y] if 0 <= y < len(lines) else Strip.blank(width)

    def reset(self, spec: planes.PlaneSpec, accent: str) -> None:
        self.spec, self.accent = spec, accent
        self.cursor, self.target, self.markers = (0.5, 0.5), None, []
        self.refresh()

    def _area(self) -> plane_view.Area:
        return plane_view.plane_area(self.size.width, self.size.height, True)

    def action_aim(self, dx: int, dy: int) -> None:
        area = self._area()
        x = min(max(self.cursor[0] + dx / max(area.width - 1, 1), 0.0), 1.0)
        y = min(max(self.cursor[1] + dy / max(area.height - 1, 1), 0.0), 1.0)
        self.cursor = (x, y)
        self.refresh()

    def action_submit(self) -> None:
        if self.spec is not None:
            self.post_message(self.Submitted(self.cursor))

    def on_click(self, event) -> None:
        if self.spec is None:
            return
        area = self._area()
        if event.x >= area.left and area.top <= event.y < area.top + area.height:
            self.cursor = plane_view.to_plane(area, event.x, event.y)
            self.refresh()
            self.post_message(self.Submitted(self.cursor))


class ResultList(OptionList):
    BINDINGS = [
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
    ]


class FinderFilter(Input):
    BINDINGS = [Binding("escape,down", "screen.focus_plane", show=False)]


class FinderScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("3", "count(3)", "Top 3"),
        Binding("5", "count(5)", "Top 5"),
        Binding("0", "count(10)", "Top 10"),
        Binding("m", "mark_result", "Mark"),
        Binding("p", "put_result", "Put on pad"),
        Binding("L", "show_in_library", "Show in library"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, count: int = 5):
        super().__init__()
        self.count = count if count in COUNTS else 5
        self.category: Category | None = None
        self.any_category = False
        self.matches: list[Match] = []
        self._filter_timer: Timer | None = None
        self._selected_rel: str | None = None

    def compose(self) -> ComposeResult:
        crate = self.app
        self.picker = PlanePicker(id="finder-plane")
        self.results = ResultList(id="finder-results")
        self.categories = Tree("Categories", id="finder-categories")
        self.filter_box = FinderFilter(value=self._starting_filter(crate.filter_input.value), id="finder-filter",
                                       placeholder="Narrow the search: key:~Am  bpm:~120  genre:house …")
        self.status = Static("", id="finder-status")
        self.heading = Static("", id="finder-heading")
        self.bar = TransportBar(id="finder-transport")
        self.viz = Visualizer(id="finder-viz")
        yield self.heading
        with Horizontal(id="filterbar"):
            yield Static("/", id="slash")
            yield self.filter_box
        with Horizontal(id="finder-body"):
            yield self.categories
            yield self.picker
            with Vertical(id="finder-side"):
                with Horizontal(id="finder-counts"):
                    yield Static("Show ", classes="count-label")
                    for n in COUNTS:
                        yield Control(f" {n} ", f"screen.count({n})", id=f"count-{n}", classes="control")
                yield self.status
                yield self.results
        yield self.bar
        yield self.viz
        yield Footer()

    @staticmethod
    def _starting_filter(text: str) -> str:
        """Keep musical terms from the main filter, but not ones the category choice would contradict."""
        for term in _CATEGORY_TERMS:
            text = query.set_term(text, term, None)
        return text

    def on_mount(self) -> None:
        self.viz.set_mode(self.app.viz.mode if self.app.viz.mode in ("wave", "spectrum", "scope", "gram") else "wave")
        self.viz.set_clip(None, message="Place a dot, then pick a result to see it here")
        self._update_counts()
        self.rebuild_categories()
        self.picker.focus()

    # ------------------------------------------------------------------ categories

    def rebuild_categories(self) -> None:
        crate = self.app
        tree = self.categories
        tree.clear()
        tree.show_root = False
        tree.guide_depth = 2
        tree.root.expand()
        space = crate.space
        found = space.finder_categories()
        if not space.footprint:
            self.picker.spec = None
            self.picker.message = "Measuring your library…"
            self.picker.refresh()
            return
        tree.root.add_leaf(Text("Any category", style="bold"), data=ANY)
        groups: dict[str, object] = {}
        for category, size in found:
            main = category[0]
            if main not in groups:
                groups[main] = tree.root.add(Text(main, style=theme.category_color(main)), expand=False)
            label = Text(planes.category_label(category))
            label.append(f"  {size}", style=theme.MUTED_COLOR)
            groups[main].add_leaf(label, data=category)
        if self.category is None and not self.any_category:
            current = crate.current
            own = planes.category_of(current) if current else None
            self.set_category(own if own in dict(found) else None)
            if own in dict(found):
                groups[own[0]].expand()

    def set_category(self, category: Category | None) -> None:
        self.category = category
        self.any_category = category is None
        if category is None:
            self.picker.reset(planes.footprint_spec(), "#7aa2f7")
        else:
            self.picker.reset(planes.character_spec(category), theme.category_color(category[0]))
        self.matches = []
        self._selected_rel = None
        self.results.clear_options()
        self.status.update("Place a dot on the plane.")
        self._update_heading()

    def _update_heading(self) -> None:
        text = Text("Find  ", style="bold")
        if self.any_category:
            text.append("Any category", style="bold #7aa2f7")
            text.append("  ·  length against Low, Mid and High", style=theme.MUTED_COLOR)
        elif self.category:
            text.append(planes.category_label(self.category), style=f"bold {theme.category_color(self.category[0])}")
            text.append(f"  ·  {self.category[0]}", style=theme.MUTED_COLOR)
        if self.app.pad_target is not None:
            text.append(f"  ·  choosing for pad {self.app.pad_target + 1}: p puts a result there", style=f"bold {theme.MARK_COLOR}")
        self.heading.update(text)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if data is None:
            return
        self.set_category(None if data == ANY else data)
        self.picker.focus()

    # ------------------------------------------------------------------ searching

    def _pool(self) -> list[Sample]:
        crate = self.app
        parsed = query.parse(self.filter_box.value, crate.table.marked)
        self.filter_box.set_class(bool(parsed.errors), "-invalid")
        if self.any_category:
            samples = crate.library.samples
        else:
            samples = crate.space.categories.get(self.category, [])
        return parsed.filter(samples)

    def on_plane_picker_submitted(self, message: PlanePicker.Submitted) -> None:
        self.picker.target = message.point
        self._search()
        if self.matches:
            self.results.focus()

    def _search(self) -> None:
        target = self.picker.target
        if target is None:
            return
        crate = self.app
        kind = "footprint" if self.any_category else "character"
        self.matches = planes.nearest(crate.space, kind, self._pool(), target, self.count)
        self.status.update(planes.summarize_matches(self.matches, self.count))
        self._show_results()

    def _option_text(self, index: int, match: Match) -> Text:
        sample = match.sample
        marked = sample.rel in self.app.table.marked
        text = Text()
        text.append(f"{index + 1:>2} ", style="bold")
        text.append("●" if marked else " ", style=theme.MARK_COLOR)
        text.append(f"{match.closeness:<6}", style=CLOSENESS_COLORS[match.closeness])
        text.append(sample.desc[:27].ljust(28))
        if self.any_category:
            text.append(f"{sample.sub[:10]:<11}", style=theme.category_color(sample.main))
        text.append(f"{sample.features.audible:6.2f}s", style=theme.MUTED_COLOR)
        return text

    def _show_results(self, keep_index: int | None = None) -> None:
        results = self.results
        wanted = keep_index
        if wanted is None:
            wanted = next((i for i, m in enumerate(self.matches) if m.sample.rel == self._selected_rel), 0)
        results.clear_options()
        results.add_options([Option(self._option_text(i, m)) for i, m in enumerate(self.matches)])
        if self.matches:
            results.highlighted = min(wanted, len(self.matches) - 1)
        self._update_markers()

    def _update_markers(self) -> None:
        active = self.results.highlighted
        self.picker.markers = [Marker(m.position[0], m.position[1], str(i + 1) if i < 9 else "0", i == active)
                               for i, m in enumerate(self.matches)]
        self.picker.refresh()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        index = event.option_index
        if not 0 <= index < len(self.matches):
            return
        self._update_markers()
        sample = self.matches[index].sample
        if sample.rel != self._selected_rel:
            self._selected_rel = sample.rel
            self.app.select_sample(sample, by_user=True)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.app.play_current_from_start()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._filter_timer:
            self._filter_timer.stop()
        self._filter_timer = self.set_timer(0.08, self._search)

    def on_input_submitted(self) -> None:
        self.action_focus_plane()

    # ------------------------------------------------------------------ actions

    def _update_counts(self) -> None:
        for n in COUNTS:
            self.query_one(f"#count-{n}", Control).show(f" {n} ", lit=n == self.count)

    def action_count(self, count: int) -> None:
        self.count = count
        self._update_counts()
        self.app.prefs_finder_count = count
        self._search()

    def action_focus_filter(self) -> None:
        self.filter_box.focus()

    def action_focus_plane(self) -> None:
        self.picker.focus()

    def action_back(self) -> None:
        if self.focused is self.results or self.focused is self.categories:
            self.picker.focus()
        else:
            self.action_close()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_mark_result(self) -> None:
        index = self.results.highlighted
        if index is None or not 0 <= index < len(self.matches):
            return
        self.app.table.marked.symmetric_difference_update({self.matches[index].sample.rel})
        self._show_results(index)

    def action_put_result(self) -> None:
        index = self.results.highlighted
        if index is not None and 0 <= index < len(self.matches):
            self.app.put_from_finder(self.matches[index].sample)

    def action_show_in_library(self) -> None:
        index = self.results.highlighted
        if index is not None and 0 <= index < len(self.matches):
            self.dismiss(self.matches[index].sample)
