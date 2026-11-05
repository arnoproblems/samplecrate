"""The crate terminal UI."""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Footer, Input, Static, Tree

from samplecrate import audio, banks, duplicates, features as fx
from samplecrate import genres as genre_tags
from samplecrate import filing, naming, organize, planes, query
from samplecrate.library import Bank, Library, Sample, folder_counts, sort_samples
from samplecrate.player import Player, create_player
from samplecrate.tui import theme
from samplecrate.tui.banks import USE_FINDER, BankScreen
from samplecrate.tui.finder import FinderScreen
from samplecrate.tui.plane import Dot, PlaneFrame
from samplecrate.tui.screens import (ConfirmScreen, EditScreen, ExportScreen, GenreScreen, HelpScreen, PathScreen,
                                     ReportScreen)
from samplecrate.tui.table import SampleTable
from samplecrate.tui.transport import TransportBar, TransportState
from samplecrate.tui.visualizer import AUDIO_MODES, MODE_LABELS, MODES, PLANE_MODES, Visualizer
from samplecrate.viz import format_time

FRAME_SECONDS = 1 / 24
FILTER_DEBOUNCE = 0.06
LOAD_DEBOUNCE = 0.03
LOADING_MESSAGE_DELAY = 0.25
PROGRESS_EVERY = 25
DOT_CLICK_REACH = 0.08
DEFAULT_EXPORT = "~/Desktop/crate-export"
SORT_COLUMNS = ("main", "sub", "type", "bpm", "key", "len", "genres", "desc")
FILTER_HINT = "kick dark  bpm:120-130  key:~Am  genre:house  ...  press ? for help"
# Actions that act on the main table, so they don't apply while the Finder or the bank builder is open
TABLE_ACTIONS = {"sort", "reverse_sort", "clear_filter", "edit", "trash", "undo", "mark", "mark_all", "clear_marks",
                 "match_tempo", "match_key", "rescan", "toggle_zoom", "toggle_details", "focus_filter", "open_finder",
                 "add_samples", "find_duplicates", "auto_file", "put_on_pad", "open_banks"}


class FilterInput(Input):
    BINDINGS = [Binding("escape,down", "app.focus_table", show=False)]


class CrateApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "crate"
    HORIZONTAL_BREAKPOINTS = [(0, "-w-small"), (86, "-w-medium"), (124, "-w-large")]
    VERTICAL_BREAKPOINTS = [(0, "-h-small"), (32, "-h-large")]
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("slash", "focus_filter", "Filter"),
        Binding("space", "toggle_play", "Play"),
        Binding("s", "stop_audio", "Stop"),
        Binding("v", "next_mode", "View"),
        Binding("t", "open_finder", "Find"),
        Binding("p", "put_on_pad", "Pad"),
        Binding("B", "open_banks", "Banks"),
        Binding("m", "mark", "Mark"),
        Binding("number_sign", "genres", "Genres"),
        Binding("e", "edit", "Edit"),
        Binding("x", "export", "Export"),
        Binding("d", "trash", "Trash"),
        Binding("u", "undo", "Undo"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+g", "stop_audio", show=False, priority=True),
        *(Binding(str(i + 1), f"sort('{c}')", show=False) for i, c in enumerate(SORT_COLUMNS)),
        Binding("r", "reverse_sort", show=False),
        Binding("escape", "clear_filter", show=False),
        Binding("left", "seek(-1)", show=False),
        Binding("right", "seek(1)", show=False),
        Binding("a", "toggle_auto", show=False),
        Binding("l", "toggle_loop", show=False),
        Binding("plus,equals_sign", "volume(0.05)", show=False),
        Binding("minus", "volume(-0.05)", show=False),
        Binding("z", "toggle_zoom", show=False),
        Binding("i", "toggle_details", show=False),
        Binding("b", "match_tempo", show=False),
        Binding("h", "match_key", show=False),
        Binding("M", "mark_all", show=False),
        Binding("n", "clear_marks", show=False),
        Binding("f", "favorite", show=False),
        Binding("o", "reveal", show=False),
        Binding("y", "copy_path", show=False),
        Binding("R", "rescan", show=False),
        Binding("A", "add_samples", show=False),
        Binding("D", "find_duplicates", show=False),
        Binding("F", "auto_file", show=False),
    ]

    def __init__(self, library: Library, *, initial_query: str = "", no_audio: bool = False,
                 player: Player | None = None):
        super().__init__()
        self.library = library
        self.initial_query = initial_query
        self.no_audio = no_audio
        self.player = player
        self.clips = audio.ClipCache()
        self.space = planes.SoundSpace(library.samples)
        self.analyzer: fx.Analyzer | None = None
        self.analysis_running = False
        self.analysis_text = ""
        self.current: Sample | None = None
        self.current_clip: audio.Clip | None = None
        self.auto_audition = False
        self.loop_enabled = True
        self.prefs_finder_count = 5
        self.finder: FinderScreen | None = None
        self.bank_screen: BankScreen | None = None
        self.bank_name: str | None = None
        self.bank_cursor = 0
        self.pad_target: int | None = None      # the pad you're choosing a sample for, if any
        self._finder_for_pad = False
        self._was_playing = False
        self._play_when_loaded = False
        self._filter_timer: Timer | None = None
        self._load_timer: Timer | None = None
        self._tree_nodes: dict[str, object] = {}
        self._last_tree_shape: tuple = ()
        self._prefs = library.load_prefs()

    def compose(self) -> ComposeResult:
        # Kept as attributes because timers and on_unmount can still use them while the DOM is being torn down
        self.filter_input = FilterInput(value=self.initial_query, placeholder=FILTER_HINT, id="filter")
        self.table = SampleTable(id="table")
        self.viz = Visualizer(id="viz")
        self.titlebar = Static(id="titlebar")
        self.count = Static(id="count")
        self.browse = Tree("All samples", id="browse")
        self.details = Static(id="details")
        self.transport = TransportBar(id="transport")
        yield self.titlebar
        with Horizontal(id="filterbar"):
            yield Static("/", id="slash")
            yield self.filter_input
            yield self.count
        with Horizontal(id="panes"):
            yield self.browse
            yield self.table
            yield self.details
        yield self.transport
        yield self.viz
        yield Footer()

    # ------------------------------------------------------------------ startup and shutdown

    def on_mount(self) -> None:
        prefs = self._prefs
        self.auto_audition = bool(prefs.get("auto", False))
        self.loop_enabled = bool(prefs.get("loop", True))
        self.prefs_finder_count = int(prefs.get("finder_count", 5))
        self.bank_name = prefs.get("bank")
        table = self.table
        table.sort_column = prefs.get("sort", "main") if prefs.get("sort") in SORT_COLUMNS else "main"
        table.sort_reverse = bool(prefs.get("reverse", False))
        if prefs.get("mode") in MODES:
            self.viz.mode = prefs["mode"]
        if self.player is None:
            self.player, warning = create_player(no_audio=self.no_audio, volume=float(prefs.get("volume", 0.8)))
            if warning:
                self.notify(warning, severity="warning")
        self._build_tree()
        self._update_titlebar()
        self.apply_filter()
        table.focus()
        self.set_interval(FRAME_SECONDS, self._tick)
        self._update_transport()
        self._start_analysis()

    def on_unmount(self) -> None:
        if self.analyzer:
            self.analyzer.stop()
        self._save_prefs()
        if self.player:
            self.player.close()

    def _save_prefs(self) -> None:
        table = self.table
        self.library.save_prefs({
            "auto": self.auto_audition, "loop": self.loop_enabled, "sort": table.sort_column,
            "reverse": table.sort_reverse, "mode": self.viz.mode, "finder_count": self.prefs_finder_count,
            "volume": round(self.player.volume, 3) if self.player else 0.8,
            "export_dir": self._prefs.get("export_dir", DEFAULT_EXPORT),
            "bank": self.bank_name, "bank_export_dir": self.bank_export_dir,
        })

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if self.overlay is not None and action in TABLE_ACTIONS:
            return False
        return True

    @property
    def overlay(self) -> FinderScreen | BankScreen | None:
        """The full-screen view in front of the main table, which has its own visualizer and transport bar."""
        return self.finder if self.finder is not None else self.bank_screen

    @property
    def active_viz(self) -> Visualizer:
        overlay = self.overlay
        return overlay.viz if overlay is not None else self.viz

    @property
    def active_bar(self) -> TransportBar:
        overlay = self.overlay
        return overlay.bar if overlay is not None else self.transport

    def _update_titlebar(self) -> None:
        if self.pad_target is not None and self.bank is not None:
            self.titlebar.update(Text.assemble((f"Choosing for pad {self.pad_target + 1} of {self.bank.name}", f"bold {theme.MARK_COLOR}"),
                                               "  ·  p puts the selected sample there  ·  B goes back"))
            return
        library = self.library
        loops = sum(s.type == naming.LOOP for s in library.samples)
        shots = sum(s.type == naming.ONESHOT for s in library.samples)
        home = str(Path.home())
        root = str(library.root)
        root = "~" + root[len(home):] if root.startswith(home) else root
        title = Text.assemble(("♪ crate", "bold"), f"  {root}  ·  {len(library.samples)} samples  ·  {loops} loops  ·  {shots} one-shots")
        if self.analysis_text:
            title.append(f"  ·  {self.analysis_text}", style=theme.MUTED_COLOR)
        if library.read_only:
            title.append("  ·  read-only", style=f"bold {theme.WARNING_COLOR}")
        self.titlebar.update(title)

    # ------------------------------------------------------------------ measuring the library

    def _start_analysis(self) -> None:
        if self.analysis_running:
            return
        if not any(s.features is None for s in self.library.samples):
            self.space = planes.SoundSpace(self.library.samples)
            self._refresh_plane()
            self._refresh_finder_categories()
            return
        self.analysis_running = True
        self.analyzer = fx.Analyzer(self.library)
        self._refresh_plane()
        self._analysis_worker(self.analyzer)

    @work(thread=True, group="analysis", exit_on_error=False)
    def _analysis_worker(self, analyzer: fx.Analyzer) -> None:
        def progress(done: int, total: int) -> None:
            if done % PROGRESS_EVERY == 0 and self.is_running:
                self.call_from_thread(self._show_analysis_progress, done, total)
        analyzer.run(progress)
        if self.is_running:
            self.call_from_thread(self._analysis_finished, analyzer)

    def _show_analysis_progress(self, done: int, total: int) -> None:
        self.analysis_text = f"measuring {round(done / max(total, 1) * 100)}%"
        self._update_titlebar()
        self._refresh_plane()

    def _analysis_finished(self, analyzer: fx.Analyzer) -> None:
        self.analysis_running = False
        self.analysis_text = ""
        self.library.attach_features()
        self.space = planes.SoundSpace(self.library.samples)
        self._update_titlebar()
        self._refresh_plane()
        self._refresh_finder_categories()
        self._update_details()
        if self.filter_input.value and query.needs_features(self.filter_input.value):
            self.apply_filter()
        if analyzer.failed:
            count = len(analyzer.failed)
            self.notify(f"Couldn't measure {count} file{'s' if count != 1 else ''}, so they won't appear on the maps",
                        severity="warning")
        # Files found here that weren't there before were added by hand, so check just those.
        # When everything is unmeasured, this is a first launch rather than new files.
        new = analyzer.measured
        if new and len(new) < len(self.library.samples):
            self._check_new_files(list(new))

    @work(thread=True, exclusive=True, group="duplicates", exit_on_error=False)
    def _check_new_files(self, new_rels: list[str]) -> None:
        """Check files added by hand, such as ones dragged into the library folder. Only ever these files."""
        analyzer = self.analyzer
        groups = duplicates.scan_library(self.library, only_rels=new_rels,
                                         stop=lambda: analyzer is not None and analyzer._stop.is_set())
        wanted = set(new_rels)
        hits = [g for g in groups if any(i.rel in wanted for i in g.items)]
        if self.is_running:
            self.call_from_thread(self._show_new_file_matches, hits)

    def _show_new_file_matches(self, hits: list[duplicates.Group]) -> None:
        if not hits:
            return
        duplicates.flag_in_library(self.library, hits, merge=True)
        self._maybe_rebuild_tree()
        self._update_tree_counts(self.filter_input.value)
        self._update_details()
        if query.needs_duplicates(self.filter_input.value):
            self.apply_filter()
        count = len(hits)
        self.notify(f"{count} new file{'s' if count != 1 else ''} match{'' if count != 1 else 'es'} a sample you already have. See Duplicates in the tree.",
                    severity="warning", timeout=10)

    def _refresh_finder_categories(self) -> None:
        if self.finder is not None:
            self.finder.rebuild_categories()

    # ------------------------------------------------------------------ browse tree

    def _build_tree(self) -> None:
        tree = self.browse
        tree.clear()
        tree.show_root = False
        tree.guide_depth = 2
        self._tree_nodes = {}
        root = tree.root
        root.expand()
        self._tree_nodes["__all__"] = root.add_leaf("All samples", data=None)
        self._tree_nodes["__fav__"] = root.add_leaf("★ Favorites", data="is:fav")
        self._tree_nodes["__marked__"] = root.add_leaf("● Marked", data="is:marked")
        if any(s.problems for s in self.library.samples):
            self._tree_nodes["__invalid__"] = root.add_leaf("! Needs fixing", data="is:invalid")
        if self.library.duplicates:
            self._tree_nodes["__dupes__"] = root.add_leaf("⧉ Duplicates", data="is:dupe")
        if self.library.last_upload:
            self._tree_nodes["__new__"] = root.add_leaf("✚ Last upload", data="is:new")
        genres = self.library.all_genres()
        if genres:
            branch = root.add("# Genres", data=None)
            self._tree_nodes["__genres__"] = branch
            self._tree_nodes["__untagged__"] = branch.add_leaf("Untagged", data="genre:na")
            for genre in genres:
                self._tree_nodes[f"genre:{genre}"] = branch.add_leaf(genre, data=f"genre:{naming.slugify(genre)}")
        folders = sorted(folder_counts(self.library.samples))
        children = defaultdict(list)
        for folder in folders:
            parent, _, _ = folder.rpartition("/")
            children[parent].append(folder)

        def add(node, folder):
            name = folder.rsplit("/", 1)[-1]
            kids = children.get(folder, [])
            item = node.add(name, data=f"in:{folder}", allow_expand=bool(kids)) if kids else node.add_leaf(name, data=f"in:{folder}")
            self._tree_nodes[folder] = item
            for kid in kids:
                add(item, kid)

        for folder in children.get("", []):
            add(root, folder)
        self._last_tree_shape = self._tree_shape()

    def _tree_shape(self) -> tuple:
        return (tuple(sorted(folder_counts(self.library.samples))), tuple(self.library.all_genres()),
                bool(self.library.duplicates), any(s.problems for s in self.library.samples), bool(self.library.last_upload))

    def _maybe_rebuild_tree(self) -> None:
        if self._tree_shape() != self._last_tree_shape:
            self._build_tree()

    def _update_tree_counts(self, text: str) -> None:
        # Counts answer "how many would I see if I clicked here", so each ignores its own kind of filter
        marked = self.table.marked
        no_place = query.set_term(query.set_term(text, "in", None), "is", None)
        base = query.parse(no_place, marked).filter(self.library.samples)
        counts = folder_counts(base)
        no_genre = query.set_term(text, "genre", None)
        genre_base = query.parse(no_genre, marked).filter(self.library.samples)
        genre_counts = Counter(tag for s in genre_base for tag in s.genres)
        specials = {"__all__": len(base), "__fav__": sum(s.favorite for s in base),
                    "__marked__": sum(s.rel in marked for s in base),
                    "__invalid__": sum(bool(s.problems) for s in base),
                    "__dupes__": sum(s.duplicate is not None for s in base),
                    "__new__": sum(s.is_new for s in base),
                    "__untagged__": sum(not s.genres for s in genre_base),
                    "__genres__": sum(bool(s.genres) for s in genre_base)}
        for key, node in self._tree_nodes.items():
            color = None
            if key in specials:
                name = str(node.label).rsplit("  ", 1)[0]
                count = specials[key]
            elif key.startswith("genre:"):
                name = key[len("genre:"):]
                count = genre_counts.get(name, 0)
            else:
                name = key.rsplit("/", 1)[-1]
                count = counts.get(key, 0)
                color = theme.category_color(key) if "/" not in key else None
            label = Text(name, style=color or "")
            label.append(f"  {count}", style=theme.MUTED_COLOR)
            if str(node.label) != str(label):
                node.set_label(label)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.control is not self.browse:
            return
        node = event.node
        term = node.data
        text = self.filter_input.value
        if term is None:
            if node is self._tree_nodes.get("__genres__"):
                return
            # "All samples" clears everything the tree can set
            for field in ("in", "is", "genre"):
                text = query.set_term(text, field, None)
        else:
            field, _, value = term.partition(":")
            if field == "genre":
                text = query.set_term(text, "genre", value)
            else:
                # A folder and a flag are alternatives, but either can combine with a genre
                text = query.set_term(query.set_term(text, "in", None), "is", None)
                text = query.set_term(text, field, value)
        self.filter_input.value = text

    # ------------------------------------------------------------------ filtering and sorting

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filter":
            return
        if self._filter_timer:
            self._filter_timer.stop()
        self._filter_timer = self.set_timer(FILTER_DEBOUNCE, self.apply_filter)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter":
            self.action_focus_table()

    def apply_filter(self, keep_rel: str | None = None) -> None:
        """Run the query string against the library and refresh everything that depends on it."""
        text = self.filter_input.value
        table = self.table
        parsed = query.parse(text, table.marked)
        rows = sort_samples(parsed.filter(self.library.samples), table.sort_column, table.sort_reverse)
        table.set_rows(rows, keep_rel)
        self.filter_input.set_class(bool(parsed.errors), "-invalid")
        self._update_count(parsed)
        self._update_tree_counts(text)
        self._refresh_plane()

    def _update_count(self, parsed: query.Query | None = None) -> None:
        table = self.table
        parts = [f"{len(table.rows)}/{len(self.library.samples)}"]
        if table.marked:
            parts.append(f"{len(table.marked)} marked")
        count = Text("  ·  ".join(parts))
        if parsed and parsed.errors:
            count = Text(parsed.errors[0], style=theme.WARNING_COLOR)
        elif query.needs_duplicates(self.filter_input.value) and not self.library.duplicates_checked and not self.library.duplicates:
            # The check only runs when asked, so an empty list here doesn't mean there are none
            count = Text("No check run yet. Press D to check for duplicates.", style=theme.WARNING_COLOR)
        self.count.update(count)

    def action_sort(self, column: str) -> None:
        table = self.table
        if table.sort_column == column:
            table.sort_reverse = not table.sort_reverse
        else:
            table.sort_column, table.sort_reverse = column, False
        self.apply_filter()

    def on_sample_table_sort_requested(self, message: SampleTable.SortRequested) -> None:
        self.action_sort(message.column)

    def action_reverse_sort(self) -> None:
        self.table.sort_reverse = not self.table.sort_reverse
        self.apply_filter()

    def action_focus_filter(self) -> None:
        self.filter_input.focus()

    def action_focus_table(self) -> None:
        self.table.focus()

    def action_clear_filter(self) -> None:
        if self.filter_input.value:
            self.filter_input.value = ""
        elif self.pad_target is not None:
            self.action_open_banks()

    def action_match_tempo(self) -> None:
        self._toggle_match("bpm", self.current.bpm if self.current else None, "tempo")

    def action_match_key(self) -> None:
        self._toggle_match("key", self.current.key if self.current else None, "key")

    def _toggle_match(self, field: str, value, label: str) -> None:
        if value is None:
            self.notify(f"This sample has no {label} to match", severity="warning")
            return
        term = f"~{value}"
        existing = f"{field}:{term}"
        text = self.filter_input.value
        if existing in query.tokenize(text):
            self.filter_input.value = query.set_term(text, field, None)
        else:
            self.filter_input.value = query.set_term(text, field, term)

    # ------------------------------------------------------------------ selecting and playing

    def on_sample_table_highlighted(self, message: SampleTable.Highlighted) -> None:
        # Closing the Finder or the bank builder announces the table's sample again
        if self.overlay is None:
            self.select_sample(message.sample, by_user=message.by_user)

    def select_sample(self, sample: Sample | None, *, by_user: bool) -> None:
        """Make a sample current. This is the one place that decides whether selecting something plays it.

        With Play on select on, choosing a sample plays it. Otherwise nothing plays until you ask,
        and anything already playing stops, so the sound never disagrees with what's on screen.
        A filter or sort moving the cursor doesn't count as you choosing.
        """
        self.current = sample
        self.current_clip = None
        self._update_details()
        if self._load_timer:
            self._load_timer.stop()
        play = by_user and self.auto_audition and sample is not None
        self._play_when_loaded = play
        if not play and self.player:
            self.player.stop()
        if sample is None:
            self.active_viz.set_clip(None, message="No samples match this filter")
            self._refresh_plane()
            self._update_transport()
            return
        self._load_timer = self.set_timer(LOAD_DEBOUNCE, lambda: self._load(sample))
        self.set_timer(LOADING_MESSAGE_DELAY, lambda: self._show_loading(sample))
        self._refresh_plane()
        self._update_transport()

    def _show_loading(self, sample: Sample) -> None:
        if sample is self.current and self.current_clip is None:
            self.active_viz.set_clip(None, message="Loading…")

    @work(thread=True, exclusive=True, group="decode", exit_on_error=False)
    def _load(self, sample: Sample) -> None:
        try:
            clip = self.clips.get(sample.path)
        except Exception as exc:
            if self.is_running:
                self.call_from_thread(self._on_load_failed, sample, str(exc))
            return
        if self.is_running:
            self.call_from_thread(self._on_loaded, sample, clip)

    def _on_load_failed(self, sample: Sample, error: str) -> None:
        if sample is self.current:
            self.active_viz.set_clip(None, message=f"Can't read this file: {error}")
            self._play_when_loaded = False
            self.player.stop()

    def _on_loaded(self, sample: Sample, clip: audio.Clip) -> None:
        if sample is not self.current:
            return
        self.current_clip = clip
        self.active_viz.set_clip(clip, theme.category_color(sample.main), sample.bpm)
        self._update_details()
        if self._play_when_loaded:
            self._play_when_loaded = False
            self._start_playback(0.0)
        self._update_transport()

    def _should_loop(self, sample: Sample) -> bool:
        return self.loop_enabled and sample.type == naming.LOOP

    def _start_playback(self, start: float) -> None:
        clip, sample = self.current_clip, self.current
        if clip is None or sample is None:
            return
        self.player.play(clip.audio, clip.sr, start=start, loop=self._should_loop(sample))

    def play_current_from_start(self) -> None:
        if self.current_clip is not None:
            self._start_playback(0.0)
        elif self.current is not None:
            self._play_when_loaded = True

    def on_sample_table_activated(self, message: SampleTable.Activated) -> None:
        self.play_current_from_start()

    def action_toggle_play(self) -> None:
        player = self.player
        if player.is_playing:
            player.pause()
        elif self.current_clip is not None:
            start = player.position if player.position < self.current_clip.duration else 0.0
            self._start_playback(start)
        elif self.current is not None:
            self._play_when_loaded = True
        self._update_transport()

    def action_stop_audio(self) -> None:
        """Silence everything now, including a sound still waiting to load."""
        self._play_when_loaded = False
        self.player.stop()
        self.active_viz.set_transport(0.0, False)
        self._update_transport()

    def action_seek(self, direction: int) -> None:
        clip = self.current_clip
        if clip is None:
            return
        step = max(clip.duration * 0.05, 0.05)
        target = min(max(self.player.position + direction * step, 0.0), clip.duration)
        self.player.seek(target)
        self.active_viz.set_transport(target, self.player.is_playing)
        self._update_transport()

    def on_visualizer_seek(self, message: Visualizer.Seek) -> None:
        clip = self.current_clip
        if clip is None:
            return
        target = message.fraction * clip.duration
        if self.player.is_playing:
            self.player.seek(target)
        else:
            self._start_playback(target)

    def on_visualizer_plane_clicked(self, message: Visualizer.PlaneClicked) -> None:
        """Clicking a dot selects that sample, which plays it if Play on select is on. A double-click always plays."""
        frame = self.viz.plane
        if frame is None or not frame.dots:
            return
        x, y = message.point
        dot = min(frame.dots, key=lambda d: (d.x - x) ** 2 + (d.y - y) ** 2)
        if (dot.x - x) ** 2 + (dot.y - y) ** 2 > DOT_CLICK_REACH ** 2:
            return
        index = next((i for i, s in enumerate(self.table.rows) if s.rel == dot.rel), None)
        if index is None:
            return
        self.table.cursor = index
        if message.double:
            self.play_current_from_start()

    def action_toggle_auto(self) -> None:
        self.auto_audition = not self.auto_audition
        self._update_transport()

    def action_toggle_loop(self) -> None:
        self.loop_enabled = not self.loop_enabled
        if self.current:
            self.player.set_loop(self._should_loop(self.current))
        self._update_transport()

    def action_volume(self, delta: float) -> None:
        self.player.volume = self.player.volume + delta
        self._update_transport()

    # ------------------------------------------------------------------ frame clock

    def _tick(self) -> None:
        player = self.player
        if player is None:
            return
        playing = player.is_playing
        if playing or self._was_playing:
            self.active_viz.set_transport(player.position, playing)
            self._update_transport()
        self._was_playing = playing
        player.release_if_idle()
        if player.error:
            self.notify(f"Audio problem, playing silently: {player.error}", severity="warning")
            player.error = None

    def _update_transport(self) -> None:
        player = self.player
        if player is None:
            return
        clip = self.current_clip
        self.active_bar.show(TransportState(
            playing=player.is_playing,
            position=format_time(player.position),
            duration=format_time(clip.duration) if clip else "-:--.-",
            play_on_select=self.auto_audition,
            loop=self.loop_enabled,
            volume=round(player.volume * 100),
            view=MODE_LABELS[self.active_viz.mode],
        ))

    def action_next_mode(self) -> None:
        viz = self.active_viz
        # The Finder already shows a plane, so its panel and the bank builder's cycle only the audio views
        modes = AUDIO_MODES if self.overlay is not None else MODES
        index = modes.index(viz.mode) if viz.mode in modes else -1
        viz.set_mode(modes[(index + 1) % len(modes)])
        self._refresh_plane()
        self._update_transport()

    def action_toggle_zoom(self) -> None:
        self.screen.toggle_class("-zoomed")

    def action_toggle_details(self) -> None:
        self.screen.toggle_class("-no-details")

    def on_resize(self) -> None:
        self._update_transport()

    # ------------------------------------------------------------------ the sound maps

    def _refresh_plane(self) -> None:
        viz = self.viz
        if viz.mode not in PLANE_MODES or not hasattr(self, "table"):
            return
        frame, message = self._plane_frame(viz.mode)
        viz.set_plane(frame, message)

    def _plane_frame(self, kind: str) -> tuple[PlaneFrame | None, str]:
        sample = self.current
        if self.analysis_running:
            return None, f"Measuring your library… {self.analysis_text.removeprefix('measuring ')}"
        if sample is None:
            return None, "No samples match this filter"
        if sample.features is None:
            return None, "This file couldn't be measured"
        rows = self.table.rows
        category = planes.category_of(sample)
        color = theme.category_color(sample.main)
        space = self.space
        if kind == "character":
            if category is None:
                return None, "Give this sample a valid name to place it on a map (press e)"
            spec = planes.character_spec(category)
            dots = [Dot(*space.character[s.rel], color, s.rel) for s in rows
                    if planes.category_of(s) == category and s.rel in space.character]
            point = space.character.get(sample.rel)
            if point is None:
                return None, "This sample hasn't been measured yet"
            return PlaneFrame(spec, dots, point, readout=planes.describe_character(spec, point), accent=color), ""
        dots = []
        for s in rows:
            position = space.footprint.get(s.rel)
            if position is not None:
                dots.append(Dot(position[0], position[1], color if planes.category_of(s) == category else None, s.rel))
        point = space.footprint.get(sample.rel)
        if point is None:
            return None, "This sample hasn't been measured yet"
        span = (planes.frequency_position(max(sample.features.low_hz, 1.0)),
                planes.frequency_position(max(sample.features.high_hz, 1.0)))
        return PlaneFrame(planes.footprint_spec(), dots, point, span,
                          readout=planes.describe_footprint(sample), accent=color), ""

    def _update_details(self) -> None:
        panel = self.details
        sample = self.current
        if sample is None:
            panel.update("")
            return
        text = Text()
        def row(label: str, value: str, style: str = "") -> None:
            text.append(f"{label:<9}", style=theme.MUTED_COLOR)
            text.append(f"{value}\n", style=style)
        cat = theme.category_color(sample.main)
        text.append(f"{sample.desc}\n\n", style=f"bold {cat}")
        row("Main", sample.main, cat)
        row("Sub", sample.sub or "-", cat)
        row("Type", sample.type or "-")
        row("BPM", str(sample.bpm) if sample.bpm else "-")
        if sample.bars:
            row("Bars", f"{sample.bars:.1f}")
        code = naming.camelot(sample.key)
        row("Key", (sample.key or "-") + (f"  ({code})" if code else ""), theme.key_color(sample.key) if sample.key else "")
        row("Length", f"{sample.length:.2f}s" if sample.length is not None else "-")
        features = sample.features
        if features:
            row("Audible", f"{features.audible:.2f}s")
            row("Range", features.range)
        if sample.genres:
            row("Genres", ", ".join(sample.genres))
        pads = self.library.pads_holding(sample.rel)
        if pads:
            row("Pads", ", ".join(f"{bank} {pad + 1}" for bank, pad in pads))
        if sample.duplicate:
            text.append(f"\n{sample.duplicate.kind} as\n", style=theme.WARNING_COLOR)
            for other in sample.duplicate.others[:3]:
                text.append(f"{Path(other).name}\n")
            text.append("\n")
        point = self.space.character.get(sample.rel)
        category = planes.category_of(sample)
        word = planes.corner_word(planes.character_spec(category), point) if point and category else None
        if word:
            row("Sounds", word, f"bold {cat}")
        clip = self.current_clip
        if clip:
            text.append("\n")
            row("Format", clip.format)
            row("Rate", f"{clip.sr} Hz, {clip.channels} ch")
            row("Peak", f"{clip.peak_db:.1f} dB")
            row("RMS", f"{clip.rms_db:.1f} dB")
        index = sample.index
        if index:
            text.append("\n")
            if index.get("original_name"):
                text.append("Original\n", style=theme.MUTED_COLOR)
                text.append(f"{index['original_name']}\n")
            sources = [(label, index.get(field)) for label, field in (("BPM from", "bpm_source"), ("Key from", "key_source"))]
            for label, value in sources:
                if value:
                    row(label, value)
            if index.get("key_guess_low_confidence") and not sample.key:
                row("Key guess", index["key_guess_low_confidence"] + "?", theme.MUTED_COLOR)
        if sample.problems:
            text.append("\nProblems\n", style=theme.WARNING_COLOR)
            for problem in sample.problems:
                text.append(f"{problem}\n", style=theme.WARNING_COLOR)
        panel.update(text)

    # ------------------------------------------------------------------ the Finder

    def action_open_finder(self) -> None:
        self._finder_for_pad = self.pad_target is not None
        self.finder = FinderScreen(self.prefs_finder_count)
        self.player.stop()
        self._play_when_loaded = False
        self.push_screen(self.finder, self._finder_closed)

    def _finder_closed(self, chosen: Sample | None) -> None:
        self.finder = None
        self.player.stop()
        self._play_when_loaded = False
        table = self.table
        table._announced = None
        if chosen is not None:
            # Show it in the library, widening the filter only if the filter is hiding it
            if chosen not in table.rows:
                self.filter_input.value = ""
                table.set_rows(sort_samples(self.library.samples, table.sort_column, table.sort_reverse), chosen.rel)
            else:
                table.set_rows(table.rows, chosen.rel)
        else:
            table._announce(by_user=False)
        self.apply_filter()
        self._update_transport()
        table.focus()
        if self._finder_for_pad and chosen is None:
            self.action_open_banks()

    # ------------------------------------------------------------------ banks

    @property
    def bank(self) -> Bank | None:
        """The bank that samples get put on: the one you last used, or else the first."""
        banks_found = self.library.banks
        named = self.library.bank_named(self.bank_name) if self.bank_name else None
        return named or (banks_found[0] if banks_found else None)

    @property
    def bank_export_dir(self) -> str:
        return self._prefs.get("bank_export_dir", DEFAULT_EXPORT)

    def action_open_banks(self) -> None:
        self.pad_target = None
        self._update_titlebar()
        if self.bank is None:
            if not self._writable():
                return
            self.bank_name = organize.new_bank(self.library, organize.next_bank_name(self.library)).name
        self.player.stop()
        self._play_when_loaded = False
        self.bank_screen = BankScreen(self.bank_cursor)
        self.push_screen(self.bank_screen, self._bank_closed)

    def _bank_closed(self, next_step: str | None) -> None:
        self.bank_cursor = self.bank_screen.cursor
        self.bank_name = self.bank.name if self.bank else None
        self.bank_screen = None
        self.player.stop()
        self._play_when_loaded = False
        table = self.table
        table._announced = None
        table._announce(by_user=False)
        self._update_titlebar()
        self._update_transport()
        table.focus()
        if next_step == USE_FINDER:
            self.action_open_finder()

    def choose_for_pad(self, pad: int, next_step: str | None = None) -> None:
        """Leave the bank builder to find a sample for this pad. p puts it there, and that brings you back."""
        if self._writable():
            self.pad_target = pad
            self.bank_screen.dismiss(next_step)

    def _put_on_target(self, sample: Sample) -> None:
        pad, bank = self.pad_target, self.bank
        self.pad_target = None
        organize.change_bank(self.library, bank, f"put {sample.desc} on pad {pad + 1} of {bank.name}",
                             lambda target: target.put(pad, sample.rel))

    def _fill_free_pads(self, samples: list[Sample]) -> None:
        bank = self.bank or organize.new_bank(self.library, organize.next_bank_name(self.library))
        self.bank_name = bank.name
        fill = organize.fill_bank(self.library, bank, samples)
        parts = []
        if len(fill.placed) == 1:
            parts.append(f"{samples[0].desc if len(samples) == 1 else 'Added'} is on pad {fill.placed[0] + 1} of {bank.name}")
        elif fill.placed:
            parts.append(f"Put {len(fill.placed)} on {bank.name}, starting at pad {fill.placed[0] + 1}")
        if fill.already:
            parts.append(f"{fill.already} {'was' if fill.already == 1 else 'were'} already there")
        if fill.left_over:
            parts.append(f"{fill.left_over} didn't fit because the bank is full")
        self.notify(". ".join(parts) + ". B shows the bank.", severity="warning" if fill.left_over or not fill.placed else "information")
        self._update_details()

    def action_put_on_pad(self) -> None:
        if not self._writable():
            return
        if self.pad_target is not None:
            if self.current is not None:
                self._put_on_target(self.current)
                self.action_open_banks()
            return
        targets = self._targets()
        if targets:
            self._fill_free_pads(targets)

    def put_from_finder(self, sample: Sample) -> None:
        if not self._writable():
            return
        if self.pad_target is None:
            self._fill_free_pads([sample])
        else:
            self._put_on_target(sample)
            self.finder.dismiss(None)

    def undo_in_banks(self) -> None:
        if not self._writable():
            return
        screen = self.bank_screen
        position = self.library.banks.index(self.bank)
        try:
            label = organize.undo(self.library)
        except OSError as exc:
            self.notify(str(exc), severity="error")
            return
        if label is None:
            self.notify("Nothing to undo")
            return
        self.table.marked &= set(self.library.by_rel)
        self._after_change()
        self.notify(f"Undid: {label}")
        found = self.library.banks
        if not found:
            screen.dismiss(None)
            return
        if self.library.bank_named(self.bank_name or "") is None:
            self.bank_name = found[min(position, len(found) - 1)].name
        screen.show_pad(by_user=False)

    @work(thread=True, group="preload", exit_on_error=False)
    def preload_clips(self, samples: list[Sample]) -> None:
        """Decode a kit ahead of time, so the first hit on each pad isn't late."""
        for sample in samples:
            try:
                self.clips.get(sample.path)
            except Exception:
                pass    # Playing it will report the problem

    def run_bank_export(self, bank: Bank, dest: str) -> None:
        self._prefs["bank_export_dir"] = dest
        self._bank_export_worker(bank, dest)

    @work(thread=True, group="export", exit_on_error=False)
    def _bank_export_worker(self, bank: Bank, dest: str) -> None:
        try:
            result = organize.export_bank(self.library, bank, dest)
        except (ValueError, OSError) as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        self.call_from_thread(self.push_screen, ReportScreen(f"Export {bank.name}", "\n".join(banks.describe_export(result))))

    # ------------------------------------------------------------------ marks

    def action_mark(self) -> None:
        table = self.table
        sample = table.current
        if sample is None:
            return
        table.marked.symmetric_difference_update({sample.rel})
        table.action_move(1)
        table.refresh()
        self._update_count()
        self._update_tree_counts(self.filter_input.value)

    def action_mark_all(self) -> None:
        self.table.marked.update(s.rel for s in self.table.rows)
        self.table.refresh()
        self._update_count()
        self._update_tree_counts(self.filter_input.value)

    def action_clear_marks(self) -> None:
        self.table.marked.clear()
        self.table.refresh()
        self.apply_filter()

    def _targets(self) -> list[Sample]:
        """What an action applies to: the marked samples if there are any, otherwise the selected one."""
        table = self.table
        if table.marked:
            return [self.library.by_rel[rel] for rel in sorted(table.marked) if rel in self.library.by_rel]
        return [self.current] if self.current else []

    # ------------------------------------------------------------------ organizing

    def _writable(self) -> bool:
        if self.library.read_only:
            self.notify("The library was opened read-only", severity="warning")
            return False
        return True

    def action_favorite(self) -> None:
        if not self._writable():
            return
        targets = self._targets()
        if not targets:
            return
        turn_on = not all(s.favorite for s in targets)
        for sample in targets:
            if sample.favorite != turn_on:
                self.library.toggle_favorite(sample)
        self.table.refresh()
        self._update_tree_counts(self.filter_input.value)

    def action_genres(self) -> None:
        if not self._writable():
            return
        targets = self._targets()
        if targets:
            self.push_screen(GenreScreen(self.library, targets), lambda result: self._apply_genres(targets, result))

    def _apply_genres(self, targets: list[Sample], result: tuple[list[str], list[str]] | None) -> None:
        if result is None:
            return
        add, remove = result
        changed = organize.set_genres(self.library, targets, add, remove)
        if not changed:
            return
        self._maybe_rebuild_tree()
        self.apply_filter()
        self._update_details()
        self.table.refresh()
        self.notify(f"Updated genres on {changed} file{'s' if changed != 1 else ''}. Press u to undo.")

    def action_edit(self) -> None:
        if not self._writable():
            return
        targets = self._targets()
        if targets:
            self.push_screen(EditScreen(self.library, targets), lambda changes: self._apply_edit(targets, changes))

    def _apply_edit(self, targets: list[Sample], changes: organize.Changes | None) -> None:
        if changes is None:
            return
        try:
            moves = organize.plan_retag(self.library, targets, changes)
            organize.apply_moves(self.library, moves, f"retag {len(moves)} file(s)")
        except (ValueError, OSError) as exc:
            self.notify(str(exc), severity="error")
            return
        if not moves:
            self.notify("Nothing changed")
            return
        renamed = {m.src: m.dst for m in moves}
        for src in renamed:
            self.clips.forget(self.library.root / src)
        table = self.table
        table.marked = {renamed.get(rel, rel) for rel in table.marked}
        keep = table.current.rel if table.current else None
        self._after_change(renamed.get(keep, keep))
        count = len(moves)
        self.notify(f"Updated {count} file{'s' if count != 1 else ''}. Press u to undo.")

    def action_trash(self) -> None:
        if not self._writable():
            return
        targets = self._targets()
        if not targets:
            return
        title = f"Move {len(targets)} samples to the Trash?" if len(targets) > 1 else "Move this sample to the Trash?"
        detail = targets[0].filename if len(targets) == 1 else "You can undo this with u."
        chosen = {s.rel for s in targets}
        whole_sets = sum(1 for s in targets if s.duplicate and s.duplicate.keeper and set(s.duplicate.others) <= chosen)
        if whole_sets:
            detail += (f"\n\nThis includes every copy of {whole_sets} sound{'s' if whole_sets != 1 else ''}. "
                       "To keep one of each, filter with is:copy instead.")
        self.push_screen(ConfirmScreen(title, detail), lambda yes: self._apply_trash(targets) if yes else None)

    def _apply_trash(self, targets: list[Sample]) -> None:
        table = self.table
        index = table.cursor
        gone = {s.rel for s in targets}
        if self.current and self.current.rel in gone:
            self.player.stop()
        try:
            count = organize.trash(self.library, targets)
        except OSError as exc:
            self.notify(str(exc), severity="error")
            return
        for sample in targets:
            self.clips.forget(sample.path)
        table.marked -= gone
        # Stay at the same row number, so the next sample slides under the cursor
        after = next((r.rel for r in table.rows[index:] if r.rel not in gone), None)
        self._after_change(after)
        self.notify(f"Moved {count} file{'s' if count != 1 else ''} to the Trash. Press u to undo.")

    def action_undo(self) -> None:
        if not self._writable():
            return
        try:
            label = organize.undo(self.library)
        except OSError as exc:
            self.notify(str(exc), severity="error")
            return
        if label is None:
            self.notify("Nothing to undo")
            return
        self.table.marked.clear()
        self._after_change()
        self.notify(f"Undid: {label}")

    def action_export(self) -> None:
        targets = self._targets()
        if targets:
            default = self._prefs.get("export_dir", DEFAULT_EXPORT)
            self.push_screen(ExportScreen(len(targets), default), lambda dest: self._run_export(targets, dest))

    def _run_export(self, targets: list[Sample], dest: str | None) -> None:
        if dest:
            self._prefs["export_dir"] = dest
            self._export_worker(targets, dest)

    @work(thread=True, group="export", exit_on_error=False)
    def _export_worker(self, targets: list[Sample], dest: str) -> None:
        try:
            result = organize.export(self.library, targets, dest)
        except (ValueError, OSError) as exc:
            self.call_from_thread(self.notify, str(exc), severity="error")
            return
        message = f"Copied {result.copied} to {dest}"
        if result.skipped:
            message += f" ({result.skipped} already there)"
        if result.conversion_note:
            message += f"; {result.conversion_note}"
        self.call_from_thread(self.notify, message)

    def action_add_samples(self) -> None:
        if not self._writable():
            return
        if self.analysis_running:
            self.notify("Still measuring your library. Try again in a moment.", severity="warning")
            return
        self.push_screen(PathScreen(
            "Add samples", "A file, a folder or a zip. Your originals stay where they are. Anything you already have is left out, and the rest is filed automatically.",
            "Check and add", placeholder="~/Downloads/new-pack.zip"), self._run_add)

    def _run_add(self, source: str | None) -> None:
        if source:
            self.analysis_text = "checking new samples"
            self._update_titlebar()
            self._add_worker(source)

    @work(thread=True, group="add", exit_on_error=False)
    def _add_worker(self, source: str) -> None:
        library = self.library
        try:
            with duplicates.expand_sources([source]) as sources:
                if not sources.files:
                    raise ValueError(sources.problems[0] if sources.problems else "no audio files found there")
                items, failed = duplicates.measure_files(sources.files)
                check = duplicates.check_upload(library, items, failed)
                proposals = filing.propose_for_upload(library, check.new, sources)
                added = organize.add_files(library, check.new, sources.origins, proposals)
        except (ValueError, OSError) as exc:
            if self.is_running:
                self.call_from_thread(self._add_failed, str(exc))
            return
        if self.is_running:
            self.call_from_thread(self._add_finished, check, added, proposals)

    def _add_failed(self, error: str) -> None:
        self.analysis_text = ""
        self._update_titlebar()
        self.notify(error, severity="error")

    def _add_finished(self, check: duplicates.UploadCheck, added: list[str], proposals: dict) -> None:
        self.analysis_text = ""
        self.clips = audio.ClipCache()
        self._after_change()
        lines = duplicates.describe_upload(check, len(added), proposals=proposals)
        if added:
            lines += ["", "Review them under Last upload in the tree, and press e to correct any. u undoes the whole upload."]
        self.push_screen(ReportScreen("Add samples", "\n".join(lines)))

    def action_auto_file(self) -> None:
        """Name and file what's waiting under Needs fixing: the marked ones if any are, otherwise all of them."""
        if not self._writable():
            return
        table = self.table
        waiting = [s for s in self.library.samples if s.problems]
        marked = [s for s in waiting if s.rel in table.marked]
        targets = marked or waiting
        if not targets:
            self.notify("Nothing is waiting to be filed")
            return
        self._auto_file_worker(targets)

    @work(thread=True, group="add", exit_on_error=False)
    def _auto_file_worker(self, targets: list[Sample]) -> None:
        proposals = filing.propose_for_samples(self.library, targets)
        if self.is_running:
            self.call_from_thread(self._apply_auto_file, targets, proposals)

    def _apply_auto_file(self, targets: list[Sample], proposals: dict) -> None:
        try:
            moves = organize.file_samples(self.library, targets, proposals)
        except OSError as exc:
            self.notify(str(exc), severity="error")
            return
        renamed = {m.src: m.dst for m in moves}
        table = self.table
        table.marked = {renamed.get(rel, rel) for rel in table.marked}
        keep = table.current.rel if table.current else None
        self._after_change(renamed.get(keep, keep))
        lines = [f"Filed {len(moves)} of {len(targets)}.", ""]
        lines += [f"{s.filename}: {proposals[s.rel].why}" for s in targets if not proposals[s.rel].name]
        lines += ["", "u undoes this."]
        self.push_screen(ReportScreen("Auto-file", "\n".join(lines)))

    def action_find_duplicates(self) -> None:
        if self.analysis_running:
            self.notify("Still measuring your library. Try again in a moment.", severity="warning")
            return
        self.push_screen(PathScreen(
            "Check for duplicates", "Leave this blank to check the whole library, or give a folder or zip to compare with it. Nothing gets changed.",
            "Check", placeholder="Whole library", blank_ok=True), self._run_duplicate_check)

    def _run_duplicate_check(self, source: str | None) -> None:
        if source is not None:
            self._duplicate_worker(source)

    @work(thread=True, group="duplicates-manual", exit_on_error=False)
    def _duplicate_worker(self, source: str) -> None:
        library = self.library
        try:
            if not source:
                groups = duplicates.scan_library(library)
                checked = len(library.samples)
            else:
                with duplicates.expand_sources([source]) as sources:
                    if not sources.files:
                        raise ValueError(sources.problems[0] if sources.problems else "no audio files found there")
                    items, _ = duplicates.measure_files(sources.files)
                    groups = duplicates.find_groups(items + duplicates.library_items(library), only=items)
                    checked = len(items)
        except (ValueError, OSError) as exc:
            if self.is_running:
                self.call_from_thread(self.notify, str(exc), severity="error")
            return
        if self.is_running:
            self.call_from_thread(self._duplicate_check_finished, source, groups, checked)

    def _duplicate_check_finished(self, source: str, groups: list[duplicates.Group], checked: int) -> None:
        if not source:
            duplicates.flag_in_library(self.library, groups)
            self._maybe_rebuild_tree()
            self.apply_filter()
        same = sum(g.kind == duplicates.SAME_AUDIO for g in groups)
        head = f"{checked} files checked: {same} sets of identical audio, {len(groups) - same} likely."
        lines = [head, ""] + duplicates.describe_groups(groups)
        if groups and not source:
            lines.append("Filter with is:dupe to see every copy, or is:copy for all but the suggested keeper. Nothing was changed.")
        self.push_screen(ReportScreen("Duplicates", "\n".join(lines)))

    def action_rescan(self) -> None:
        self.library.scan()
        self._after_change()
        self.notify(f"Found {len(self.library.samples)} samples")

    def _after_change(self, keep_rel: str | None = None) -> None:
        """Rebuild the views after files have moved. Samples are new objects now, so look them up again."""
        self._build_tree()
        self._update_titlebar()
        self.table._announced = None
        self.current = None
        self.space = planes.SoundSpace(self.library.samples)
        self.apply_filter(keep_rel)
        self._start_analysis()

    def action_reveal(self) -> None:
        if self.current:
            subprocess.run(["open", "-R", str(self.current.path)], check=False)

    def action_copy_path(self) -> None:
        if not self.current:
            return
        path = str(self.current.path)
        try:
            subprocess.run(["pbcopy"], input=path.encode(), check=True)
        except (OSError, subprocess.CalledProcessError):
            self.copy_to_clipboard(path)
        self.notify("Copied the path")

    def action_help(self) -> None:
        self.push_screen(HelpScreen())
