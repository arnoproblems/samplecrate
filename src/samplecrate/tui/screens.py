"""Modal dialogs: editing metadata, confirming, exporting, and help."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Input, Label, SelectionList, Static

from samplecrate import filing
from samplecrate import genres as genre_tags
from samplecrate import naming, organize
from samplecrate.library import Library, Sample

HELP_TEXT = """\
[b]Browsing[/b]
  ↑ ↓  j k       move                  PgUp PgDn   page
  g G            top / bottom          Tab         switch pane
  /              filter                Esc         clear filter
  1-8            sort by column        r           reverse sort

[b]Listening[/b]
  Space          play / pause          Enter       play from the start
  s              stop                  ctrl+g      stop, even while typing
  a              play on select        l           loop
  ← →            seek                  click wave  jump there
  + -            volume

[b]Views[/b]
  v              waveform, spectrum, scope, spectrogram, character, footprint
  z              enlarge panel         i           details panel

[b]Find a sound[/b]
  t              open the Finder: pick a category, place a dot, hear the closest
  b              same tempo (±3%, half and double time)
  h              harmonically compatible keys

[b]Organizing[/b]
  m              mark, then next       M           mark all shown
  n              clear marks           f           favorite
  #              genres                e           edit metadata
  x              export to a folder    d           move to Trash
  u              undo                  R           rescan the library
  A              add samples           D           check for duplicates
  F              auto-file what's under Needs fixing
  o              show in Finder        y           copy path

[b]Banks[/b]  (16 pads, laid out like the SP-404MK2's)
  B              open the bank builder
  p              put on the next empty pad, or on the pad you're choosing for
  Inside the bank builder:
  ↑ ↓ ← →        move between pads     Enter       play from the start
  p              pick from the library t           pick with the Finder
  g              pick up and carry     Backspace   clear the pad
  k              kit mode: 1234 qwer asdf zxcv play the pads
  x              export, numbered by pad
  [ ]            previous / next bank  n           new bank
  r              rename the bank       D           delete the bank
  u              undo

[b]Filter terms[/b]  (combine freely; put - in front of any to exclude it)
  kick dark              words, also searching original filenames
  main:drums  sub:kick   category; sub:=Hat for exactly that sub
  type:loop  type:shot
  bpm:120  bpm:120-130  bpm:>140  bpm:~120  bpm:na
  key:Am  key:Am,Cmaj  key:~Am  key:8A  key:na
  genre:house  genre:house,techno  genre:na
  range:low  range:mid  range:high
  len:<2  len:1-4        file length in seconds
  audible:<1             seconds of actual sound
  desc:vinyl             description contains
  in:Drums/Loops/Kick    folder
  is:fav  is:marked  is:invalid
  is:dupe (every copy)  is:copy (all but the one to keep)
  is:new                 from the most recent upload

[dim]Esc or ? closes this page.[/dim]
"""


# Dialogs block the app's own keys, so each one needs the stop-audio key for itself
STOP_BINDING = Binding("ctrl+g", "app.stop_audio", show=False, priority=True)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "Close"), STOP_BINDING]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-box"):
            yield Static(HELP_TEXT)

    def action_dismiss(self, result=None) -> None:
        self.dismiss(result)


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No"), STOP_BINDING]

    def __init__(self, title: str, detail: str = ""):
        super().__init__()
        self.title_text, self.detail = title, detail

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.detail:
                yield Static(self.detail, classes="dialog-detail")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Yes (y)", id="yes", variant="error")
                yield Button("No (n)", id="no")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class PathScreen(ModalScreen[str | None]):
    """Ask for a file, folder or name. Blank is allowed when `blank_ok`, for "the whole library"."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"), STOP_BINDING]

    def __init__(self, title: str, detail: str, button: str, *, value: str = "", placeholder: str = "", blank_ok: bool = False):
        super().__init__()
        self.title_text, self.detail, self.button = title, detail, button
        self.value, self.placeholder, self.blank_ok = value, placeholder, blank_ok

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(self.detail, classes="dialog-detail")
            yield Input(value=self.value, placeholder=self.placeholder, id="path-input")
            with Horizontal(classes="dialog-buttons"):
                yield Button(self.button, id="go", variant="primary")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _finish(self) -> None:
        value = self.query_one("#path-input", Input).value.strip()
        if value or self.blank_ok:
            self.dismiss(value)

    def on_input_submitted(self) -> None:
        self._finish()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._finish()
        else:
            self.dismiss(None)


class ReportScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,q,enter", "dismiss", "Close"), STOP_BINDING]

    def __init__(self, title: str, text: str):
        super().__init__()
        self.title_text, self.text = title, text

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="report-box"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(self.text, markup=False, id="report-text")
            yield Static("Esc closes this page.", classes="dialog-detail")

    def action_dismiss(self, result=None) -> None:
        self.dismiss(result)


class ExportScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel"), STOP_BINDING]

    def __init__(self, count: int, default_dir: str):
        super().__init__()
        self.count, self.default_dir = count, default_dir

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(f"Export {self.count} sample{'s' if self.count != 1 else ''}", classes="dialog-title")
            yield Static("Copies into this folder, ready for the SP-404: names lose their NA fields, and float files "
                         "become 24-bit. Existing files are left alone.", classes="dialog-detail")
            yield Input(value=self.default_dir, id="export-dir")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Copy", id="go", variant="primary")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _finish(self) -> None:
        value = self.query_one("#export-dir", Input).value.strip()
        self.dismiss(value or None)

    def on_input_submitted(self) -> None:
        self._finish()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "go":
            self._finish()
        else:
            self.dismiss(None)


class EditScreen(ModalScreen[organize.Changes | None]):
    """Edit one sample's metadata, or apply shared changes to many. Shows the resulting path before anything moves."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("ctrl+s", "apply", "Apply"), STOP_BINDING]
    FIELDS = ("main", "sub", "type", "bpm", "key", "desc")

    def __init__(self, library: Library, samples: list[Sample]):
        super().__init__()
        self.library, self.samples = library, samples
        self.bulk = len(samples) > 1
        self._changes: organize.Changes | None = None
        self._guesses: dict[str, str] = {}
        if not self.bulk and samples[0].name is None and samples[0].features is not None:
            # Start from whatever can be worked out, so only the rest needs typing
            proposal = filing.propose_for_samples(library, samples)[samples[0].rel]
            if proposal.name:
                n = proposal.name
                self._guesses = {"main": n.main, "sub": n.sub, "type": n.type, "bpm": str(n.bpm or ""), "key": n.key or "", "desc": n.desc}
            else:
                self._guesses = proposal.guesses

    def _initial(self, field: str) -> str:
        if field in self._guesses:
            return self._guesses[field]
        values = {_field_text(s, field) for s in self.samples}
        return values.pop() if len(values) == 1 else ""

    def compose(self) -> ComposeResult:
        mains = sorted({s.main for s in self.library.samples if s.name})
        subs = sorted({s.sub for s in self.library.samples if s.name})
        suggest = {"main": mains, "sub": subs, "type": [naming.LOOP, naming.ONESHOT],
                   "key": [f"{n}{m}" for n in naming.NOTES for m in ("min", "maj")]}
        labels = {"main": "Main", "sub": "Sub", "type": "Type", "bpm": "BPM", "key": "Key", "desc": "Description"}
        hints = {"bpm": "number, or NA", "key": "Am, F#min, Bbmaj, 8A, or NA", "type": "Loop or OneShot"}
        with Vertical(classes="dialog edit-dialog"):
            title = f"Edit {len(self.samples)} samples" if self.bulk else "Edit sample"
            yield Label(title, classes="dialog-title")
            if self.bulk:
                yield Static("Blank fields stay as they are on each file.", classes="dialog-detail")
            for field in self.FIELDS:
                with Horizontal(classes="edit-row"):
                    yield Label(labels[field], classes="edit-label")
                    choices = suggest.get(field)
                    yield Input(value=self._initial(field), id=f"edit-{field}",
                                placeholder=hints.get(field, "") if not self.bulk else "keep",
                                suggester=SuggestFromList(choices, case_sensitive=False) if choices else None)
            guess = self.samples[0].index.get("key_guess_low_confidence") if not self.bulk and self.samples[0].index else ""
            if guess and not self.samples[0].key:
                yield Static(f"Low-confidence key guess from the audio: {guess}", classes="dialog-detail")
            yield Static("", id="edit-preview")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Apply (ctrl+s)", id="apply", variant="primary")
                yield Button("Cancel (esc)", id="cancel")

    def on_mount(self) -> None:
        self._refresh_preview()

    def on_input_changed(self) -> None:
        self._refresh_preview()

    def on_input_submitted(self) -> None:
        self.action_apply()

    def _refresh_preview(self) -> None:
        preview = self.query_one("#edit-preview", Static)
        apply_button = self.query_one("#apply", Button)
        try:
            values = {f: self.query_one(f"#edit-{f}", Input).value for f in self.FIELDS}
            self._changes = organize.Changes.from_text(**values)
            target = organize.preview_name(self.library, self.samples[0], self._changes)
        except (ValueError, OSError, RuntimeError) as exc:
            self._changes = None
            apply_button.disabled = True
            preview.update(f"[#ff5f5f]{_escape(str(exc))}[/]")
            return
        apply_button.disabled = False
        unchanged = target == self.samples[0].rel
        note = " [dim](example)[/dim]" if self.bulk else ""
        preview.update("[dim]No change[/dim]" if unchanged and not self.bulk else f"→ {_escape(target)}{note}")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_apply(self) -> None:
        if self._changes is not None:
            self.dismiss(self._changes)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
        else:
            self.dismiss(None)


class GenreScreen(ModalScreen[tuple[list[str], list[str]] | None]):
    """Tick genres for one sample or many. For many, a genre only some of them have stays as it is unless you tick it."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("ctrl+s", "apply", "Apply"), STOP_BINDING]

    def __init__(self, library: Library, samples: list[Sample]):
        super().__init__()
        self.library, self.samples = library, samples
        counts: dict[str, int] = {}
        for sample in samples:
            for tag in sample.genres:
                counts[tag] = counts.get(tag, 0) + 1
        self.counts = counts
        self.on_all = {tag for tag, n in counts.items() if n == len(samples)}
        self.known = sorted({*genre_tags.VOCABULARY, *library.all_genres(), *counts}, key=str.lower)

    def _label(self, genre: str) -> str:
        count = self.counts.get(genre, 0)
        return f"{genre}  ({count} of {len(self.samples)})" if 0 < count < len(self.samples) else genre

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog genre-dialog"):
            many = len(self.samples) > 1
            yield Label(f"Genres for {len(self.samples)} samples" if many else "Genres", classes="dialog-title")
            yield Static("Space ticks a genre. Tab moves to the box below, where you can type a new one.", classes="dialog-detail")
            yield SelectionList[str](*[(self._label(g), g, g in self.on_all) for g in self.known], id="genre-list")
            yield Input(placeholder="Add a new genre, then press Enter", id="genre-new")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Apply (ctrl+s)", id="apply", variant="primary")
                yield Button("Cancel (esc)", id="cancel")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            names = genre_tags.parse_list(event.value)
        except ValueError:
            return
        options = self.query_one("#genre-list", SelectionList)
        existing = {genre_tags.key(g): g for g in self.known}
        for name in names:
            match = existing.get(genre_tags.key(name))
            if match is None:
                self.known.append(name)
                existing[genre_tags.key(name)] = name
                options.add_option((name, name, True))
            else:
                options.select(match)
        event.input.value = ""

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_apply(self) -> None:
        chosen = set(self.query_one("#genre-list", SelectionList).selected)
        self.dismiss((sorted(chosen - self.on_all), sorted(self.on_all - chosen)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
        else:
            self.dismiss(None)


def _field_text(sample: Sample, field: str) -> str:
    if sample.name is None:
        return naming.slugify(Path(sample.filename).stem) if field == "desc" else ""
    value = {"main": sample.main, "sub": sample.sub, "type": sample.type, "desc": sample.desc,
             "bpm": sample.bpm, "key": sample.key}[field]
    return str(value) if value else naming.NA


def _escape(text: str) -> str:
    return text.replace("[", "\\[")
