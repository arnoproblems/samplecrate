"""The bank builder: 16 pads laid out like the SP-404MK2's, to fill, audition as a kit and export."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.screen import Screen
from textual.widgets import Footer, Static

from samplecrate import banks, naming, organize
from samplecrate.library import PADS, Bank, Sample
from samplecrate.tui import theme
from samplecrate.tui.screens import ConfirmScreen, PathScreen
from samplecrate.tui.transport import TransportBar
from samplecrate.tui.visualizer import AUDIO_MODES, Visualizer

COLUMNS = 4
KIT_KEYS = "1234qwerasdfzxcv"   # the same shape as the pads
USE_FINDER = "finder"


class Pad(Static):
    def __init__(self, index: int):
        super().__init__(id=f"pad-{index + 1}", classes="pad")
        self.index = index

    def on_click(self, event: events.Click) -> None:
        self.screen.pad_clicked(self.index, event.chain >= 2)


class BankScreen(Screen[str | None]):
    """Dismisses with USE_FINDER when the Finder should open next, to choose a sample for a pad."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("up", "move(0, -1)", show=False),
        Binding("down", "move(0, 1)", show=False),
        Binding("left", "move(-1, 0)", show=False),
        Binding("right", "move(1, 0)", show=False),
        Binding("shift+up", "carry(0, -1)", show=False),
        Binding("shift+down", "carry(0, 1)", show=False),
        Binding("shift+left", "carry(-1, 0)", show=False),
        Binding("shift+right", "carry(1, 0)", show=False),
        Binding("enter", "play", show=False),
        Binding("p", "choose", "Pick"),
        Binding("t", "choose_with_finder", "Find"),
        Binding("g", "grab", "Move"),
        Binding("backspace,delete", "clear_pad", "Clear"),
        Binding("k", "kit", "Kit"),
        Binding("x", "export_bank", "Export"),
        Binding("left_square_bracket", "switch(-1)", show=False),
        Binding("right_square_bracket", "switch(1)", "Next bank"),
        Binding("n", "new_bank", "New bank"),
        Binding("r", "rename_bank", show=False),
        Binding("D", "delete_bank", show=False),
        Binding("u", "undo", "Undo"),
        Binding("q", "close", show=False),
    ]

    def __init__(self, cursor: int = 0):
        super().__init__()
        self.cursor = cursor
        self.kit_mode = False
        self.carrying = False

    def compose(self) -> ComposeResult:
        self.heading = Static("", id="bank-heading")
        self.pads = [Pad(index) for index in range(PADS)]
        self.selected_line = Static("", id="bank-selected")
        self.bar = TransportBar(id="bank-transport")
        self.viz = Visualizer(id="bank-viz")
        yield self.heading
        with Grid(id="bank-grid"):
            yield from self.pads
        yield self.selected_line
        yield self.bar
        yield self.viz
        yield Footer()

    def on_mount(self) -> None:
        mode = self.app.viz.mode
        self.viz.set_mode(mode if mode in AUDIO_MODES else "wave")
        self.set_interval(0.1, self._show_playing)
        self.show_pad(by_user=False)

    @property
    def bank(self) -> Bank:
        return self.app.bank

    def _sample_at(self, pad: int) -> Sample | None:
        rel = self.bank.pads[pad]
        return self.app.library.by_rel.get(rel) if rel else None

    # ------------------------------------------------------------------ drawing

    def show_pad(self, *, by_user: bool) -> None:
        """Select the sample under the cursor. Whether that plays it is up to Play on select, as everywhere else."""
        sample = self._sample_at(self.cursor)
        self.app.select_sample(sample, by_user=by_user)
        if sample is None:
            gone = self.bank.pads[self.cursor] is not None
            self.viz.set_clip(None, message="This pad's file is gone. u can bring it back, or Backspace clears the pad."
                              if gone else "Empty pad. p picks a sample from the library, t uses the Finder.")
        self.redraw()

    def redraw(self) -> None:
        bank = self.bank
        reading = banks.read_bank(self.app.library, bank)
        for pad in self.pads:
            pad.update(self._pad_text(pad.index, bank, reading))
            pad.set_class(bank.pads[pad.index] is None, "-empty")
            pad.set_class(pad.index == self.cursor, "-cursor")
        self.heading.update(self._heading_text(bank, reading))
        self.selected_line.update(self._selected_text(bank))

    def _heading_text(self, bank: Bank, reading: banks.BankReading) -> Text:
        library = self.app.library
        text = Text()
        text.append(bank.name, style="bold")
        text.append(f"  ·  bank {library.banks.index(bank) + 1} of {len(library.banks)}  ·  {bank.filled} of {PADS} pads",
                    style=theme.MUTED_COLOR)
        if reading.tempo:
            text.append(f"  ·  tempo {reading.tempo}")
        if reading.key:
            text.append(f"  ·  key {reading.key}", style=theme.key_color(reading.key))
        if self.kit_mode:
            text.append("  ·  KIT: 1234 qwer asdf zxcv play the pads, ctrl+g stops, Esc leaves", style=f"bold {theme.MARK_COLOR}")
        elif self.carrying:
            text.append(f"  ·  MOVING pad {self.cursor + 1}: the arrows carry it, g puts it down", style=f"bold {theme.MARK_COLOR}")
        return text

    def _pad_text(self, pad: int, bank: Bank, reading: banks.BankReading) -> Text:
        text = Text()
        text.append(f" {pad + 1} ", style="bold reverse" if pad == self.cursor else "bold")
        if self.kit_mode:
            text.append(f" {KIT_KEYS[pad]}", style=f"bold {theme.MARK_COLOR}")
        rel = bank.pads[pad]
        sample = self._sample_at(pad)
        if rel is None:
            return text
        if sample is None:
            text.append(f"\n{Path(rel).name}", style=theme.MUTED_COLOR)
        else:
            text.append(f" {sample.sub or sample.main}", style=f"bold {theme.category_color(sample.main)}")
            if sample.type == naming.LOOP:
                text.append(" ↻", style=theme.MUTED_COLOR)
            text.append(f"\n{sample.desc}\n")
            facts = [f"{sample.bpm}bpm" if sample.bpm else "", sample.key or "", f"{sample.length:.2f}s" if sample.length else ""]
            text.append("  ".join(fact for fact in facts if fact), style=theme.MUTED_COLOR)
        notes = reading.notes.get(pad)
        if notes:
            text.append("\n" + "  ".join(notes), style=theme.WARNING_COLOR)
        return text

    def _selected_text(self, bank: Bank) -> Text:
        rel = bank.pads[self.cursor]
        text = Text()
        text.append(f"Pad {self.cursor + 1}", style="bold")
        if rel is None:
            text.append("  ·  empty", style=theme.MUTED_COLOR)
            return text
        text.append(f"  ·  {Path(rel).name}", style=theme.MUTED_COLOR)
        sample = self._sample_at(self.cursor)
        if sample is not None and sample.genres:
            text.append(f"  ·  {', '.join(sample.genres)}")
        return text

    def _show_playing(self) -> None:
        playing = self.app.player is not None and self.app.player.is_playing
        for pad in self.pads:
            pad.set_class(playing and pad.index == self.cursor, "-playing")

    # ------------------------------------------------------------------ moving around

    def _neighbor(self, dx: int, dy: int) -> int | None:
        column, row = self.cursor % COLUMNS + dx, self.cursor // COLUMNS + dy
        if 0 <= column < COLUMNS and 0 <= row < PADS // COLUMNS:
            return row * COLUMNS + column
        return None

    def action_move(self, dx: int, dy: int) -> None:
        target = self._neighbor(dx, dy)
        if target is None:
            return
        if self.carrying:
            self._carry_to(target)
        else:
            self.cursor = target
            self.show_pad(by_user=True)

    def pad_clicked(self, pad: int, double: bool) -> None:
        if self.kit_mode:
            self._trigger(pad)
            return
        if pad != self.cursor:
            self.cursor = pad
            self.show_pad(by_user=True)
        if double:
            self.app.play_current_from_start()

    def action_play(self) -> None:
        if self.carrying:
            self.action_grab()
        else:
            self.app.play_current_from_start()

    def action_back(self) -> None:
        if self.kit_mode:
            self.action_kit()
        elif self.carrying:
            self.action_grab()
        else:
            self.action_close()

    def action_close(self) -> None:
        self.dismiss(None)

    # ------------------------------------------------------------------ filling and arranging

    def action_choose(self) -> None:
        self.app.choose_for_pad(self.cursor)

    def action_choose_with_finder(self) -> None:
        self.app.choose_for_pad(self.cursor, USE_FINDER)

    def action_grab(self) -> None:
        if self.carrying:
            self.carrying = False
        elif self.bank.pads[self.cursor] is not None and self.app._writable():
            self.carrying = True
        self.redraw()

    def action_carry(self, dx: int, dy: int) -> None:
        target = self._neighbor(dx, dy)
        if target is not None and self.bank.pads[self.cursor] is not None and self.app._writable():
            self._carry_to(target)

    def _carry_to(self, target: int) -> None:
        pad = self.cursor
        organize.change_bank(self.app.library, self.bank, f"move pad {pad + 1} to pad {target + 1} of {self.bank.name}",
                             lambda bank: bank.swap(pad, target))
        self.cursor = target
        self.redraw()

    def action_clear_pad(self) -> None:
        pad = self.cursor
        if self.bank.pads[pad] is None or not self.app._writable():
            return
        organize.change_bank(self.app.library, self.bank, f"clear pad {pad + 1} of {self.bank.name}", lambda bank: bank.put(pad, None))
        self.show_pad(by_user=False)
        self.notify(f"Cleared pad {pad + 1}. The sample is still in your library, and u undoes this.")

    def action_undo(self) -> None:
        self.app.undo_in_banks()

    # ------------------------------------------------------------------ the kit

    def action_kit(self) -> None:
        self.kit_mode = not self.kit_mode
        self.carrying = False
        if self.kit_mode:
            self.app.preload_clips([s for s in map(self._sample_at, range(PADS)) if s is not None])
        self.redraw()

    def on_key(self, event: events.Key) -> None:
        # Ahead of the bindings, so s and x play pads instead of stopping and exporting. ctrl+g still stops.
        if self.kit_mode and event.key in KIT_KEYS:
            event.stop()
            event.prevent_default()
            self._trigger(KIT_KEYS.index(event.key))

    def _trigger(self, pad: int) -> None:
        self.cursor = pad
        self.show_pad(by_user=False)
        self.app.play_current_from_start()

    # ------------------------------------------------------------------ managing banks

    def action_switch(self, step: int) -> None:
        library = self.app.library
        index = library.banks.index(self.bank)
        self.app.bank_name = library.banks[(index + step) % len(library.banks)].name
        self.carrying = False
        self.show_pad(by_user=False)

    def action_new_bank(self) -> None:
        if self.app._writable():
            self.app.push_screen(PathScreen("New bank", "Give it a name. r renames it later.", "Create",
                                            value=organize.next_bank_name(self.app.library)), self._create_bank)

    def _create_bank(self, name: str | None) -> None:
        if not name:
            return
        try:
            bank = organize.new_bank(self.app.library, name)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.app.bank_name = bank.name
        self.cursor = 0
        self.carrying = False
        self.show_pad(by_user=False)

    def action_rename_bank(self) -> None:
        if self.app._writable():
            self.app.push_screen(PathScreen("Rename bank", "The samples on its pads stay as they are.", "Rename",
                                            value=self.bank.name), self._rename_bank)

    def _rename_bank(self, name: str | None) -> None:
        if not name:
            return
        bank = self.bank
        try:
            organize.rename_bank(self.app.library, bank, name)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self.app.bank_name = bank.name
        self.redraw()

    def action_delete_bank(self) -> None:
        if self.app._writable():
            self.app.push_screen(ConfirmScreen(f"Delete {self.bank.name}?", "The samples stay in your library, and u undoes this."),
                                 lambda yes: self._delete_bank() if yes else None)

    def _delete_bank(self) -> None:
        library = self.app.library
        bank = self.bank
        index = library.banks.index(bank)
        organize.delete_bank(library, bank)
        if not library.banks:
            self.dismiss(None)
            return
        self.app.bank_name = library.banks[min(index, len(library.banks) - 1)].name
        self.carrying = False
        self.show_pad(by_user=False)

    def action_export_bank(self) -> None:
        bank = self.bank
        if not any(map(self._sample_at, range(PADS))):
            self.notify("This bank has no samples to export", severity="warning")
            return
        detail = (f"Copies {bank.filled} sample{'s' if bank.filled != 1 else ''} into a new folder called "
                  f"{naming.slugify(bank.name)} inside this one, each numbered by its pad. Use your SD card's IMPORT folder.")
        self.app.push_screen(PathScreen(f"Export {bank.name}", detail, "Copy", value=self.app.bank_export_dir),
                             lambda dest: self.app.run_bank_export(bank, dest) if dest else None)
