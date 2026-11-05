"""Drive the real app with simulated keys. All audio goes to a silent stream."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import UNSORTED, snapshot, write_wav
from samplecrate import duplicates, organize
from samplecrate import features as fx
from samplecrate.library import Library
from samplecrate.player import Player
from samplecrate.tui.app import CrateApp
from samplecrate.tui.finder import FinderScreen
from samplecrate.tui.screens import (ConfirmScreen, EditScreen, ExportScreen, GenreScreen, HelpScreen, PathScreen,
                                     ReportScreen)
from samplecrate.tui.visualizer import MODES

SIZE = (150, 45)
KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"
PAD = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav"


def make_app(root, query="", read_only=False):
    return CrateApp(Library(root, read_only=read_only).scan(), initial_query=query, player=Player(volume=0.0))


async def settle(pilot, seconds=0.12):
    await pilot.pause(seconds)
    await pilot.pause()


async def type_filter(pilot, text):
    await pilot.press("slash")
    await pilot.press(*("space" if c == " " else c for c in text))
    await settle(pilot)


def descs(app):
    return [s.desc for s in app.table.rows]


async def wait_for(pilot, condition, timeout=5.0):
    """Wait for something asynchronous, like a decode, rather than guessing how long it takes."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the app"
        await pilot.pause(0.01)


def finished_loading(app):
    return app.current_clip is not None or "Can't read" in app.viz.message


async def open_on(pilot, app, desc):
    """Move the cursor to a row by description, and wait for its audio to load."""
    index = descs(app).index(desc)
    app.table.cursor = index
    await pilot.pause()
    assert app.current.desc == desc
    await wait_for(pilot, lambda: finished_loading(app))


# ---------------------------------------------------------------------- browsing

async def test_starts_showing_the_library(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        assert len(app.table.rows) == 12
        assert app.focused is app.table
        assert app.current is not None and app.current_clip is not None
        assert app.viz.clip is app.current_clip


async def test_typing_a_filter_narrows_the_table_live(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await type_filter(pilot, "kick dark")
        assert descs(app) == ["dark-tuned"]
        assert app.focused is app.filter_input


async def test_keys_typed_in_the_filter_do_not_trigger_commands(library_root):
    app = make_app(library_root)
    state = snapshot(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await type_filter(pilot, "qedumxfvz")
        assert app.filter_input.value == "qedumxfvz"
        assert len(app.screen_stack) == 1 and app.is_running
        assert app.viz.mode == "wave"
        assert not app.table.marked
    assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == state


async def test_enter_and_escape_return_from_the_filter(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await type_filter(pilot, "pad")
        await pilot.press("enter")
        await settle(pilot)
        assert app.focused is app.table and app.filter_input.value == "pad"
        await pilot.press("slash")
        await pilot.press("escape")
        await settle(pilot)
        assert app.focused is app.table and app.filter_input.value == "pad"
        # A second escape, from the table, clears the filter
        await pilot.press("escape")
        await settle(pilot)
        assert app.filter_input.value == "" and len(app.table.rows) == 12


async def test_bad_filter_shows_an_error_and_keeps_working(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await type_filter(pilot, "bpm:abc")
        assert app.filter_input.has_class("-invalid")
        assert len(app.table.rows) == 12
        await pilot.press("ctrl+u")
        await settle(pilot)
        assert not app.filter_input.has_class("-invalid")


async def test_filter_with_no_matches(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await type_filter(pilot, "zzzzz")
        assert app.table.rows == [] and app.current is None
        await pilot.press("enter")
        for key in ("space", "enter", "e", "d", "f", "m", "v", "left", "right", "b", "h", "x", "j", "k", "G", "g"):
            await pilot.press(key)
        await settle(pilot)
        assert len(app.screen_stack) == 1 and app.is_running


async def test_initial_query(library_root):
    app = make_app(library_root, query="type:loop key:Am")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        assert sorted(descs(app)) == ["warm", "wide"]


async def test_cursor_navigation(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("j", "j")
        assert app.table.cursor == 2
        await pilot.press("k")
        assert app.table.cursor == 1
        await pilot.press("G")
        assert app.table.cursor == 11
        await pilot.press("j")
        assert app.table.cursor == 11
        await pilot.press("g")
        assert app.table.cursor == 0
        await pilot.press("k")
        assert app.table.cursor == 0
        await pilot.press("pagedown")
        assert app.table.cursor > 0


async def test_cursor_stays_visible_in_a_short_window(library_root):
    app = make_app(library_root)
    async with app.run_test(size=(150, 16)) as pilot:
        await settle(pilot)
        await pilot.press("G")
        await settle(pilot)
        table = app.table
        top = int(table.scroll_y)
        assert top <= table.cursor < top + table.size.height - 1
        await pilot.press("g")
        await settle(pilot)
        assert int(table.scroll_y) == 0


async def test_filter_keeps_cursor_on_the_same_sample(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await type_filter(pilot, "type:loop")
        assert app.table.current.desc == "warm"


async def test_cursor_returns_to_your_sample_when_a_filter_stops_hiding_it(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await type_filter(pilot, "kick")
        assert "warm" not in descs(app)
        await pilot.press("enter", "escape")
        await settle(pilot)
        assert app.filter_input.value == "" and app.table.current.desc == "warm"


async def test_frame_clock_never_looks_up_widgets(library_root, monkeypatch):
    """Regression: timers can fire during shutdown, after the widgets have left the DOM."""
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("space")
        await settle(pilot, 0.1)

        def fail(*args, **kwargs):
            raise AssertionError("query_one used on the frame path")

        monkeypatch.setattr(app, "query_one", fail)
        app._tick()
        app._update_transport()
        app._update_details()
        app._update_count()
        app._save_prefs()
        monkeypatch.undo()


async def test_sorting_by_keys_and_header_clicks(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("4")
        await settle(pilot)
        bpms = [s.bpm for s in app.table.rows if s.bpm]
        assert bpms == sorted(bpms)
        assert app.table.rows[-1].bpm is None
        await pilot.press("4")
        await settle(pilot)
        bpms = [s.bpm for s in app.table.rows if s.bpm]
        assert bpms == sorted(bpms, reverse=True)
        assert app.table.rows[-1].bpm is None
        await pilot.press("r")
        await settle(pilot)
        assert [s.bpm for s in app.table.rows if s.bpm] == sorted(bpms)
        await pilot.press("6")
        await settle(pilot)
        assert app.table.sort_column == "len"
        # Clicking a header sorts by that column
        table = app.table
        x = 3 + 12 + 12 + 6 + 2    # within the BPM column, with Main shown
        await pilot.click("#table", offset=(x, 0))
        await settle(pilot)
        assert table.sort_column == "bpm"


async def test_browse_tree_edits_the_query(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        tree = app.query_one("#browse")
        node = app._tree_nodes["Drums"]
        tree.select_node(node)
        await settle(pilot)
        assert app.filter_input.value == "in:Drums"
        assert all(s.main == "Drums" for s in app.table.rows)
        node = app._tree_nodes["Drums/One-Shots/Kick"]
        tree.select_node(node)
        await settle(pilot)
        assert app.filter_input.value == "in:Drums/One-Shots/Kick"
        assert sorted(descs(app)) == ["dark-tuned", "punchy"]
        tree.select_node(app._tree_nodes["__all__"])
        await settle(pilot)
        assert app.filter_input.value == ""


async def test_tree_keeps_other_filter_terms(library_root):
    app = make_app(library_root, query="dark")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        app.query_one("#browse").select_node(app._tree_nodes["Drums"])
        await settle(pilot)
        assert app.filter_input.value == "in:Drums dark"


async def test_tree_counts_follow_the_filter(library_root):
    app = make_app(library_root, query="key:Am")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        assert str(app._tree_nodes["Bass"].label).endswith("  1")
        assert str(app._tree_nodes["Drums"].label).endswith("  0")


async def test_invalid_files_get_a_tree_entry(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        app.query_one("#browse").select_node(app._tree_nodes["__invalid__"])
        await settle(pilot)
        assert descs(app) == ["my new kick"]


async def test_match_tempo_and_key_toggle(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "wide")
        await pilot.press("b")
        await settle(pilot)
        assert app.filter_input.value == "bpm:~87"
        assert "jungle" in descs(app)       # 174 is double 87
        await pilot.press("h")
        await settle(pilot)
        assert "key:~Amin" in app.filter_input.value
        await pilot.press("b")
        await settle(pilot)
        assert "bpm:" not in app.filter_input.value


async def test_match_on_a_sample_without_that_field(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "punchy")
        await pilot.press("b", "h")
        await settle(pilot)
        assert app.filter_input.value == ""


# ---------------------------------------------------------------------- playback and the visualizer

async def test_space_plays_pauses_and_resumes(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.3)
        assert app.player.is_playing
        await pilot.press("space")
        await settle(pilot)
        assert not app.player.is_playing
        held = app.player.position
        assert held > 0.05
        await pilot.press("space")
        await settle(pilot)
        assert app.player.is_playing and app.player.position >= held


async def test_enter_restarts_from_the_top(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.5)
        before = app.player.position
        await pilot.press("enter")
        await settle(pilot, 0.05)
        assert app.player.position < before


async def test_one_shot_finishes_on_its_own(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "punchy")
        await pilot.press("space")
        await settle(pilot, 0.6)
        assert not app.player.is_playing and app.player.position == 0


async def test_loops_repeat_only_when_loop_is_on(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "jungle")
        await pilot.press("space")
        await settle(pilot)
        assert app.player._voice.loop is True
        await pilot.press("l")
        assert app.player._voice.loop is False
        await pilot.press("l")
        assert app.player._voice.loop is True


async def test_one_shots_never_loop(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "whoosh")
        await pilot.press("space")
        await settle(pilot, 0.1)
        assert app.player._voice.loop is False


async def test_moving_with_play_on_select_off_stops_the_sound(library_root):
    """Nothing plays unless you ask, so a sound never keeps going under a different selected sample."""
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.2)
        assert app.player.is_playing
        await pilot.press("k")
        await settle(pilot, 0.2)
        assert not app.player.is_playing
        assert app.current_clip is not None     # but it still loads, so you can see it


async def test_moving_with_play_on_select_on_plays_each_sample(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("a")
        await pilot.press("k")
        await wait_for(pilot, lambda: app.player._voice is not None)
        first = app.player._voice.data
        assert first is app.current_clip.audio
        await pilot.press("k")
        await wait_for(pilot, lambda: app.player._voice is not None and app.player._voice.data is not first)
        assert app.player._voice.data is app.current_clip.audio


async def test_play_on_select_never_fires_at_startup(library_root):
    (library_root / ".crate").mkdir()
    (library_root / ".crate/prefs.json").write_text(json.dumps({"auto": True}))
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, 0.3)
        assert app.auto_audition and app.current_clip is not None
        assert not app.player.is_playing


async def test_typing_a_filter_never_plays_anything(library_root):
    """A filter moving the cursor isn't you choosing a sample, even with Play on select on."""
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("a")
        await type_filter(pilot, "pad")
        await settle(pilot, 0.2)
        assert app.current.desc == "warm" and not app.player.is_playing


async def test_stop_key_silences_and_rewinds(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.3)
        assert app.player.is_playing and app.player.position > 0.1
        await pilot.press("s")
        await settle(pilot, 0.05)
        assert not app.player.is_playing and app.player.position == 0


async def test_stop_also_cancels_a_sound_still_loading(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        app.current_clip = None
        app._play_when_loaded = True
        await pilot.press("s")
        assert app._play_when_loaded is False


async def test_ctrl_g_stops_audio_while_typing(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.1)
        await pilot.press("slash")
        await pilot.press("s", "y")     # plain s is just typing here; "sy" still matches the playing sample
        await settle(pilot)
        assert app.player.is_playing and app.filter_input.value == "sy"
        await pilot.press("ctrl+g")
        await settle(pilot, 0.05)
        assert not app.player.is_playing and app.filter_input.value == "sy"


async def test_ctrl_g_stops_audio_inside_a_dialog(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.1)
        await pilot.press("e")
        await settle(pilot)
        assert isinstance(app.screen, EditScreen) and app.player.is_playing
        await pilot.press("ctrl+g")
        await settle(pilot, 0.05)
        assert not app.player.is_playing and isinstance(app.screen, EditScreen)


async def test_transport_buttons_work_with_the_mouse(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        bar = app.transport
        await pilot.click(bar.play)
        await settle(pilot, 0.1)
        assert app.player.is_playing and "Pause" in str(bar.play.render())
        await pilot.click(bar.stop)
        await settle(pilot, 0.05)
        assert not app.player.is_playing and "Play" in str(bar.play.render())
        await pilot.click(bar.auto)
        await settle(pilot, 0.05)
        assert app.auto_audition and "☑" in str(bar.auto.render())
        await pilot.click(bar.auto)
        await settle(pilot, 0.05)
        assert not app.auto_audition and "☐" in str(bar.auto.render())
        await pilot.click(bar.view)
        await settle(pilot, 0.05)
        assert app.viz.mode == "spectrum"
        assert app.focused is app.table     # clicking a control doesn't take the keyboard away


async def test_moving_while_stopped_stays_silent_unless_auto_is_on(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("j")
        await settle(pilot, 0.2)
        assert not app.player.is_playing
        await pilot.press("a")
        await pilot.press("j")
        await settle(pilot, 0.2)
        assert app.player.is_playing


async def test_seek_keys_while_stopped_set_the_start_point(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("right", "right")
        assert app.player.position == pytest.approx(0.2, abs=0.01)
        await pilot.press("left")
        assert app.player.position == pytest.approx(0.1, abs=0.01)
        assert app.viz.position == pytest.approx(0.1, abs=0.01)


async def test_clicking_the_waveform_jumps_there(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        width = app.viz.size.width
        await pilot.click("#viz", offset=(width // 2, 2))
        await settle(pilot, 0.05)
        assert app.player.is_playing
        assert 0.9 <= app.player.position <= 1.3


async def test_narrow_window_still_shows_the_active_visualizer_mode(library_root):
    app = make_app(library_root)
    async with app.run_test(size=(70, 24)) as pilot:
        await settle(pilot)
        await pilot.press("v", "v", "v")
        await settle(pilot)
        assert app.viz.mode == "gram"
        assert "Spectrogram" in str(app.transport.view.render())


async def test_volume_keys(library_root):
    app = make_app(library_root)
    app.player.volume = 0.5
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("plus")
        assert app.player.volume == pytest.approx(0.55)
        await pilot.press("minus", "minus")
        assert app.player.volume == pytest.approx(0.45)


@pytest.mark.parametrize("size", [(200, 60), (120, 40), (100, 30), (80, 24), (60, 20), (40, 12)])
async def test_every_visualizer_mode_draws_at_every_size(library_root, size):
    app = make_app(library_root)
    async with app.run_test(size=size) as pilot:
        await settle(pilot)
        await wait_for(pilot, lambda: not app.analysis_running)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        for mode in MODES:
            await settle(pilot, 0.08)
            assert app.viz.mode == mode
            strips = [app.viz.render_line(y) for y in range(app.viz.size.height)]
            assert all(s.cell_length == app.viz.size.width for s in strips)
            assert any(s.text.strip() for s in strips), f"{mode} drew nothing"
            await pilot.press("v")
        await pilot.press("z", "i")
        await settle(pilot)


async def test_visualizer_animates_while_playing(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("v")      # spectrum
        await pilot.press("space")
        await settle(pilot, 0.2)
        frame_a = [s.text for s in app.viz._draw(app.viz.size.width, app.viz.size.height)]
        await settle(pilot, 0.6)
        frame_b = [s.text for s in app.viz._draw(app.viz.size.width, app.viz.size.height)]
        assert app.viz.position > 0
        assert frame_a != frame_b


async def test_waveform_playhead_advances(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot, 0.3)
        first = app.viz._play_x(app.viz.size.width)
        await settle(pilot, 0.5)
        assert app.viz._play_x(app.viz.size.width) > first
        assert "│" in app.viz.render_line(1).text


async def test_unreadable_file_shows_a_message_and_does_not_crash(library_root):
    (library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.10s__corrupt.wav").write_bytes(b"not audio")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "corrupt")
        assert app.current_clip is None and app.viz.clip is None
        assert "Can't read" in app.viz.message
        await pilot.press("space", "enter", "v", "left")
        await settle(pilot, 0.1)
        assert app.is_running
        # Moving away recovers
        await pilot.press("j")
        await wait_for(pilot, lambda: app.current_clip is not None)


# ---------------------------------------------------------------------- marking and favorites

async def test_marking(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("m")
        assert len(app.table.marked) == 1 and app.table.cursor == 1
        await pilot.press("k", "m")
        assert not app.table.marked
        await pilot.press("M")
        assert len(app.table.marked) == 12
        await pilot.press("n")
        await settle(pilot)
        assert not app.table.marked


async def test_is_marked_filter(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("m", "m")
        await type_filter(pilot, "is:marked")
        assert len(app.table.rows) == 2


async def test_favorites_persist(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("f")
        await settle(pilot)
        assert app.current.favorite
        assert str(app._tree_nodes["__fav__"].label).endswith("  1")
    fresh = Library(library_root).scan()
    assert fresh.by_rel[PAD].favorite


async def test_favoriting_marked_samples_applies_to_all(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("m", "m", "f")
        await settle(pilot)
        assert len(app.library.favorites) == 2
        await pilot.press("f")
        await settle(pilot)
        assert not app.library.favorites


# ---------------------------------------------------------------------- organizing

async def edit_fields(pilot, app, **fields):
    for field, value in fields.items():
        app.screen.query_one(f"#edit-{field}").value = value
    await settle(pilot, 0.05)


async def test_edit_renames_refiles_and_keeps_the_cursor(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("e")
        await settle(pilot)
        assert isinstance(app.screen, EditScreen)
        assert app.screen.query_one("#edit-bpm").value == "120"
        assert app.screen.query_one("#edit-key").value == "Amin"
        await edit_fields(pilot, app, sub="Lead", bpm="130", desc="Very Warm")
        assert "Synths/Loops/Lead/" in str(app.screen.query_one("#edit-preview").render())
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert not isinstance(app.screen, EditScreen)
    new = "Synths/Loops/Lead/Synths__Lead__Loop__130bpm__Amin__2.00s__very-warm.wav"
    assert (library_root / new).exists() and not (library_root / PAD).exists()
    assert app.current.rel == new


async def test_typing_in_the_edit_dialog_does_not_trigger_commands(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("e")
        await settle(pilot)
        app.screen.query_one("#edit-desc").value = ""
        app.screen.query_one("#edit-desc").focus()
        await pilot.press("q", "d", "u", "m", "v", "space")
        await settle(pilot)
        assert app.screen.query_one("#edit-desc").value == "qdumv "
        assert isinstance(app.screen, EditScreen) and app.is_running
        assert not app.player.is_playing and app.viz.mode == "wave"


async def test_edit_invalid_value_blocks_apply(library_root):
    app = make_app(library_root)
    state = {k: v for k, v in snapshot(library_root).items()}
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("e")
        await settle(pilot)
        await edit_fields(pilot, app, key="H#", bpm="130")
        assert app.screen.query_one("#apply").disabled
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert isinstance(app.screen, EditScreen)
        await pilot.press("escape")
        await settle(pilot)
        assert not isinstance(app.screen, EditScreen)
    now = snapshot(library_root)
    assert {k: v for k, v in now.items() if not k.startswith(".crate")} == state


async def test_cancelling_an_edit_changes_nothing(library_root):
    app = make_app(library_root)
    state = snapshot(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("e")
        await settle(pilot)
        await edit_fields(pilot, app, sub="Other")
        await pilot.press("escape")
        await settle(pilot)
    assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == state


async def test_bulk_edit_applies_to_marked_only_and_keeps_marks(library_root):
    app = make_app(library_root, query="sub:kick type:shot")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("M", "e")
        await settle(pilot)
        assert app.screen.bulk
        # Fields that differ start blank, and blank means keep
        assert app.screen.query_one("#edit-desc").value == ""
        assert app.screen.query_one("#edit-main").value == "Drums"
        await edit_fields(pilot, app, sub="Thump")
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert all(rel.startswith("Drums/One-Shots/Thump/") for rel in app.table.marked)
        assert len(app.table.marked) == 2
    root = library_root / "Drums/One-Shots/Thump"
    names = sorted(p.name for p in root.iterdir())
    assert names == ["Drums__Thump__OneShot__NA__F__0.30s__dark-tuned.wav",
                     "Drums__Thump__OneShot__NA__NA__0.20s__punchy.wav"]


async def test_edit_with_nothing_to_go_on_needs_the_categories(library_root):
    (library_root / "unsorted-stuff/my new kick.wav").rename(library_root / "unsorted-stuff/mystery 4.wav")
    app = make_app(library_root, query="is:invalid")
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await settle(pilot)
        await pilot.press("e")
        await settle(pilot)
        assert app.screen.query_one("#apply").disabled     # main and sub are still missing
        await edit_fields(pilot, app, main="Drums", sub="Kick", type="shot")
        assert not app.screen.query_one("#apply").disabled


async def test_edit_fixes_an_unsorted_file(library_root):
    app = make_app(library_root, query="is:invalid")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("e")
        await settle(pilot)
        await edit_fields(pilot, app, main="Drums", sub="Kick", type="shot")
        assert not app.screen.query_one("#apply").disabled
        await pilot.press("ctrl+s")
        await settle(pilot)
    filed = next(s for s in Library(library_root).scan().samples if s.index and s.index.get("original_name") == "my new kick.wav")
    assert filed.rel.startswith("Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.25s__") and not filed.problems
    assert not (library_root / "unsorted-stuff").exists()


async def test_undo_key(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("e")
        await settle(pilot)
        await edit_fields(pilot, app, desc="other")
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert not (library_root / PAD).exists()
        await pilot.press("u")
        await settle(pilot)
        assert (library_root / PAD).exists()
        assert "warm" in descs(app)
        # Nothing further to undo: still fine
        await pilot.press("u")
        await settle(pilot)
        assert app.is_running


async def test_trash_needs_confirmation_and_can_be_undone(library_root, tmp_path):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("d")
        await settle(pilot)
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("n")
        await settle(pilot)
        assert (library_root / PAD).exists()
        await pilot.press("d")
        await settle(pilot)
        await pilot.press("y")
        await settle(pilot)
        assert not (library_root / PAD).exists()
        assert (tmp_path / "trash" / Path(PAD).name).exists()
        assert "warm" not in descs(app)
        # The cursor slides to the next row rather than jumping away
        assert app.current is not None
        await pilot.press("u")
        await settle(pilot)
        assert (library_root / PAD).exists()


async def test_trashing_the_playing_sample_stops_it(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "warm")
        await pilot.press("space")
        await settle(pilot)
        await pilot.press("d")
        await settle(pilot)
        await pilot.press("y")
        await settle(pilot, 0.2)
        assert not app.player.is_playing
        assert app.current is not None and app.current.desc != "warm"


async def test_export_copies_marked_files(library_root, tmp_path):
    dest = tmp_path / "exported"
    app = make_app(library_root, query="sub:kick type:shot")
    state = snapshot(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("M", "x")
        await settle(pilot)
        assert isinstance(app.screen, ExportScreen)
        app.screen.query_one("#export-dir").value = str(dest)
        await pilot.press("enter")
        await settle(pilot, 0.3)
    assert len(list(dest.iterdir())) == 2
    assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == state


async def test_export_refusal_is_reported_not_fatal(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("x")
        await settle(pilot)
        app.screen.query_one("#export-dir").value = str(library_root / "inside")
        await pilot.press("enter")
        await settle(pilot, 0.3)
        assert app.is_running


async def test_read_only_mode_refuses_changes(library_root):
    state = snapshot(library_root)
    app = make_app(library_root, read_only=True)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("e", "d", "u", "f")
        await settle(pilot)
        assert len(app.screen_stack) == 1
        # Browsing still works
        await pilot.press("j", "4", "v", "space", "m")
        await type_filter(pilot, "kick")
    assert snapshot(library_root) == state


async def test_rescan_finds_new_files(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        write_wav(library_root / "Drums/One-Shots/Tom/Drums__Tom__OneShot__NA__NA__0.30s__new.wav", 0.3)
        await pilot.press("R")
        await settle(pilot)
        assert "new" in descs(app)
        assert "Drums/One-Shots/Tom" in app._tree_nodes


async def test_reveal_and_copy_path_use_system_tools(library_root, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)))
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("o", "y")
        await settle(pilot)
    assert calls[0][0][0][:2] == ["open", "-R"]
    assert calls[1][0][0] == ["pbcopy"]
    assert calls[1][1]["input"] == str(app.current.path).encode()


async def test_help_opens_and_q_only_closes_help(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("question_mark")
        await settle(pilot)
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("q")
        await settle(pilot)
        assert not isinstance(app.screen, HelpScreen) and app.is_running


async def test_preferences_survive_a_restart(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("v", "a", "l", "4")
        await settle(pilot)
    prefs = json.loads((library_root / ".crate/prefs.json").read_text())
    assert prefs["mode"] == "spectrum" and prefs["auto"] is True and prefs["loop"] is False
    assert prefs["sort"] == "bpm"
    app2 = make_app(library_root)
    async with app2.run_test(size=SIZE) as pilot:
        await settle(pilot)
        assert app2.viz.mode == "spectrum" and app2.auto_audition and not app2.loop_enabled
        assert app2.table.sort_column == "bpm"


async def test_quit_key_exits_cleanly(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("space")
        await pilot.press("q")
        await settle(pilot)
    assert not app.is_running


async def test_empty_library_opens_and_every_key_is_safe(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    app = make_app(root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        assert app.table.rows == [] and app.current is None
        for key in ("space", "enter", "e", "d", "f", "m", "M", "n", "x", "u", "v", "v", "z", "i", "b", "h", "left", "right", "j", "4", "s", "number_sign"):
            await pilot.press(key)
        await settle(pilot)
        assert app.is_running and len(app.screen_stack) == 1


# ---------------------------------------------------------------------- the sound maps

async def test_map_views_show_the_selected_sample(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await open_on(pilot, app, "punchy")
        await pilot.press("v", "v", "v", "v")
        await settle(pilot)
        assert app.viz.mode == "character"
        assert "Kick shots" in app.viz.plane.readout
        assert any("◉" in app.viz.render_line(y).text for y in range(app.viz.size.height))
        await pilot.press("v")
        await settle(pilot)
        assert app.viz.mode == "footprint"
        assert app.viz.plane.readout.startswith("Low range")
        text = "".join(app.viz.render_line(y).text for y in range(app.viz.size.height))
        assert "LOW" in text and "MID" in text and "HIGH" in text and "HITS" in text


async def test_footprint_dots_follow_the_filter(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("v", "v", "v", "v", "v")
        await settle(pilot)
        assert len(app.viz.plane.dots) == 12    # everything measured, even a file whose name still needs fixing
        await type_filter(pilot, "main:bass")
        await settle(pilot)
        assert len(app.viz.plane.dots) == 2


async def test_clicking_a_dot_selects_without_playing_when_toggle_is_off(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await open_on(pilot, app, "punchy")
        await pilot.press("v", "v", "v", "v", "v")
        await settle(pilot)
        viz = app.viz
        # The dot with the most room around it, so the click can only mean that one
        dots = [d for d in viz.plane.dots if not d.rel.endswith("punchy.wav")]
        target = max(dots, key=lambda d: min((d.x - o.x) ** 2 + (d.y - o.y) ** 2 for o in viz.plane.dots if o is not d))
        from samplecrate.tui import plane as plane_view
        area = plane_view.plane_area(viz.size.width, viz.size.height, True)
        x, y = plane_view.to_cell(area, (target.x, target.y))
        await pilot.click("#viz", offset=(area.left + x, area.top + y))
        await settle(pilot, 0.15)
        assert app.current.rel == target.rel and not app.player.is_playing


async def test_details_panel_shows_measurements_and_genres(library_root):
    app = make_app(library_root)
    async with app.run_test(size=(170, 45)) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        organize.set_genres(app.library, [app.library.by_rel[KICK]], ["House"])
        await open_on(pilot, app, "punchy")
        app._update_details()
        text = str(app.details.render())
        assert "Audible" in text and "Range" in text and "House" in text


# ---------------------------------------------------------------------- genres in the TUI

async def test_genre_dialog_tags_and_filters(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "punchy")
        await pilot.press("number_sign")
        await settle(pilot)
        assert isinstance(app.screen, GenreScreen)
        options = app.screen.query_one("#genre-list")
        options.select("House")
        options.select("Techno")
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert app.current.genres == ("House", "Techno")
        assert (library_root / KICK).exists()   # the file wasn't renamed
        assert "genre:House" in app._tree_nodes
        await type_filter(pilot, "genre:techno")
        assert descs(app) == ["punchy"]


async def test_genre_dialog_can_add_a_new_genre(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "punchy")
        await pilot.press("number_sign")
        await settle(pilot)
        box = app.screen.query_one("#genre-new")
        box.focus()
        await pilot.press(*"glitch", "space", *"hop", "enter")
        await settle(pilot)
        assert box.value == ""
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert app.current.genres == ("glitch hop",)


async def test_genre_dialog_cancel_changes_nothing(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("number_sign")
        await settle(pilot)
        app.screen.query_one("#genre-list").select("House")
        await pilot.press("escape")
        await settle(pilot)
        assert not app.library.genres


async def test_bulk_genres_leave_partly_tagged_genres_alone(library_root):
    app = make_app(library_root, query="sub:kick type:shot")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        kick = app.library.by_rel[KICK]
        organize.set_genres(app.library, [kick], ["Trap"])
        await pilot.press("M", "number_sign")
        await settle(pilot)
        options = app.screen.query_one("#genre-list")
        assert "Trap" not in options.selected    # only one of the two has it
        options.select("House")
        await pilot.press("ctrl+s")
        await settle(pilot)
        assert kick.genres == ("House", "Trap")
        other = next(s for s in app.library.samples if s.desc == "dark-tuned")
        assert other.genres == ("House",)


async def test_undo_reverses_genre_changes_in_the_tui(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "punchy")
        await pilot.press("number_sign")
        await settle(pilot)
        app.screen.query_one("#genre-list").select("House")
        await pilot.press("ctrl+s")
        await settle(pilot)
        await pilot.press("u")
        await settle(pilot)
        assert not app.library.genres and app.current.genres == ()


async def test_browse_tree_filters_by_genre_alongside_a_folder(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        organize.set_genres(app.library, [app.library.by_rel[KICK], app.library.by_rel[PAD]], ["Drum & Bass"])
        app._maybe_rebuild_tree()
        app.apply_filter()
        await settle(pilot)
        app.browse.select_node(app._tree_nodes["genre:Drum & Bass"])
        await settle(pilot)
        assert app.filter_input.value == "genre:drum-bass"
        assert sorted(descs(app)) == ["punchy", "warm"]
        app.browse.select_node(app._tree_nodes["Drums"])
        await settle(pilot)
        assert app.filter_input.value == "in:Drums genre:drum-bass" and descs(app) == ["punchy"]
        assert str(app._tree_nodes["genre:Drum & Bass"].label).endswith("  1")
        app.browse.select_node(app._tree_nodes["__all__"])
        await settle(pilot)
        assert app.filter_input.value == ""


async def test_sorting_by_genre_shows_the_column_even_when_narrow(library_root):
    from samplecrate.tui.table import layout
    assert "genres" not in [c.id for c in layout(80)]
    assert "genres" in [c.id for c in layout(80, True)]
    app = make_app(library_root)
    async with app.run_test(size=(120, 40)) as pilot:
        await settle(pilot)
        organize.set_genres(app.library, [app.library.by_rel[PAD]], ["House"])
        await pilot.press("7")
        await settle(pilot)
        assert app.table.sort_column == "genres"
        assert app.table.rows[0].desc == "warm"
        assert "House" in app.table.render_line(1).text


async def test_read_only_refuses_genre_dialog(library_root):
    app = make_app(library_root, read_only=True)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("number_sign")
        await settle(pilot)
        assert len(app.screen_stack) == 1


# ---------------------------------------------------------------------- the Finder

@pytest.fixture
def finder_root(tmp_path):
    root = tmp_path / "finder"
    for i in range(12):
        seconds = 0.08 + i * 0.09
        name = f"Drums__Kick__OneShot__NA__NA__{seconds:.2f}s__kick-{i}.wav"
        write_wav(root / "Drums/One-Shots/Kick" / name, seconds, hz=45 + i * 12)
    for i in range(6):
        name = f"Synths__Pad__Loop__120bpm__{'Amin' if i % 2 else 'Cmaj'}__2.00s__pad-{i}.wav"
        write_wav(root / "Synths/Loops/Pad" / name, 2.0, hz=300 + i * 100, decay=False)
    for i in range(2):
        write_wav(root / "Drums/One-Shots/Tom" / f"Drums__Tom__OneShot__NA__NA__0.30s__tom-{i}.wav", 0.3, hz=150)
    return root


async def open_finder(pilot, app):
    await wait_for(pilot, lambda: not app.analysis_running)
    await pilot.press("t")
    await settle(pilot)
    assert isinstance(app.screen, FinderScreen)
    return app.finder


async def place_dot(pilot, finder, x=0.5, y=0.5):
    from samplecrate.tui import plane as plane_view
    area = finder.picker._area()
    cx, cy = plane_view.to_cell(area, (x, y))
    await pilot.click("#finder-plane", offset=(area.left + cx, area.top + cy))
    await settle(pilot)


async def test_finder_starts_blank_on_the_current_category(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        assert finder.category == ("Drums", "OneShot", "Kick")
        assert finder.picker.target is None and finder.picker.markers == [] and finder.matches == []
        assert app.focused is finder.picker
        text = "".join(finder.picker.render_line(y).text for y in range(finder.picker.size.height))
        assert "CLICKY" in text and "BOOMY" in text
        assert "•" not in text      # no existing sounds on the plane


async def test_finder_category_list_skips_small_categories(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        labels = [str(n.label) for n in finder.categories.root.children]
        assert "Any category" in labels
        leaves = [leaf.data for n in finder.categories.root.children for leaf in n.children]
        assert ("Drums", "OneShot", "Kick") in leaves and ("Drums", "OneShot", "Tom") not in leaves


@pytest.mark.parametrize("key,count", [("3", 3), ("5", 5), ("0", 10)])
async def test_finder_lists_top_3_5_or_10(finder_root, key, count):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        assert app.focused is finder.results
        await pilot.press(key)
        await settle(pilot)
        assert len(finder.matches) == count == finder.results.option_count
        assert finder.matches == sorted(finder.matches, key=lambda m: m.distance)


async def test_finder_keyboard_aim_and_submit(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        start = finder.picker.cursor
        await pilot.press("right", "right", "up")
        assert finder.picker.cursor[0] > start[0] and finder.picker.cursor[1] > start[1]
        assert app.player.position == 0      # arrow keys aim here, they don't seek
        await pilot.press("enter")
        await settle(pilot)
        assert finder.picker.target == finder.picker.cursor and finder.matches
        assert len(finder.picker.markers) == len(finder.matches)


async def test_results_play_as_you_arrow_with_the_toggle_on(finder_root):
    """The scenario asked for: submit a dot, then arrow back and forth to compare by ear."""
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await pilot.press("a")
        await place_dot(pilot, finder)
        await wait_for(pilot, lambda: app.player._voice is not None)
        first = app.current
        assert app.player._voice.data is app.current_clip.audio
        assert first is finder.matches[0].sample
        await pilot.press("down")
        await wait_for(pilot, lambda: app.current is finder.matches[1].sample and app.player._voice is not None
                       and app.current_clip is not None and app.player._voice.data is app.current_clip.audio)
        await pilot.press("up")
        await wait_for(pilot, lambda: app.current is first and app.player._voice is not None
                       and app.current_clip is not None and app.player._voice.data is app.current_clip.audio)
        await pilot.press("s")
        await settle(pilot, 0.05)
        assert not app.player.is_playing


async def test_results_stay_silent_with_the_toggle_off_until_invoked(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        await settle(pilot, 0.2)
        assert not app.player.is_playing
        await pilot.press("down", "down", "up")
        await settle(pilot, 0.2)
        assert app.current is finder.matches[1].sample and not app.player.is_playing
        # The waveform still follows along
        assert finder.viz.clip is app.current_clip
        await pilot.press("space")
        await settle(pilot, 0.05)
        assert app.player.is_playing
        await pilot.press("s")
        await settle(pilot, 0.05)
        assert not app.player.is_playing
        await pilot.press("enter")
        await settle(pilot, 0.05)
        assert app.player.is_playing


async def test_finder_transport_stop_button(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        await pilot.press("space")
        await settle(pilot, 0.05)
        assert app.player.is_playing
        await pilot.click(finder.bar.stop)
        await settle(pilot, 0.05)
        assert not app.player.is_playing


async def test_finder_filter_narrows_results(finder_root):
    app = make_app(finder_root, query="")
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        app.table.cursor = next(i for i, s in enumerate(app.table.rows) if s.sub == "Pad")
        await settle(pilot)
        finder = await open_finder(pilot, app)
        assert finder.category == ("Synths", "Loop", "Pad")
        await place_dot(pilot, finder)
        await pilot.press("0")
        await settle(pilot)
        assert len(finder.matches) == 6
        finder.filter_box.value = "key:Am"
        await settle(pilot, 0.2)
        assert len(finder.matches) == 3 and all(m.sample.key == "Amin" for m in finder.matches)
        assert "Only 3" in str(finder.status.render())


async def test_finder_starting_filter_keeps_music_terms_only(finder_root):
    app = make_app(finder_root, query="in:Drums sub:kick bpm:~120 key:~Am")
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        app.filter_input.value = "in:Drums sub:kick bpm:~120 key:~Am"
        await pilot.press("t")
        await settle(pilot)
        assert app.finder.filter_box.value == "bpm:~120 key:~Am"


async def test_finder_any_category_uses_footprint_across_everything(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        finder.set_category(None)
        await settle(pilot)
        assert finder.picker.spec.kind == "footprint"
        await place_dot(pilot, finder, x=0.5, y=0.5)
        await pilot.press("0")
        await settle(pilot)
        assert finder.matches
        pool = finder._pool()
        assert any(s.sub == "Tom" for s in pool)     # small categories are searchable here


async def test_finder_empty_spot_is_labelled_honestly(tmp_path):
    root = tmp_path / "tiny"
    for i in range(5):
        write_wav(root / "Drums/One-Shots/Kick" / f"Drums__Kick__OneShot__NA__NA__0.20s__k-{i}.wav", 0.2, hz=60 + i)
    app = make_app(root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        finder.set_category(None)
        await settle(pilot)
        await place_dot(pilot, finder, x=0.98, y=0.98)
        assert finder.matches[0].closeness == "Far"
        assert "Nothing in here is close" in str(finder.status.render())


async def test_finder_marks_are_shared_with_the_library(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        await pilot.press("m")
        await settle(pilot)
        marked = finder.matches[0].sample.rel
        assert app.table.marked == {marked}
        await pilot.press("q")
        await settle(pilot)
        assert app.table.marked == {marked} and app.table.has_focus


async def test_escape_steps_back_and_closing_stops_audio(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        await pilot.press("space")
        await settle(pilot, 0.05)
        assert app.player.is_playing
        await pilot.press("escape")
        await settle(pilot, 0.05)
        assert app.focused is finder.picker and isinstance(app.screen, FinderScreen)
        await pilot.press("escape")
        await settle(pilot, 0.15)
        assert not isinstance(app.screen, FinderScreen) and app.finder is None
        assert not app.player.is_playing
        assert app.current is app.table.current and app.viz.clip is app.current_clip


async def test_show_in_library_jumps_to_the_sample(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder, x=0.9, y=0.9)
        await pilot.press("down")
        await settle(pilot)
        chosen = finder.matches[1].sample
        await pilot.press("L")
        await settle(pilot, 0.2)
        assert app.table.current is chosen and app.current is chosen


async def test_show_in_library_clears_a_filter_that_hides_it(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_on(pilot, app, "kick-1")
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        chosen = finder.matches[0].sample
        app.filter_input.value = "pad"
        await settle(pilot)
        await pilot.press("L")
        await settle(pilot, 0.2)
        assert app.filter_input.value == "" and app.table.current.rel == chosen.rel


async def test_library_only_keys_are_off_in_the_finder(finder_root):
    app = make_app(finder_root)
    state = {k: v for k, v in snapshot(finder_root).items() if not k.startswith(".crate")}
    async with app.run_test(size=SIZE) as pilot:
        finder = await open_finder(pilot, app)
        await place_dot(pilot, finder)
        sort = app.table.sort_column
        await pilot.press("e", "d", "u", "1", "2", "r", "M", "n", "b", "h", "t", "R")
        await settle(pilot)
        assert isinstance(app.screen, FinderScreen) and app.table.sort_column == sort
        assert not app.table.marked
    assert {k: v for k, v in snapshot(finder_root).items() if not k.startswith(".crate")} == state


async def test_finder_with_nothing_measured_yet(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    app = make_app(root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("t")
        await settle(pilot)
        assert app.finder.picker.spec is None
        await pilot.press("enter", "left", "3", "s", "space")
        await settle(pilot)
        assert app.is_running


async def test_finder_draws_at_small_sizes(finder_root):
    for size in ((200, 60), (100, 30), (80, 24), (60, 20), (40, 12)):
        app = make_app(finder_root)
        async with app.run_test(size=size) as pilot:
            finder = await open_finder(pilot, app)
            await pilot.press("enter")
            await settle(pilot)
            assert app.is_running
            for y in range(finder.picker.size.height):
                assert finder.picker.render_line(y).cell_length == finder.picker.size.width


# ---------------------------------------------------------------------- adding samples and duplicates

async def test_add_samples_dialog_files_new_samples(library_root, tmp_path):
    write_wav(tmp_path / "pack/Snare 3.wav", 0.5, hz=333)
    write_wav(tmp_path / "pack/Mystery 9.wav", 0.5, hz=444)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("A")
        await settle(pilot)
        app.screen.query_one("#path-input").value = str(tmp_path / "pack")
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        text = str(app.screen.query_one("#report-text").render())
        assert "filed automatically: 1" in text and "need you to file them: 1" in text and "Mystery 9.wav" in text
        await pilot.press("escape")
        await settle(pilot)
        assert "__new__" in app._tree_nodes
        app.browse.select_node(app._tree_nodes["__new__"])
        await settle(pilot)
        assert len(app.table.rows) == 2
        assert any(s.sub == "Snare" and not s.problems for s in app.table.rows)
        # One undo takes the whole upload back out
        await pilot.press("u")
        await settle(pilot, 0.2)
        assert not app.library.last_upload and not any(s.is_new for s in app.library.samples)


async def test_edit_dialog_starts_from_what_can_be_worked_out(library_root):
    write_wav(library_root / "Kick 5 128bpm.wav", 0.3, hz=70)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await type_filter(pilot, "is:invalid")
        await pilot.press("enter")
        await settle(pilot)
        app.table.cursor = descs(app).index("Kick 5 128bpm")
        await settle(pilot)
        await pilot.press("e")
        await settle(pilot)
        assert app.screen.query_one("#edit-main").value == "Drums"
        assert app.screen.query_one("#edit-sub").value == "Kick"
        assert app.screen.query_one("#edit-bpm").value == "128"


async def test_auto_file_key_handles_dragged_in_files(library_root):
    write_wav(library_root / "Snare 8.wav", 0.4, hz=250)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("F")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        assert "Filed 2 of 2" in str(app.screen.query_one("#report-text").render())   # the fixture's own unsorted file too
        await pilot.press("escape")
        await settle(pilot, 0.1)
        assert not (library_root / "Snare 8.wav").exists()
        assert any(s.sub == "Snare" and s.index and s.index["original_name"] == "Snare 8.wav" for s in app.library.samples)
        await pilot.press("u")
        await settle(pilot, 0.2)
        assert (library_root / "Snare 8.wav").exists()


async def test_add_samples_dialog_recognizes_what_you_already_have(library_root, tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    shutil.copy(library_root / KICK, pack / "kick copy.wav")
    write_wav(pack / "brand new.wav", 0.6, hz=777)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("A")
        await settle(pilot)
        assert isinstance(app.screen, PathScreen)
        app.screen.query_one("#path-input").value = str(pack)
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        text = str(app.screen.query_one("#report-text").render())
        assert "Added: 1" in text and "Already in your library: 1" in text and KICK in text
        await pilot.press("escape")
        await settle(pilot)
        assert (library_root / organize.INCOMING_FOLDER / "brand new.wav").exists()
        assert not any(s.index and s.index.get("original_name") == "kick copy.wav" for s in app.library.samples)
        assert "brand new" in descs(app)
        await pilot.press("u")
        await settle(pilot)
        assert not (library_root / organize.INCOMING_FOLDER).exists()


async def test_add_samples_bad_path_is_reported_not_fatal(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("A")
        await settle(pilot)
        app.screen.query_one("#path-input").value = "/no/such/place"
        await pilot.press("enter")
        await settle(pilot, 0.3)
        assert app.is_running and len(app.screen_stack) == 1


async def test_file_dropped_into_the_library_is_recognized_on_launch(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
    shutil.copy(library_root / KICK, library_root / "dropped here.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running and "dropped here.wav" in app.library.duplicates)
        await settle(pilot)
        assert "__dupes__" in app._tree_nodes
        app.browse.select_node(app._tree_nodes["__dupes__"])
        await settle(pilot)
        assert sorted(s.rel for s in app.table.rows) == sorted([KICK, "dropped here.wav"])
        assert app.library.duplicates["dropped here.wav"].kind == duplicates.SAME_AUDIO
        await open_on(pilot, app, "dropped here")
        app._update_details()
        assert "Same audio as" in str(app.details.render())


async def run_full_check(pilot, app):
    await pilot.press("D")
    await settle(pilot)
    await pilot.press("enter")      # blank means the whole library
    await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
    await pilot.press("escape")
    await settle(pilot)


@pytest.fixture
def scan_calls(monkeypatch):
    """Records every duplicate scan the app starts, and whether each was limited to particular files."""
    calls = []
    real = duplicates.scan_library

    def spy(library, **kwargs):
        calls.append(kwargs.get("only_rels"))
        return real(library, **kwargs)

    monkeypatch.setattr(duplicates, "scan_library", spy)
    return calls


async def test_first_launch_doesnt_check_for_duplicates(library_root, scan_calls):
    shutil.copy(library_root / KICK, library_root / "existing copy.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await settle(pilot, 0.3)
        assert scan_calls == [] and not app.library.duplicates
        assert "__dupes__" not in app._tree_nodes


async def test_later_launches_dont_check_for_duplicates(library_root, scan_calls):
    shutil.copy(library_root / KICK, library_root / "existing copy.wav")
    fx.Analyzer(Library(library_root).scan()).run()
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot, 0.3)
        assert all(s.features for s in app.library.samples)
        assert scan_calls == [] and not app.library.duplicates


async def test_editing_and_rescanning_dont_check_for_duplicates(library_root, scan_calls):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await open_on(pilot, app, "warm")
        await pilot.press("e")
        await settle(pilot)
        app.screen.query_one("#edit-desc").value = "other"
        await pilot.press("ctrl+s")
        await settle(pilot, 0.2)
        await pilot.press("R")
        await settle(pilot, 0.3)
        assert scan_calls == []


async def test_a_dragged_in_file_is_checked_alone(library_root, scan_calls):
    """Only the new file is looked at. Older duplicates stay unreported until you ask."""
    old_copy = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__old-copy.wav"
    shutil.copy(library_root / KICK, library_root / old_copy)
    fx.Analyzer(Library(library_root).scan()).run()
    shutil.copy(library_root / PAD, library_root / "dragged in.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: "dragged in.wav" in app.library.duplicates)
        assert scan_calls == [["dragged in.wav"]]
        assert old_copy not in app.library.duplicates and KICK not in app.library.duplicates
        assert PAD in app.library.duplicates


async def test_a_file_found_by_rescan_is_checked(library_root, scan_calls):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        shutil.copy(library_root / PAD, library_root / "dragged in.wav")
        await pilot.press("R")
        await wait_for(pilot, lambda: "dragged in.wav" in app.library.duplicates)
        assert scan_calls == [["dragged in.wav"]]


async def test_adding_through_the_dialog_needs_no_second_check(library_root, tmp_path, scan_calls):
    write_wav(tmp_path / "pack/new.wav", 0.6, hz=777)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("A")
        await settle(pilot)
        app.screen.query_one("#path-input").value = str(tmp_path / "pack")
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        await settle(pilot, 0.2)
        assert scan_calls == []


async def test_filter_says_when_no_check_has_run(library_root):
    shutil.copy(library_root / KICK, library_root / "existing copy.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await type_filter(pilot, "is:copy")
        assert app.table.rows == []
        assert "No check run yet" in str(app.count.render())
        await pilot.press("enter")
        await run_full_check(pilot, app)
        assert len(app.table.rows) == 1
        assert "No check run yet" not in str(app.count.render())


async def test_results_stay_right_as_copies_are_removed(library_root):
    shutil.copy(library_root / KICK, library_root / "second.wav")
    shutil.copy(library_root / KICK, library_root / "third.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await run_full_check(pilot, app)
        await type_filter(pilot, "is:copy")
        assert len(app.table.rows) == 2
        await pilot.press("enter", "d")
        await settle(pilot)
        await pilot.press("y")
        await settle(pilot, 0.2)
        assert len(app.table.rows) == 1
        await pilot.press("d")
        await settle(pilot)
        await pilot.press("y")
        await settle(pilot, 0.2)
        # The last survivor isn't a duplicate of anything now, so nothing is offered for removal
        assert app.table.rows == [] and not app.library.duplicates
        assert "__dupes__" not in app._tree_nodes
        assert (library_root / KICK).exists()


async def test_manual_duplicate_check_on_the_library(library_root):
    shutil.copy(library_root / KICK, library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__second.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("D")
        await settle(pilot)
        await pilot.press("enter")      # blank means the whole library
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        text = str(app.screen.query_one("#report-text").render())
        assert "1 sets of identical audio" in text and "keep" in text and "copy" in text
        await pilot.press("escape")
        await settle(pilot)
        await type_filter(pilot, "is:copy")
        assert len(app.table.rows) == 1


async def test_manual_duplicate_check_on_a_folder_changes_nothing(library_root, tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    shutil.copy(library_root / KICK, folder / "c.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        state = {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")}
        await pilot.press("D")
        await settle(pilot)
        app.screen.query_one("#path-input").value = str(folder)
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        assert "c.wav" in str(app.screen.query_one("#report-text").render())
        assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == state


async def test_trashing_every_copy_gets_a_warning(library_root):
    shutil.copy(library_root / KICK, library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__second.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await run_full_check(pilot, app)
        await type_filter(pilot, "is:dupe")
        await pilot.press("enter", "M", "d")
        await settle(pilot)
        assert isinstance(app.screen, ConfirmScreen) and "every copy" in app.screen.detail
        await pilot.press("n")


async def test_add_and_duplicate_keys_are_off_in_the_finder(finder_root):
    app = make_app(finder_root)
    async with app.run_test(size=SIZE) as pilot:
        await open_finder(pilot, app)
        await pilot.press("A", "D")
        await settle(pilot)
        assert isinstance(app.screen, FinderScreen)


async def test_read_only_refuses_add_but_allows_checking(library_root):
    app = make_app(library_root, read_only=True)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for(pilot, lambda: not app.analysis_running)
        await pilot.press("A")
        await settle(pilot)
        assert len(app.screen_stack) == 1
        await pilot.press("D")
        await settle(pilot)
        assert isinstance(app.screen, PathScreen)
