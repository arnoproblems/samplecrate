"""The bank builder: saving banks, keeping them right as files change, exporting them, and the screen itself."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import UNSORTED, snapshot, tone, write_wav
from samplecrate import banks, cli, organize
from samplecrate.library import PADS, Bank, Library
from samplecrate.organize import Changes
from samplecrate.tui.banks import KIT_KEYS, BankScreen
from samplecrate.tui.finder import FinderScreen
from samplecrate.tui.screens import ConfirmScreen, PathScreen, ReportScreen
from test_tui import SIZE, make_app, settle, wait_for

KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"
TUNED = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__F__0.30s__dark-tuned.wav"
BREAK = "Drums/Loops/Full/Drums__Full__Loop__120bpm__NA__2.00s__busy-break.wav"
JUNGLE = "Drums/Loops/Full/Drums__Full__Loop__174bpm__NA__1.38s__jungle.wav"
REESE = "Bass/Loops/Reese/Bass__Reese__Loop__087bpm__Amin__2.76s__wide.wav"
PAD = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav"
PIANO = "Instruments/Loops/Piano/Instruments__Piano__Loop__122bpm__Cmaj__1.97s__soft.wav"
VOCAL = "Vocals/Loops/Phrase/Vocals__Phrase__Loop__090bpm__Emin__2.67s__hello.wav"


def a_bank(library, name="Kit", **pads) -> Bank:
    bank = organize.new_bank(library, name)
    for pad, rel in pads.items():
        bank.put(int(pad[1:]) - 1, rel)
    library.write_banks()
    return bank


def reopen(root) -> Library:
    return Library(root).scan()


# ---------------------------------------------------------------------- saving

def test_banks_survive_a_restart(library, library_root):
    a_bank(library, "Dusty Kit", p1=KICK, p16=PAD)
    bank = reopen(library_root).bank_named("Dusty Kit")
    assert (bank.name, bank.pads[0], bank.pads[15], bank.filled) == ("Dusty Kit", KICK, PAD, 2)


def test_banks_file_numbers_pads_like_the_device(library, library_root):
    a_bank(library, p1=KICK, p16=PAD)
    state = json.loads((library_root / ".crate" / "banks.json").read_text())
    assert state == [{"name": "Kit", "pads": {"1": KICK, "16": PAD}}]


@pytest.mark.parametrize("junk", ["not json", '{"a": 1}', '[{"name": "x"}]', '[{"name": "x", "pads": {"z": "a"}}]'])
def test_a_damaged_banks_file_is_ignored(library_root, junk):
    (library_root / ".crate").mkdir()
    (library_root / ".crate" / "banks.json").write_text(junk)
    assert reopen(library_root).banks == []


def test_pad_numbers_out_of_range_are_ignored(library_root):
    (library_root / ".crate").mkdir()
    (library_root / ".crate" / "banks.json").write_text(json.dumps([{"name": "x", "pads": {"17": KICK, "0": PAD, "2": PAD}}]))
    assert reopen(library_root).banks[0].pads == [None, PAD] + [None] * (PADS - 2)


def test_no_banks_writes_no_file(library, library_root):
    library.write_banks()
    assert not (library_root / ".crate" / "banks.json").exists()


def test_bank_names_must_differ_as_folder_names(library):
    organize.new_bank(library, "Dusty Kit")
    for clash in ("dusty kit", "DUSTY-KIT", "  Dusty   Kit "):
        with pytest.raises(ValueError, match="already"):
            organize.new_bank(library, clash)
    with pytest.raises(ValueError, match="letter or number"):
        organize.new_bank(library, "!!!")
    assert organize.new_bank(library, "  Dusty   Kit 2 ").name == "Dusty Kit 2"


def test_next_bank_name_skips_ones_in_use(library):
    assert organize.next_bank_name(library) == "Bank 1"
    organize.new_bank(library, "Bank 1")
    assert organize.next_bank_name(library) == "Bank 2"


def test_read_only_library_refuses_bank_changes(library_root):
    writable = reopen(library_root)
    a_bank(writable, p1=KICK)
    library = Library(library_root, read_only=True).scan()
    before = snapshot(library_root)
    bank = library.banks[0]
    for change in (lambda: organize.new_bank(library, "New"), lambda: organize.delete_bank(library, bank),
                   lambda: organize.rename_bank(library, bank, "Other"),
                   lambda: organize.change_bank(library, bank, "x", lambda b: b.put(1, PAD))):
        with pytest.raises(PermissionError):
            change()
    assert snapshot(library_root) == before


# ---------------------------------------------------------------------- filling

def test_fill_uses_empty_pads_in_order(library):
    bank = a_bank(library, p1=KICK, p3=PAD)
    fill = organize.fill_bank(library, bank, [library.by_rel[TUNED], library.by_rel[BREAK]])
    assert fill.placed == [1, 3]
    assert bank.pads[:4] == [KICK, TUNED, PAD, BREAK]


def test_fill_leaves_samples_already_in_the_bank_alone(library):
    bank = a_bank(library, p5=KICK)
    fill = organize.fill_bank(library, bank, [library.by_rel[KICK], library.by_rel[PAD]])
    assert (fill.placed, fill.already) == ([0], 1)
    assert bank.pads[0] == PAD and bank.pads[4] == KICK


def test_fill_reports_what_did_not_fit(library):
    bank = a_bank(library, **{f"p{n}": f"other-{n}.wav" for n in range(1, 16)})
    fill = organize.fill_bank(library, bank, [library.by_rel[KICK], library.by_rel[PAD]])
    assert (fill.placed, fill.left_over) == ([15], 1)


def test_one_undo_reverses_a_whole_fill(library, library_root):
    bank = a_bank(library)
    organize.fill_bank(library, bank, [library.by_rel[KICK], library.by_rel[PAD]])
    organize.undo(library)
    assert library.banks[0].filled == 0
    assert reopen(library_root).banks[0].filled == 0


def test_a_change_that_changes_nothing_is_not_an_undo_step(library):
    bank = a_bank(library, p1=KICK)
    depth = organize.undo_depth(library)
    assert organize.change_bank(library, bank, "x", lambda b: b.put(0, KICK)) is False
    assert organize.undo_depth(library) == depth


# ---------------------------------------------------------------------- keeping up with files

def test_pads_follow_a_retag_and_its_undo(library, library_root):
    a_bank(library, p1=KICK, p2=PAD)
    moved = organize.retag(library, [library.by_rel[KICK]], Changes(desc="renamed"))[0].dst
    assert library.banks[0].pads[:2] == [moved, PAD]
    assert reopen(library_root).banks[0].pads[0] == moved
    organize.undo(library)
    assert library.banks[0].pads[0] == KICK
    assert reopen(library_root).banks[0].pads[0] == KICK


def test_every_pad_holding_a_file_follows_it(library):
    a_bank(library, "One", p1=KICK, p2=KICK)
    a_bank(library, "Two", p9=KICK)
    moved = organize.retag(library, [library.by_rel[KICK]], Changes(desc="renamed"))[0].dst
    assert library.pads_holding(moved) == [("One", 0), ("One", 1), ("Two", 8)]
    assert library.pads_holding(KICK) == []


def test_a_trashed_file_leaves_its_pad_waiting_until_undo(library):
    a_bank(library, p1=KICK)
    organize.trash(library, [library.by_rel[KICK]])
    assert library.banks[0].pads[0] == KICK
    assert banks.read_bank(library, library.banks[0]).notes[0] == [banks.MISSING]
    organize.undo(library)
    assert 0 not in banks.read_bank(library, library.banks[0]).notes


@pytest.mark.parametrize("change, check", [
    (lambda lib, bank: organize.change_bank(lib, bank, "x", lambda b: b.put(0, None)), lambda bank: bank.pads[0] == KICK),
    (lambda lib, bank: organize.change_bank(lib, bank, "x", lambda b: b.swap(0, 1)), lambda bank: bank.pads[:2] == [KICK, PAD]),
    (lambda lib, bank: organize.rename_bank(lib, bank, "Other"), lambda bank: bank.name == "Kit"),
    (lambda lib, bank: organize.delete_bank(lib, bank), lambda bank: bank.pads[:2] == [KICK, PAD]),
])
def test_bank_changes_undo(library, library_root, change, check):
    a_bank(library, "First")
    a_bank(library, p1=KICK, p2=PAD)
    a_bank(library, "Last")
    change(library, library.bank_named("Kit"))
    assert organize.undo(library) is not None
    for found in (library, reopen(library_root)):
        assert [b.name for b in found.banks] == ["First", "Kit", "Last"]    # a deleted bank goes back where it was
        assert check(found.bank_named("Kit"))


def test_undoing_a_new_bank_removes_it(library):
    organize.new_bank(library, "Kit")
    organize.undo(library)
    assert library.banks == []


def test_undo_runs_back_through_banks_and_files_in_order(library):
    bank = a_bank(library, p1=KICK)
    moved = organize.retag(library, [library.by_rel[KICK]], Changes(desc="renamed"))[0].dst
    organize.change_bank(library, library.banks[0], "x", lambda b: b.swap(0, 1))
    assert library.banks[0].pads[:2] == [None, moved]
    organize.undo(library)
    assert library.banks[0].pads[:2] == [moved, None]
    organize.undo(library)
    assert library.banks[0].pads[:2] == [KICK, None]
    assert bank.name == "Kit"


# ---------------------------------------------------------------------- whether it belongs together

def bank_of(library, *rels) -> Bank:
    return a_bank(library, **{f"p{i + 1}": rel for i, rel in enumerate(rels)})


def test_common_tempo_is_what_most_loops_can_play_at(library):
    reading = banks.read_bank(library, bank_of(library, BREAK, PAD, JUNGLE))
    assert reading.tempo == 120
    assert reading.notes[2] == ["≠ 120 bpm"]


def test_half_and_double_time_count_as_the_same_tempo(library):
    root = library.root
    write_wav(root / "Drums/Loops/Full/Drums__Full__Loop__060bpm__NA__4.00s__half.wav", 0.1)
    library.scan()
    reading = banks.read_bank(library, bank_of(library, BREAK, "Drums/Loops/Full/Drums__Full__Loop__060bpm__NA__4.00s__half.wav"))
    assert reading.notes == {}


def test_one_shots_have_no_say_in_the_tempo(library):
    assert banks.read_bank(library, bank_of(library, KICK, TUNED)).tempo is None


def test_common_key_and_compatible_keys(library):
    # A minor and C major are relatives. E minor is one step away on the wheel. None of them clash.
    reading = banks.read_bank(library, bank_of(library, PAD, REESE, PIANO, VOCAL))
    assert reading.key == "Amin"
    assert not any("≠ A" in note for notes in reading.notes.values() for note in notes)


def test_a_key_that_clashes_is_flagged(library):
    clash = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__D#maj__2.00s__clash.wav"
    write_wav(library.root / clash, 0.1)
    library.scan()
    reading = banks.read_bank(library, bank_of(library, PAD, REESE, clash))
    assert reading.key == "Amin" and reading.notes[2] == ["≠ Amin"]
    assert not any("≠ Amin" in note for pad in (0, 1) for note in reading.notes.get(pad, []))


def test_a_tuned_drum_does_not_decide_or_clash_with_the_key(library):
    reading = banks.read_bank(library, bank_of(library, TUNED))
    assert reading.key is None
    reading = banks.read_bank(library, a_bank(library, "Two", p1=PAD, p2=REESE, p3=TUNED))
    assert reading.key == "Amin" and 2 not in reading.notes


def test_very_short_samples_are_flagged(library):
    short = "Drums/One-Shots/Hat/Drums__Hat__OneShot__NA__NA__0.05s__tick.wav"
    write_wav(library.root / short, 0.05)
    library.scan()
    assert banks.read_bank(library, bank_of(library, short, KICK)).notes == {0: ["under 100 ms"]}


# ---------------------------------------------------------------------- exporting

def exported(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_export_numbers_files_by_pad_and_keeps_gaps(library, library_root, tmp_path):
    bank = a_bank(library, "Dusty Kit", p1=KICK, p5=PAD, p16=BREAK)
    before = {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")}
    result = organize.export_bank(library, bank, tmp_path / "IMPORT")
    folder = tmp_path / "IMPORT" / "dusty-kit"
    assert result.folder == folder and result.copied == 3
    assert exported(folder) == ["01__Drums__Kick__OneShot__0.20s__punchy.wav",
                                "05__Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav",
                                "16__Drums__Full__Loop__120bpm__2.00s__busy-break.wav"]
    assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == before


def test_export_order_sorts_the_same_as_pad_order(library, tmp_path):
    bank = a_bank(library, **{f"p{n}": BREAK for n in range(1, 17)})
    organize.export_bank(library, bank, tmp_path)
    assert [name[:2] for name in exported(tmp_path / "kit")] == [f"{n:02d}" for n in range(1, 17)]


def test_export_keeps_a_sample_on_two_pads(library, tmp_path):
    organize.export_bank(library, a_bank(library, p1=KICK, p2=KICK), tmp_path)
    assert [name[:2] for name in exported(tmp_path / "kit")] == ["01", "02"]


def test_export_converts_float_to_24_bit(library, library_root, tmp_path):
    rel = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__float.wav"
    sf.write(library_root / rel, tone(0.2) * 2.0, 22050, subtype="FLOAT")
    library.scan()
    result = organize.export_bank(library, a_bank(library, p3=rel), tmp_path)
    out = tmp_path / "kit" / "03__Drums__Kick__OneShot__0.20s__float.wav"
    assert (result.converted, result.turned_down) == (1, 1)
    assert sf.info(out).subtype == "PCM_24"
    assert np.abs(sf.read(out)[0]).max() <= 1.0


def test_exporting_again_unchanged_skips_everything(library, tmp_path):
    bank = a_bank(library, p1=KICK, p2=PAD)
    dest = tmp_path / "IMPORT"
    organize.export_bank(library, bank, dest)
    result = organize.export_bank(library, bank, dest)
    assert (result.copied, result.skipped, result.folder.name, result.replaces) == (0, 2, "kit", None)
    assert exported(dest) == ["kit"]


def test_filling_another_pad_adds_to_the_same_folder(library, tmp_path):
    bank = a_bank(library, p1=KICK)
    organize.export_bank(library, bank, tmp_path)
    bank.put(1, PAD)
    result = organize.export_bank(library, bank, tmp_path)
    assert (result.copied, result.skipped, result.folder.name) == (1, 1, "kit")
    assert len(exported(tmp_path / "kit")) == 2


def test_rearranging_leaves_the_older_layout_alone(library, tmp_path):
    bank = a_bank(library, p1=KICK, p2=PAD)
    organize.export_bank(library, bank, tmp_path)
    old = snapshot(tmp_path / "kit")
    bank.swap(0, 1)
    result = organize.export_bank(library, bank, tmp_path)
    assert result.folder.name == "kit-v2" and result.replaces == "kit"
    assert snapshot(tmp_path / "kit") == old
    assert [name[:2] for name in exported(tmp_path / "kit-v2")] == ["01", "02"]
    assert "older layout" in "\n".join(banks.describe_export(result))


def test_other_files_in_the_folder_do_not_count_as_a_layout(library, tmp_path):
    (tmp_path / "kit").mkdir()
    (tmp_path / "kit" / "notes.txt").write_text("mine")
    (tmp_path / "kit" / "._01__x.wav").write_bytes(b"tag")
    result = organize.export_bank(library, a_bank(library, p1=KICK), tmp_path)
    assert result.folder.name == "kit"


def test_export_reports_missing_and_short_pads(library, tmp_path):
    short = "Drums/One-Shots/Hat/Drums__Hat__OneShot__NA__NA__0.05s__tick.wav"
    write_wav(library.root / short, 0.05)
    library.scan()
    bank = a_bank(library, p1=KICK, p2=short, p3="gone.wav", p4="also-gone.wav")
    result = organize.export_bank(library, bank, tmp_path)
    assert (result.short, result.missing, result.copied) == ([1], [2, 3], 2)
    text = "\n".join(banks.describe_export(result))
    assert "Pad 2 is under 100 ms" in text
    assert "Pads 3 and 4 were left out" in text
    assert "IMPORT from SD-CARD" in text


def test_exporting_an_empty_bank_is_refused(library, tmp_path):
    dest = tmp_path / "IMPORT"
    with pytest.raises(ValueError, match="no samples"):
        organize.export_bank(library, a_bank(library), dest)
    with pytest.raises(ValueError, match="no samples"):
        organize.export_bank(library, a_bank(library, "Gone", p1="gone.wav"), dest)
    assert not dest.exists()


def test_exporting_into_the_library_is_refused(library, library_root):
    with pytest.raises(ValueError, match="inside the library"):
        organize.export_bank(library, a_bank(library, p1=KICK), library_root / "Drums")


def test_export_dry_run_writes_nothing(library, tmp_path):
    result = organize.export_bank(library, a_bank(library, p1=KICK), tmp_path / "IMPORT", dry_run=True)
    assert result.copied == 1 and not (tmp_path / "IMPORT").exists()
    assert banks.describe_export(result)[0].startswith("Would copy 1 sample ")


def test_read_only_library_can_still_export(library, library_root, tmp_path):
    a_bank(library, p1=KICK)
    locked = Library(library_root, read_only=True).scan()
    assert organize.export_bank(locked, locked.banks[0], tmp_path).copied == 1


def test_export_removes_the_companion_files_macos_writes_on_fat32(library, tmp_path, monkeypatch):
    def copy_like_fat32(src, dst):
        dst.write_bytes(src.read_bytes())
        dst.with_name("._" + dst.name).write_bytes(b"tag")
    monkeypatch.setattr(organize, "_copy", copy_like_fat32)
    organize.export_bank(library, a_bank(library, p1=KICK), tmp_path)
    assert exported(tmp_path / "kit") == ["01__Drums__Kick__OneShot__0.20s__punchy.wav"]


def test_a_file_not_named_by_the_convention_can_go_on_a_pad(library, tmp_path):
    organize.export_bank(library, a_bank(library, p1=UNSORTED), tmp_path)
    assert exported(tmp_path / "kit") == [f"01__{Path(UNSORTED).name}"]


# ---------------------------------------------------------------------- the command line

def run(capsys, *args):
    code = cli.main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_builds_and_exports_a_bank(library_root, tmp_path, capsys):
    lib = ("-l", str(library_root))
    assert "no banks yet" in run(capsys, "bank", *lib)[1]
    assert run(capsys, "bank", "new", "Dusty Kit", *lib)[0] == 0
    assert run(capsys, "bank", "put", "dusty kit", "1", KICK, *lib)[0] == 0
    code, out, _ = run(capsys, "bank", "put", "Dusty Kit", "5", "sub:pad", *lib)
    assert code == 0 and PAD in out
    assert "  2 of 16  Dusty Kit" in run(capsys, "bank", *lib)[1]
    out = run(capsys, "bank", "show", "Dusty Kit", *lib)[1]
    assert "2 of 16 pads" in out and f"  1  {KICK}" in out and f"  5  {PAD}" in out and "  2  -" in out
    code, out, _ = run(capsys, "bank", "export", "Dusty Kit", "--to", str(tmp_path), *lib)
    assert code == 0 and "numbered by pad" in out
    assert len(exported(tmp_path / "dusty-kit")) == 2


def test_cli_put_can_exclude_terms(library_root, capsys):
    lib = ("-l", str(library_root))
    run(capsys, "bank", "new", "Kit", *lib)
    code, out, _ = run(capsys, "bank", "put", "Kit", "1", "sub:kick", "-punchy", *lib)
    assert code == 0 and TUNED in out


def test_cli_clear_rename_delete_and_undo(library_root, capsys):
    lib = ("-l", str(library_root))
    run(capsys, "bank", "new", "Kit", *lib)
    run(capsys, "bank", "put", "Kit", "1", KICK, *lib)
    run(capsys, "bank", "clear", "Kit", "1", *lib)
    assert "0 of 16  Kit" in run(capsys, "bank", *lib)[1]
    run(capsys, "undo", *lib)
    assert "1 of 16  Kit" in run(capsys, "bank", *lib)[1]
    run(capsys, "bank", "rename", "Kit", "Other", *lib)
    assert "1 of 16  Other" in run(capsys, "bank", *lib)[1]
    run(capsys, "bank", "delete", "Other", *lib)
    assert "no banks yet" in run(capsys, "bank", *lib)[1]
    run(capsys, "undo", *lib)
    assert "1 of 16  Other" in run(capsys, "bank", *lib)[1]


@pytest.mark.parametrize("args, message", [
    (("show", "Nope"), "no bank called Nope"),
    (("put", "Kit", "17", KICK), "pad must be 1 to 16"),
    (("put", "Kit", "0", KICK), "pad must be 1 to 16"),
    (("put", "Kit", "1", "desc:nothing-like-this"), "nothing matches"),
    (("new", "Kit"), "already"),
])
def test_cli_explains_mistakes(library_root, capsys, args, message):
    lib = ("-l", str(library_root))
    run(capsys, "bank", "new", "Kit", *lib)
    code, _, err = run(capsys, "bank", *args, *lib)
    assert code == 2 and message in err


def test_cli_read_only_refuses_changes_but_exports(library_root, tmp_path, capsys):
    lib = ("-l", str(library_root))
    run(capsys, "bank", "new", "Kit", *lib)
    run(capsys, "bank", "put", "Kit", "1", KICK, *lib)
    assert run(capsys, "bank", "clear", "Kit", "1", "--read-only", *lib)[0] == 2
    assert run(capsys, "bank", "export", "Kit", "--to", str(tmp_path), "--read-only", *lib)[0] == 0


# ---------------------------------------------------------------------- the screen

def seeded(root, *rels, name="Kit") -> None:
    library = reopen(root)
    a_bank(library, name, **{f"p{i + 1}": rel for i, rel in enumerate(rels) if rel})


async def open_bank(pilot):
    await pilot.press("B")
    await settle(pilot)
    assert isinstance(pilot.app.screen, BankScreen)
    return pilot.app.screen


async def test_opening_makes_a_first_bank_and_plays_nothing(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        assert [b.name for b in app.library.banks] == ["Bank 1"]
        assert screen.cursor == 0 and not app.player.is_playing
        assert "Empty pad" in screen.viz.message


async def test_read_only_with_no_banks_does_not_open(library_root):
    app = make_app(library_root, read_only=True)
    state = snapshot(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("B")
        await settle(pilot)
        assert not isinstance(app.screen, BankScreen) and app.bank_screen is None
    assert snapshot(library_root) == state


async def test_p_puts_the_selected_sample_on_the_next_empty_pad(library_root):
    app = make_app(library_root, query="desc:punchy")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("p")
        await settle(pilot)
        assert app.library.banks[0].pads[0] == KICK
        assert isinstance(app.screen, type(app.screen)) and app.bank_screen is None     # stays in the library
        assert "Bank 1 1" in str(app.details.render())
    assert reopen(library_root).banks[0].pads[0] == KICK


async def test_p_puts_all_marked_samples_and_skips_ones_already_there(library_root):
    app = make_app(library_root, query="sub:kick")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("M", "p")
        await settle(pilot)
        assert app.library.banks[0].filled == 2
        await pilot.press("p")
        await settle(pilot)
        assert app.library.banks[0].filled == 2


async def test_full_bank_says_so(library_root):
    seeded(library_root, *[PAD] * 16)
    app = make_app(library_root, query="desc:punchy")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await pilot.press("p")
        await settle(pilot)
        assert KICK not in app.library.banks[0].pads
        assert any("didn't fit" in n.message for n in app._notifications)


async def test_arrows_walk_the_grid_without_wrapping(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("left", "up")
        assert screen.cursor == 0
        await pilot.press("right", "down")
        assert screen.cursor == 5
        await pilot.press("down", "down", "down", "right", "right", "right", "right")
        assert screen.cursor == 15
        assert screen.pads[15].has_class("-cursor") and not screen.pads[0].has_class("-cursor")


async def test_nothing_plays_when_selecting_with_play_on_select_off(library_root):
    seeded(library_root, KICK, PAD)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("right")
        await wait_for(pilot, lambda: app.current_clip is not None and app.current.rel == PAD)
        assert not app.player.is_playing


async def test_play_on_select_plays_each_pad_and_stops_on_an_empty_one(library_root):
    seeded(library_root, KICK, PAD)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("a", "right")
        await wait_for(pilot, lambda: app.current is not None and app.current.rel == PAD and app.player.is_playing)
        await pilot.press("right")
        await settle(pilot)
        assert app.current is None and not app.player.is_playing


async def test_space_enter_and_stop_work_on_a_pad(library_root):
    seeded(library_root, PAD)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.player.is_playing)
        await settle(pilot, 0.2)
        assert screen.pads[0].has_class("-playing")
        await pilot.press("s")
        assert not app.player.is_playing
        await pilot.press("space")
        await wait_for(pilot, lambda: app.player.is_playing)
        await pilot.press("ctrl+g")
        assert not app.player.is_playing


async def test_leaving_stops_the_sound_and_returns_to_the_table(library_root):
    seeded(library_root, PAD)
    app = make_app(library_root, query="desc:punchy")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.player.is_playing)
        await pilot.press("escape")
        await settle(pilot)
        assert app.bank_screen is None and screen is not app.screen
        assert not app.player.is_playing
        assert app.current.rel == KICK and app.focused is app.table


async def test_pick_from_the_library_fills_the_pad_and_comes_back(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root, query="sub:pad")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("right", "down", "p")     # pad 6
        await settle(pilot)
        assert app.bank_screen is None and app.pad_target == 5
        assert "Choosing for pad 6" in str(app.titlebar.render())
        await pilot.press("p")
        await settle(pilot)
        assert isinstance(app.screen, BankScreen) and app.screen.cursor == 5
        assert app.library.banks[0].pads[5] == PAD and app.pad_target is None


async def test_picking_for_a_filled_pad_replaces_it(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root, query="sub:pad")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("p")
        await settle(pilot)
        await pilot.press("p")
        await settle(pilot)
        assert app.library.banks[0].pads[0] == PAD
        await pilot.press("u")
        await settle(pilot)
        assert app.library.banks[0].pads[0] == KICK


@pytest.mark.parametrize("cancel", ["B", "escape"])
async def test_cancelling_a_pick_goes_back_and_changes_nothing(library_root, cancel):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("right", "p")
        await settle(pilot)
        await pilot.press(cancel)
        await settle(pilot)
        assert isinstance(app.screen, BankScreen) and app.screen.cursor == 1
        assert app.pad_target is None and app.library.banks[0].filled == 1


async def test_escape_clears_the_filter_first_while_picking(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root, query="sub:pad")
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("p")
        await settle(pilot)
        await pilot.press("escape")
        await settle(pilot)
        assert app.filter_input.value == "" and app.pad_target == 0 and app.bank_screen is None


async def test_pick_with_the_finder_fills_the_pad_and_comes_back(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await wait_for(pilot, lambda: not app.analysis_running, timeout=20)
        await open_bank(pilot)
        await pilot.press("right", "t")
        await settle(pilot, 0.3)
        finder = app.screen
        assert isinstance(finder, FinderScreen) and app.pad_target == 1
        assert "choosing for pad 2" in str(finder.heading.render())
        await pilot.press("enter")
        await settle(pilot, 0.3)
        chosen = finder.matches[0].sample
        await pilot.press("p")
        await settle(pilot, 0.3)
        assert isinstance(app.screen, BankScreen) and app.finder is None and app.pad_target is None
        assert app.library.banks[0].pads[1] == chosen.rel


async def test_leaving_the_finder_goes_back_to_the_bank_when_picking(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await wait_for(pilot, lambda: not app.analysis_running, timeout=20)
        await open_bank(pilot)
        await pilot.press("t")
        await settle(pilot, 0.3)
        await pilot.press("q")
        await settle(pilot, 0.3)
        assert isinstance(app.screen, BankScreen) and app.pad_target is None
        assert app.library.banks[0].pads[0] == KICK


async def test_finder_opened_on_its_own_adds_to_the_next_empty_pad_and_stays(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await wait_for(pilot, lambda: not app.analysis_running, timeout=20)
        await pilot.press("t")
        await settle(pilot, 0.3)
        await pilot.press("enter")
        await settle(pilot, 0.3)
        finder = app.screen
        chosen = finder.matches[0].sample
        await pilot.press("p")
        await settle(pilot)
        assert app.screen is finder
        assert app.library.banks[0].pads[0] == chosen.rel
        await pilot.press("q")
        await settle(pilot, 0.3)
        assert app.bank_screen is None and app.focused is app.table


async def test_grab_carries_a_pad_and_swaps_with_what_it_lands_on(library_root):
    seeded(library_root, KICK, PAD)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("g", "right")
        assert app.library.banks[0].pads[:2] == [PAD, KICK] and screen.cursor == 1
        await pilot.press("down")
        assert app.library.banks[0].pads[5] == KICK and app.library.banks[0].pads[1] is None
        await pilot.press("g", "left")
        assert screen.cursor == 4 and app.library.banks[0].pads[5] == KICK   # put down, so the arrows just move again
    assert reopen(library_root).banks[0].pads[5] == KICK


async def test_shift_arrows_carry_a_pad(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("shift+down")
        assert screen.cursor == 4 and app.library.banks[0].pads[4] == KICK
        await pilot.press("shift+left")     # off the edge
        assert screen.cursor == 4


async def test_grab_does_nothing_on_an_empty_pad(library_root):
    seeded(library_root, None, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("g", "right")
        assert not screen.carrying and screen.cursor == 1
        assert app.library.banks[0].pads[:2] == [None, KICK]


async def test_escape_puts_down_then_leaves(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("g", "escape")
        assert not screen.carrying and app.screen is screen
        await pilot.press("escape")
        await settle(pilot)
        assert app.bank_screen is None


async def test_clearing_a_pad_keeps_the_file_and_can_be_undone(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("backspace")
        await settle(pilot)
        assert app.library.banks[0].pads[0] is None and (library_root / KICK).exists()
        await pilot.press("u")
        await settle(pilot)
        assert app.library.banks[0].pads[0] == KICK and app.screen is screen
        assert screen.pads[0].has_class("-empty") is False


async def test_table_actions_do_not_reach_through_the_bank_screen(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    state = {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")}
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        sort = app.table.sort_column
        await pilot.press("d", "e", "m", "1", "A", "F", "/", "B")
        await settle(pilot)
        assert app.screen is screen and app.table.sort_column == sort and not app.table.marked
    assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == state


async def test_kit_keys_play_pads_instead_of_their_usual_jobs(library_root):
    seeded(library_root, KICK, *[None] * 8, PAD)     # pad 10 is the s key
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("k")
        await settle(pilot)
        assert "KIT" in str(screen.heading.render()) and "s" in str(screen.pads[9].render())
        assert KIT_KEYS[9] == "s"
        await pilot.press("s")
        await wait_for(pilot, lambda: app.player.is_playing)
        assert screen.cursor == 9 and app.current.rel == PAD
        await pilot.press("ctrl+g")
        assert not app.player.is_playing
        await pilot.press("x")      # pad 14, which is empty, rather than the export dialog
        await settle(pilot)
        assert app.screen is screen and screen.cursor == 13 and not app.player.is_playing
        await pilot.press("escape")
        await settle(pilot)
        assert app.screen is screen and not screen.kit_mode
        await pilot.press("s")
        assert screen.cursor == 13  # s is stop again


async def test_every_kit_key_reaches_its_pad(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("k")
        for pad, key in enumerate(KIT_KEYS):
            await pilot.press(key)
            assert screen.cursor == pad
        assert app.screen is screen


async def test_kit_plays_regardless_of_play_on_select(library_root):
    seeded(library_root, PAD)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        assert app.auto_audition is False
        await pilot.press("k", "1")
        await wait_for(pilot, lambda: app.player.is_playing)


async def test_clicking_selects_and_double_clicking_plays(library_root):
    seeded(library_root, KICK, PAD)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.click("#pad-2")
        await settle(pilot)
        assert screen.cursor == 1 and not app.player.is_playing
        await pilot.click("#pad-2", times=2)
        await wait_for(pilot, lambda: app.player.is_playing)


async def test_switching_creating_renaming_and_deleting_banks(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("n")
        await settle(pilot)
        assert isinstance(app.screen, PathScreen)
        await pilot.press("enter")
        await settle(pilot)
        assert screen.bank.name == "Bank 1" and len(app.library.banks) == 2
        await pilot.press("right_square_bracket")
        assert screen.bank.name == "Kit"
        await pilot.press("left_square_bracket")
        assert screen.bank.name == "Bank 1"
        await pilot.press("r")
        await settle(pilot)
        app.screen.query_one("#path-input").value = "Dusty"
        await pilot.press("enter")
        await settle(pilot)
        assert screen.bank.name == "Dusty" and "Dusty" in str(screen.heading.render())
        await pilot.press("D")
        await settle(pilot)
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("n")
        await settle(pilot)
        assert len(app.library.banks) == 2
        await pilot.press("D", "y")
        await settle(pilot)
        assert [b.name for b in app.library.banks] == ["Kit"] and screen.bank.name == "Kit"
        await pilot.press("u")
        await settle(pilot)
        assert [b.name for b in app.library.banks] == ["Kit", "Dusty"]


async def test_a_clashing_name_is_reported_not_fatal(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("n")
        await settle(pilot)
        app.screen.query_one("#path-input").value = "kit"
        await pilot.press("enter")
        await settle(pilot)
        assert app.screen is screen and len(app.library.banks) == 1
        assert any("already" in n.message for n in app._notifications)


async def test_deleting_the_last_bank_leaves_the_screen(library_root):
    seeded(library_root, KICK)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("D", "y")
        await settle(pilot)
        assert app.bank_screen is None and app.library.banks == []
        assert (library_root / KICK).exists()


async def test_undoing_the_only_bank_away_leaves_the_screen(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("u")
        await settle(pilot)
        assert app.bank_screen is None and app.library.banks == []


async def test_the_bank_you_used_last_is_remembered(library_root):
    seeded(library_root, KICK)
    seeded(library_root, PAD, name="Second")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("right_square_bracket", "escape")
        await settle(pilot)
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        assert screen.bank.name == "Second"


async def test_export_from_the_screen_reports_what_it_did(library_root, tmp_path):
    seeded(library_root, KICK, PAD, "gone.wav")
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await open_bank(pilot)
        await pilot.press("x")
        await settle(pilot)
        assert isinstance(app.screen, PathScreen)
        app.screen.query_one("#path-input").value = str(tmp_path / "IMPORT")
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
        text = str(app.screen.text)
        assert "Copied 2 samples" in text and "Pad 3 was left out" in text
        await pilot.press("escape")
        await settle(pilot)
        assert isinstance(app.screen, BankScreen)
    assert exported(tmp_path / "IMPORT" / "kit")[0].startswith("01__")
    assert json.loads((library_root / ".crate" / "prefs.json").read_text())["bank_export_dir"] == str(tmp_path / "IMPORT")


async def test_exporting_an_empty_bank_is_refused_on_screen(library_root):
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("x")
        await settle(pilot)
        assert app.screen is screen


async def test_read_only_can_listen_and_export_but_not_edit(library_root, tmp_path):
    seeded(library_root, PAD)
    app = make_app(library_root, read_only=True)
    state = snapshot(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.player.is_playing)
        await pilot.press("s", "backspace", "g", "shift+right", "p", "t", "n", "r", "D", "u")
        await settle(pilot)
        assert app.screen is screen and not screen.carrying and app.pad_target is None
        assert app.library.banks[0].pads[0] == PAD
        await pilot.press("x")
        await settle(pilot)
        app.screen.query_one("#path-input").value = str(tmp_path)
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ReportScreen))
    assert snapshot(library_root) == state


async def test_a_pad_whose_file_is_gone_says_so_and_stays_silent(library_root):
    seeded(library_root, KICK)
    (library_root / KICK).unlink()
    app = make_app(library_root)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        assert "gone" in screen.viz.message
        assert banks.MISSING in str(screen.pads[0].render())
        await pilot.press("enter", "space")
        await settle(pilot)
        assert not app.player.is_playing


@pytest.mark.parametrize("size", [(80, 24), (150, 45), (60, 20)])
async def test_the_grid_fits_small_and_large_terminals(library_root, size):
    seeded(library_root, *[PAD] * 16)
    app = make_app(library_root)
    async with app.run_test(size=size) as pilot:
        await settle(pilot)
        screen = await open_bank(pilot)
        await settle(pilot)
        rows = sorted({pad.region.y for pad in screen.pads})
        columns = sorted({pad.region.x for pad in screen.pads})
        assert len(rows) == 4 and len(columns) == 4
        assert all(pad.region.height >= 2 and pad.region.width >= 10 for pad in screen.pads)
        assert max(pad.region.bottom for pad in screen.pads) <= size[1]
        assert max(pad.region.right for pad in screen.pads) <= size[0]
