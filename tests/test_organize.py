import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import UNSORTED, snapshot, tone, write_wav
from samplecrate import organize
from samplecrate.library import Library
from samplecrate.organize import Changes, KEEP

KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"
KICK_EXPORTED = "Drums__Kick__OneShot__0.20s__punchy.wav"
PAD = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav"


def index_rows(root):
    with open(root / "_index.csv", newline="") as fh:
        return {row["path"]: row for row in csv.DictReader(fh)}


def test_from_text_blank_means_keep_and_na_clears():
    c = Changes.from_text(main="", bpm="NA", key="Bbm", sub=None)
    assert c.main is KEEP and c.sub is KEEP
    assert c.bpm is None
    assert c.key == "A#min"


def test_from_text_validates():
    for kwargs in ({"bpm": "abc"}, {"key": "H"}, {"type": "banana"}, {"desc": "!!!"}, {"main": "123"}):
        with pytest.raises(ValueError):
            Changes.from_text(**kwargs)


def test_retag_renames_and_refiles(library, library_root):
    sample = library.by_rel[KICK]
    moves = organize.retag(library, [sample], Changes(sub="Snare", key="F", desc="crack"))
    new = "Drums/One-Shots/Snare/Drums__Snare__OneShot__NA__F__0.20s__crack.wav"
    assert [m.dst for m in moves] == [new]
    assert (library_root / new).exists()
    assert not (library_root / KICK).exists()
    assert new in library.by_rel and KICK not in library.by_rel


def test_changing_type_changes_folder(library, library_root):
    moves = organize.retag(library, [library.by_rel[KICK]], Changes(type="Loop", bpm=120))
    assert moves[0].dst == "Drums/Loops/Kick/Drums__Kick__Loop__120bpm__NA__0.20s__punchy.wav"


def test_no_change_is_a_no_op(library, library_root):
    before = snapshot(library_root)
    assert organize.retag(library, [library.by_rel[KICK]], Changes(sub="Kick")) == []
    assert snapshot(library_root) == before
    assert not organize.can_undo(library)


def test_emptied_folders_are_removed(library, library_root):
    rel = "Drums/One-Shots/Hat-Open/Drums__Hat-Open__OneShot__NA__NA__0.40s__bright.wav"
    (library_root / Path(rel).parent / ".DS_Store").write_bytes(b"x")
    organize.retag(library, [library.by_rel[rel]], Changes(sub="Hat"))
    assert not (library_root / "Drums/One-Shots/Hat-Open").exists()
    assert (library_root / "Drums/One-Shots").exists()


def test_collision_gets_version_suffix_and_never_overwrites(library, library_root):
    a = library.by_rel["Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"]
    write_wav(library_root / "Drums/One-Shots/Snare/Drums__Snare__OneShot__NA__NA__0.20s__punchy.wav", 0.2, hz=300)
    library.scan()
    existing = (library_root / "Drums/One-Shots/Snare/Drums__Snare__OneShot__NA__NA__0.20s__punchy.wav").read_bytes()
    moves = organize.retag(library, [library.by_rel[a.rel]], Changes(sub="Snare"))
    assert moves[0].dst.endswith("punchy-v2.wav")
    assert (library_root / "Drums/One-Shots/Snare/Drums__Snare__OneShot__NA__NA__0.20s__punchy.wav").read_bytes() == existing


def test_bulk_retag_to_same_name_all_survive(library, library_root):
    kicks = [library.by_rel[KICK], library.by_rel["Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__F__0.30s__dark-tuned.wav"]]
    files_before = len(library.samples)
    organize.retag(library, kicks, Changes(desc="same"))
    assert len(library.samples) == files_before


def test_swapping_names_in_one_batch_never_loses_a_file(library, library_root):
    a = library.by_rel[KICK]
    other = write_wav(library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__other.wav", 0.2, hz=500)
    library.scan()
    b = library.by_rel["Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__other.wav"]
    content_a, content_b = a.path.read_bytes(), other.read_bytes()
    organize.retag(library, [library.by_rel[a.rel], b], Changes(desc="other"))
    contents = {s.path.read_bytes() for s in library.samples if s.sub == "Kick" and s.length == 0.2}
    assert content_a in contents and content_b in contents


def test_failed_batch_rolls_back(library, library_root, monkeypatch):
    before = snapshot(library_root)
    real = os.rename
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(organize.os, "rename", flaky)
    samples = [library.by_rel[KICK], library.by_rel[PAD]]
    with pytest.raises(OSError):
        organize.retag(library, samples, Changes(desc="new"))
    monkeypatch.setattr(organize.os, "rename", real)
    assert snapshot(library_root) == before


def test_retag_updates_index_and_marks_manual(library, library_root):
    organize.retag(library, [library.by_rel[PAD]], Changes(bpm=130))
    rows = index_rows(library_root)
    new = "Synths/Loops/Pad/Synths__Pad__Loop__130bpm__Amin__2.00s__warm.wav"
    assert PAD not in rows
    row = rows[new]
    assert row["bpm"] == "130" and row["bpm_source"] == "manual"
    assert row["key_source"] == "name"              # untouched fields keep their source
    assert row["original_name"] == "ORIG_warm.wav"   # provenance survives


def test_library_without_index_works_and_stays_without_one(tmp_path):
    root = tmp_path / "plain"
    write_wav(root / KICK, 0.2)
    lib = Library(root).scan()
    organize.retag(lib, [lib.samples[0]], Changes(sub="Snare"))
    assert not (root / "_index.csv").exists()


def test_undo_restores_everything(library, library_root):
    before = snapshot(library_root)
    library.toggle_favorite(library.by_rel[PAD])
    favorites_before = (library_root / ".crate/favorites.json").read_text()
    organize.retag(library, [library.by_rel[PAD], library.by_rel[KICK]], Changes(desc="renamed", sub="Other"))
    assert organize.undo(library).startswith("retag")
    after = snapshot(library_root)
    for rel, data in before.items():
        assert after[rel] == data or rel == "_index.csv"
    assert index_rows(library_root).keys() == {r for r in index_rows(library_root)}
    assert PAD in index_rows(library_root) and KICK in index_rows(library_root)
    assert (library_root / ".crate/favorites.json").read_text() == favorites_before
    assert library.by_rel[PAD].favorite


def test_undo_with_empty_history_is_harmless(library):
    assert organize.undo(library) is None


def test_undo_survives_a_restart(library, library_root):
    organize.retag(library, [library.by_rel[KICK]], Changes(desc="later"))
    fresh = Library(library_root).scan()
    assert organize.can_undo(fresh)
    organize.undo(fresh)
    assert (library_root / KICK).exists()


def test_multi_level_undo(library, library_root):
    organize.retag(library, [library.by_rel[KICK]], Changes(desc="one"))
    one = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__one.wav"
    organize.retag(library, [library.by_rel[one]], Changes(desc="two"))
    organize.undo(library)
    assert (library_root / one).exists()
    organize.undo(library)
    assert (library_root / KICK).exists()


def test_undo_refuses_if_a_file_vanished_and_changes_nothing(library, library_root):
    organize.retag(library, [library.by_rel[KICK], library.by_rel[PAD]], Changes(desc="x"))
    moved = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__x.wav"
    (library_root / moved).unlink()
    other = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__x.wav"
    with pytest.raises(FileNotFoundError):
        organize.undo(library)
    assert (library_root / other).exists()
    assert not (library_root / KICK).exists()


def test_trash_and_undo(library, library_root, tmp_path):
    data = (library_root / KICK).read_bytes()
    library.toggle_favorite(library.by_rel[KICK])
    assert organize.trash(library, [library.by_rel[KICK]]) == 1
    assert not (library_root / KICK).exists()
    assert (tmp_path / "trash" / Path(KICK).name).read_bytes() == data
    assert KICK not in index_rows(library_root)
    assert KICK not in library.favorites
    organize.undo(library)
    assert (library_root / KICK).read_bytes() == data
    assert KICK in index_rows(library_root)
    assert library.by_rel[KICK].favorite


def test_trash_never_overwrites_files_already_in_the_trash(library, tmp_path):
    trash_dir = tmp_path / "trash"
    trash_dir.mkdir()
    (trash_dir / Path(KICK).name).write_bytes(b"older")
    organize.trash(library, [library.by_rel[KICK]])
    assert (trash_dir / Path(KICK).name).read_bytes() == b"older"
    assert len(list(trash_dir.iterdir())) == 2


def test_partial_trash_failure_is_still_undoable(library, library_root, monkeypatch):
    real = organize.shutil.move
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("nope")
        return real(src, dst)

    monkeypatch.setattr(organize.shutil, "move", flaky)
    with pytest.raises(OSError):
        organize.trash(library, [library.by_rel[KICK], library.by_rel[PAD]])
    monkeypatch.setattr(organize.shutil, "move", real)
    assert not (library_root / KICK).exists()
    assert (library_root / PAD).exists()
    organize.undo(library)
    assert (library_root / KICK).exists()


def test_export_copies_and_never_overwrites(library, library_root, tmp_path):
    dest = tmp_path / "out"
    samples = [library.by_rel[KICK], library.by_rel[PAD]]
    before = snapshot(library_root)
    assert organize.export(library, samples, dest).copied == 2
    assert snapshot(library_root) == before     # copy, not move
    (dest / KICK_EXPORTED).write_bytes(b"mine")
    result = organize.export(library, samples, dest)
    assert (result.copied, result.skipped) == (0, 2)
    assert (dest / KICK_EXPORTED).read_bytes() == b"mine"


def test_export_names_leave_out_na_fields(library, tmp_path):
    dest = tmp_path / "out"
    organize.export(library, library.samples, dest)
    names = {p.name for p in dest.iterdir()}
    assert len(names) == len(library.samples)
    assert not any("__NA__" in name for name in names)
    assert {KICK_EXPORTED,                                          # neither
            "Drums__Kick__OneShot__F__0.30s__dark-tuned.wav",       # key only
            "Drums__Full__Loop__120bpm__2.00s__busy-break.wav",     # tempo only
            Path(PAD).name,                                         # both, so nothing to remove
            Path(UNSORTED).name} <= names                           # not named by the convention, so left alone


def test_export_leaves_integer_files_byte_identical(library_root, tmp_path):
    source = write_wav(library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.50s__deep.wav", 0.5, "PCM_24")
    library = Library(library_root).scan()
    dest = tmp_path / "out"
    result = organize.export(library, [library.by_rel[KICK], library.by_rel[str(source.relative_to(library_root))]], dest)
    assert (result.copied, result.converted) == (2, 0)
    assert (dest / "Drums__Kick__OneShot__0.50s__deep.wav").read_bytes() == source.read_bytes()
    assert (dest / KICK_EXPORTED).read_bytes() == (library_root / KICK).read_bytes()


def float_sample(library_root, data):
    rel = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__float.wav"
    (library_root / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(library_root / rel, data, 22050, subtype="FLOAT")
    library = Library(library_root).scan()
    return library, library.by_rel[rel]


FLOAT_EXPORTED = "Drums__Kick__OneShot__0.20s__float.wav"


def test_export_converts_float_to_24_bit(library_root, tmp_path):
    data = tone(0.2)
    library, sample = float_sample(library_root, data)
    before = sample.path.read_bytes()
    dest = tmp_path / "out"
    result = organize.export(library, [sample], dest)
    assert (result.copied, result.converted, result.turned_down) == (1, 1, 0)
    assert result.conversion_note == "converted 1 from float to 24-bit"
    assert sf.info(dest / FLOAT_EXPORTED).subtype == "PCM_24"
    assert np.allclose(sf.read(dest / FLOAT_EXPORTED, always_2d=True)[0], data, atol=1e-6)
    assert sample.path.read_bytes() == before
    assert [p.name for p in dest.iterdir()] == [FLOAT_EXPORTED]


def test_export_turns_down_float_past_full_scale_instead_of_clipping(library_root, tmp_path):
    data = tone(0.2)
    data = data / np.abs(data).max() * 2.5
    library, sample = float_sample(library_root, data)
    dest = tmp_path / "out"
    result = organize.export(library, [sample], dest)
    assert result.turned_down == 1
    assert "turning 1 down" in result.conversion_note
    written = sf.read(dest / FLOAT_EXPORTED, always_2d=True)[0]
    assert np.abs(written).max() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(written, data / 2.5, atol=1e-6)      # the same shape, just quieter


def test_export_silences_broken_float_values(library_root, tmp_path):
    data = tone(0.2)
    data[10] = np.nan
    data[20] = np.inf
    library, sample = float_sample(library_root, data)
    dest = tmp_path / "out"
    organize.export(library, [sample], dest)
    written = sf.read(dest / FLOAT_EXPORTED, always_2d=True)[0]
    assert np.isfinite(written).all()
    assert (written[10] == 0).all() and (written[20] == 0).all()
    assert np.abs(written).max() < 1.0


def test_failed_conversion_leaves_nothing_behind(library_root, tmp_path, monkeypatch):
    library, sample = float_sample(library_root, tone(0.2))
    dest = tmp_path / "out"

    def card_is_full(path, *args, **kwargs):
        Path(path).write_bytes(b"half a file")
        raise OSError("no space left on device")
    monkeypatch.setattr(organize.sf, "write", card_is_full)
    with pytest.raises(OSError):
        organize.export(library, [sample], dest)
    assert list(dest.iterdir()) == []


def test_export_removes_the_companion_files_macos_writes_on_fat32(library, library_root, tmp_path, monkeypatch):
    def copy_like_fat32(src, dst):
        dst.write_bytes(src.read_bytes())
        dst.with_name("._" + dst.name).write_bytes(b"tag")
    monkeypatch.setattr(organize, "_copy", copy_like_fat32)
    dest = tmp_path / "out"
    (tmp_path / "._out").write_bytes(b"tag")
    organize.export(library, [library.by_rel[KICK]], dest)
    assert [p.name for p in dest.iterdir()] == [KICK_EXPORTED]
    assert not (tmp_path / "._out").exists()


def test_export_dry_run_writes_nothing(library, tmp_path):
    dest = tmp_path / "out"
    assert organize.export(library, [library.by_rel[KICK]], dest, dry_run=True).copied == 1
    assert not dest.exists()


def test_export_dry_run_counts_conversions(library_root, tmp_path):
    library, sample = float_sample(library_root, tone(0.2))
    dest = tmp_path / "out"
    result = organize.export(library, [sample], dest, dry_run=True)
    assert (result.copied, result.converted) == (1, 1)
    assert not dest.exists()


def test_export_into_library_is_refused(library, library_root):
    with pytest.raises(ValueError):
        organize.export(library, [library.by_rel[KICK]], library_root / "Drums")


def test_read_only_library_refuses_every_change(library_root):
    before = snapshot(library_root)
    lib = Library(library_root, read_only=True).scan()
    sample = lib.by_rel[KICK]
    with pytest.raises(PermissionError):
        organize.retag(lib, [sample], Changes(desc="x"))
    with pytest.raises(PermissionError):
        organize.trash(lib, [sample])
    with pytest.raises(PermissionError):
        organize.undo(lib)
    with pytest.raises(PermissionError):
        lib.toggle_favorite(sample)
    lib.save_prefs({"a": 1})
    assert snapshot(library_root) == before


def test_retag_unparsed_file_requires_categories(library):
    sample = library.by_rel[UNSORTED]
    with pytest.raises(ValueError, match="main, sub, type"):
        organize.plan_retag(library, [sample], Changes(desc="x"))


def test_retag_unparsed_file_measures_length_and_files_it(library, library_root):
    sample = library.by_rel[UNSORTED]
    moves = organize.retag(library, [sample], Changes(main="Drums", sub="Kick", type="OneShot"))
    assert moves[0].dst == "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.25s__my-new-kick.wav"
    assert not (library_root / "unsorted-stuff").exists()
    row = index_rows(library_root)[moves[0].dst]
    assert row["original_name"] == "my new kick.wav"
    assert row["category_source"] == "manual"


def test_lint_finds_problems(library, library_root):
    assert [i.rel for i in organize.lint(library)] == [UNSORTED]
    bad = write_wav(library_root / "Wrong/Place/Drums__Kick__OneShot__NA__NA__0.50s__misfiled.wav", 0.5)
    wrong_len = write_wav(library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__9.99s__liar.wav", 0.5)
    library.scan()
    deep = {i.rel: i for i in organize.lint(library, deep=True)}
    assert "should be at" in deep[str(bad.relative_to(library_root))].problem
    assert "9.99s" in deep[str(wrong_len.relative_to(library_root))].problem


def test_lint_fix_is_safe_and_undoable(library, library_root):
    write_wav(library_root / "Wrong/Place/Drums__Kick__OneShot__NA__NA__0.50s__misfiled.wav", 0.5)
    write_wav(library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__9.99s__liar.wav", 0.5)
    library.scan()
    assert organize.fix_lint(library, organize.lint(library, deep=True)) == 2
    assert (library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.50s__misfiled.wav").exists()
    assert (library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.50s__liar.wav").exists()
    assert (library_root / UNSORTED).exists()      # can't guess a category, so it's left alone
    organize.undo(library)
    assert (library_root / "Wrong/Place/Drums__Kick__OneShot__NA__NA__0.50s__misfiled.wav").exists()
