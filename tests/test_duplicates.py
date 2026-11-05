import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import snapshot, write_wav
from samplecrate import cli, duplicates as D, features as fx, organize, query
from samplecrate.library import Library

SR = 44100
KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"
PAD = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def rich_sound(seed=1, seconds=1.5, sr=SR):
    """Something more like a real sample than a sine: noise and partials under a decay."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * rng.uniform(60, 3000) * t + rng.uniform(0, 6)) * rng.uniform(.1, .4) for _ in range(6))
    noise = np.convolve(rng.normal(0, .3, len(t)), np.ones(8) / 8, mode="same")
    wave = (tone + noise) * np.exp(-2 * t)
    return (0.8 * wave / np.abs(wave).max()).astype("float32")


def save(path, signal, sr=SR, subtype=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sr, subtype=subtype)
    return path


def item(path, rel=None):
    return D.Item(Path(path), fx.measure(path), rel)


def groups_for(*paths):
    return D.find_groups([item(p) for p in paths])


# ---------------------------------------------------------------------- the two tiers

def test_byte_identical_copy_is_same_audio(tmp_path):
    a = save(tmp_path / "a.wav", rich_sound())
    b = tmp_path / "b.wav"
    shutil.copy(a, b)
    found = groups_for(a, b)
    assert len(found) == 1 and found[0].kind == D.SAME_AUDIO


def test_same_sound_in_another_lossless_format_is_same_audio(tmp_path):
    # Integers, because soundfile scales floats differently when writing WAV and FLAC, and they'd really differ
    pcm = np.round(rich_sound() * 32000).astype("int16")
    found = groups_for(save(tmp_path / "a.wav", pcm, subtype="PCM_16"), save(tmp_path / "b.flac", pcm, subtype="PCM_16"))
    assert found and found[0].kind == D.SAME_AUDIO


def test_different_metadata_is_same_audio(tmp_path):
    sound = rich_sound()
    a, b = save(tmp_path / "a.wav", sound), save(tmp_path / "b.wav", sound)
    with sf.SoundFile(str(b), "r+") as fh:
        fh.title = "a different title"
    assert a.read_bytes() != b.read_bytes()
    assert groups_for(a, b)[0].kind == D.SAME_AUDIO


def test_24_bit_and_16_bit_copies_match(tmp_path):
    sound = rich_sound()
    found = groups_for(save(tmp_path / "a.wav", sound, subtype="PCM_24"), save(tmp_path / "b.wav", sound, subtype="PCM_16"))
    assert found and found[0].kind in (D.SAME_AUDIO, D.LIKELY)


def test_quieter_copy_is_likely(tmp_path):
    sound = rich_sound()
    found = groups_for(save(tmp_path / "a.wav", sound), save(tmp_path / "b.wav", sound * 0.5))
    assert found and found[0].kind == D.LIKELY


def test_resampled_copy_is_likely(tmp_path):
    sound = rich_sound()
    t = np.arange(int(len(sound) * 48000 / SR)) / 48000
    resampled = np.interp(t, np.arange(len(sound)) / SR, sound).astype("float32")
    found = groups_for(save(tmp_path / "a.wav", sound), save(tmp_path / "b.wav", resampled, sr=48000))
    assert found and found[0].kind == D.LIKELY


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT"])
def test_float_and_integer_files_are_compared_correctly(tmp_path, subtype):
    """Regression: float WAVs were read as near silence, so they never matched anything."""
    sound = rich_sound()
    found = groups_for(save(tmp_path / "a.wav", sound, subtype="FLOAT"), save(tmp_path / "b.wav", sound, subtype=subtype))
    assert found


def test_scores_are_remembered_between_scans(tmp_path):
    sound = rich_sound()
    items = [item(save(tmp_path / "a.wav", sound)), item(save(tmp_path / "b.wav", sound * 0.5))]
    scores = {}
    assert D.find_groups(items, scores=scores) and len(scores) == 1
    items[0].path.unlink()      # a later scan can't read this file, so it must be using the remembered score
    assert D.find_groups(items, scores=scores)


def test_library_scan_saves_scores_unless_read_only(library_root):
    sound = rich_sound()
    save(library_root / "Drums/One-Shots/Tom/Drums__Tom__OneShot__NA__NA__1.50s__a.wav", sound)
    save(library_root / "Drums/One-Shots/Tom/Drums__Tom__OneShot__NA__NA__1.50s__b.wav", sound * 0.5)
    read_only = Library(library_root, read_only=True).scan()
    fx.Analyzer(read_only).run()
    assert D.scan_library(read_only)
    assert not (library_root / ".crate").exists()
    lib = Library(library_root).scan()
    fx.Analyzer(lib).run()
    assert D.scan_library(lib)
    assert (library_root / ".crate" / D.SCORES_FILE).exists()


def test_kicks_that_merely_resemble_each_other_are_not_duplicates(tmp_path):
    """Kicks are nearly pure sine waves, so different ones correlate strongly."""
    t = np.arange(int(SR * 0.4)) / SR
    def kick(hz, fall):
        return (0.8 * np.sin(2 * np.pi * hz * t + 25 * (1 - np.exp(-t * 30))) * np.exp(-fall * t)).astype("float32")
    assert groups_for(save(tmp_path / "a.wav", kick(55, 9)), save(tmp_path / "b.wav", kick(58, 7))) == []


def test_offset_copy_is_likely(tmp_path):
    """Same sound, with a little silence in front, as some exports add."""
    sound = rich_sound()
    padded = np.concatenate([np.zeros(2000, dtype="float32"), sound])
    found = groups_for(save(tmp_path / "a.wav", sound), save(tmp_path / "b.wav", padded))
    assert found and found[0].kind == D.LIKELY


@needs_ffmpeg
def test_mp3_of_a_wav_is_likely(tmp_path):
    a = save(tmp_path / "a.wav", rich_sound())
    b = tmp_path / "b.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(a), "-b:a", "320k", str(b)], check=True)
    found = groups_for(a, b)
    assert found and found[0].kind == D.LIKELY


def test_different_sounds_are_not_duplicates(tmp_path):
    assert groups_for(save(tmp_path / "a.wav", rich_sound(1)), save(tmp_path / "b.wav", rich_sound(2))) == []


def test_same_envelope_different_note_is_not_a_duplicate(tmp_path):
    t = np.arange(SR) / SR
    env = np.exp(-6 * t)
    c = (0.6 * np.sin(2 * np.pi * 65.4 * t) * env).astype("float32")
    d = (0.6 * np.sin(2 * np.pi * 73.4 * t) * env).astype("float32")
    assert groups_for(save(tmp_path / "c.wav", c), save(tmp_path / "d.wav", d)) == []


def test_silence_doesnt_match_everything(tmp_path):
    q1, q2 = save(tmp_path / "s1.wav", np.zeros(SR, dtype="float32")), save(tmp_path / "s2.wav", np.zeros(SR // 2, dtype="float32"))
    assert groups_for(q1, q2) == []


def test_exact_only_skips_likely(tmp_path):
    sound = rich_sound()
    items = [item(save(tmp_path / "a.wav", sound)), item(save(tmp_path / "b.wav", sound * 0.5))]
    assert D.find_groups(items, exact_only=True) == []


def test_three_way_group_keeps_one(tmp_path):
    sound = rich_sound()
    found = groups_for(*(save(tmp_path / f"{n}.wav", sound) for n in "abc"))
    assert len(found) == 1 and len(found[0].items) == 3


def test_a_scan_compares_very_few_pairs(tmp_path):
    items = [item(save(tmp_path / f"{i}.wav", rich_sound(i, seconds=0.5))) for i in range(12)]
    compared = []
    D.find_groups(items, on_compare=lambda a, b, s: compared.append(s))
    assert len(compared) <= 5


def test_a_file_that_vanishes_mid_scan_is_not_fatal(tmp_path):
    sound = rich_sound()
    a, b = item(save(tmp_path / "a.wav", sound)), item(save(tmp_path / "b.wav", sound * 0.5))
    b.path.unlink()
    assert D.find_groups([a, b]) == []


# ---------------------------------------------------------------------- the library

@pytest.fixture
def measured(library):
    fx.Analyzer(library).run()
    return library


def test_clean_library_has_no_duplicates(measured):
    assert D.scan_library(measured) == []


def test_flags_power_is_dupe_and_is_copy(measured, library_root):
    shutil.copy(library_root / KICK, library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__second.wav")
    measured.scan()
    fx.Analyzer(measured).run()
    groups = D.scan_library(measured)
    assert len(groups) == 1
    D.flag_in_library(measured, groups)
    everyone = query.parse("is:dupe").filter(measured.samples)
    extras = query.parse("is:copy").filter(measured.samples)
    assert len(everyone) == 2 and len(extras) == 1
    assert everyone[0].duplicate.kind == D.SAME_AUDIO
    # A rescan keeps the flags
    measured.scan()
    assert len(query.parse("is:dupe").filter(measured.samples)) == 2


def make_set(measured, library_root, *names):
    """A set of identical files, flagged the way a full check would."""
    for name in names:
        shutil.copy(library_root / KICK, library_root / name)
    measured.scan()
    fx.Analyzer(measured).run()
    D.flag_in_library(measured, D.scan_library(measured))
    return [KICK, *names]


def test_every_set_always_has_exactly_one_keeper(measured, library_root):
    make_set(measured, library_root, "b.wav", "c.wav")
    def keepers():
        return [rel for rel, info in measured.duplicates.items() if info.keeper]
    assert len(keepers()) == 1
    # Trash the suggested keeper: another must take over, or is:copy would offer every remaining file
    keeper = keepers()[0]
    organize.trash(measured, [measured.by_rel[keeper]])
    assert len(keepers()) == 1 and len(measured.duplicates) == 2
    assert len(query.parse("is:copy").filter(measured.samples)) == 1


def test_a_lone_survivor_is_no_longer_a_duplicate(measured, library_root):
    make_set(measured, library_root, "b.wav")
    extra = next(s for s in query.parse("is:copy").filter(measured.samples))
    organize.trash(measured, [extra])
    assert measured.duplicates == {}
    assert query.parse("is:dupe").filter(measured.samples) == []


def test_results_follow_a_rename_and_its_undo(measured, library_root):
    make_set(measured, library_root, "b.wav")
    moves = organize.retag(measured, [measured.by_rel[KICK]], organize.Changes(desc="renamed"))
    new = moves[0].dst
    assert new in measured.duplicates and KICK not in measured.duplicates
    assert measured.duplicates["b.wav"].others == (new,)
    assert measured.by_rel[new].duplicate is not None
    organize.undo(measured)
    assert KICK in measured.duplicates and measured.duplicates["b.wav"].others == (KICK,)


def test_merge_adds_to_earlier_results(measured, library_root):
    make_set(measured, library_root, "b.wav")
    first = dict(measured.duplicates)
    D.flag_in_library(measured, [], merge=True)
    assert measured.duplicates == first
    D.flag_in_library(measured, [])
    assert measured.duplicates == {} and measured.duplicates_checked


def test_keeper_prefers_favorites(measured, library_root):
    other = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__a-long-winded-second-copy.wav"
    shutil.copy(library_root / KICK, library_root / other)
    measured.scan()
    fx.Analyzer(measured).run()
    measured.toggle_favorite(measured.by_rel[other])
    groups = D.scan_library(measured)
    assert groups[0].items[0].rel == other


def test_only_rels_limits_likely_search_but_exact_still_counts(measured, library_root):
    sound = rich_sound()
    a, b = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__1.50s__one.wav", "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__1.50s__two.wav"
    save(library_root / a, sound), save(library_root / b, sound * 0.5)
    measured.scan()
    fx.Analyzer(measured).run()
    assert D.scan_library(measured, only_rels=[KICK]) == []
    assert len(D.scan_library(measured, only_rels=[a])) == 1


# ---------------------------------------------------------------------- uploads

def incoming(library_root):
    """What's waiting to be filed. rich_sound names in these tests say nothing, so they stay here."""
    folder = library_root / organize.INCOMING_FOLDER
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_upload_check_splits_new_from_existing(measured, library_root, tmp_path):
    source = tmp_path / "pack"
    source.mkdir()
    shutil.copy(library_root / KICK, source / "kick copy.wav")
    save(source / "brand new.wav", rich_sound())
    items, failed = D.measure_files(sorted(source.iterdir()))
    check = D.check_upload(measured, items, failed)
    assert [i.path.name for i in check.new] == ["brand new.wav"]
    (file, kind, match), = check.duplicates
    assert file.path.name == "kick copy.wav" and kind == D.SAME_AUDIO and match.rel == KICK


def test_upload_reports_its_own_tier_even_in_a_mixed_group(measured, library_root, tmp_path):
    sound = rich_sound()
    a = "Drums/One-Shots/Tom/Drums__Tom__OneShot__NA__NA__1.50s__base.wav"
    b = "Drums/One-Shots/Tom/Drums__Tom__OneShot__NA__NA__1.50s__quiet.wav"
    save(library_root / a, sound), save(library_root / b, sound * 0.5)
    measured.scan()
    fx.Analyzer(measured).run()
    items, _ = D.measure_files([save(tmp_path / "copy.wav", sound * 0.5)])
    (file, kind, match), = D.check_upload(measured, items).duplicates
    assert kind == D.SAME_AUDIO and match.rel == b      # exactly the quieter one, though the group's keeper is another


def test_copies_inside_an_upload_count_once(measured, tmp_path):
    sound = rich_sound()
    paths = [save(tmp_path / "p/a.wav", sound), save(tmp_path / "p/b.wav", sound)]
    items, _ = D.measure_files(paths)
    check = D.check_upload(measured, items)
    assert len(check.new) == 1 and len(check.duplicates) == 1


def test_add_files_copies_into_incoming_and_undo_trashes(measured, library_root, tmp_path):
    src = save(tmp_path / "new one.wav", rich_sound())
    data = src.read_bytes()
    items, _ = D.measure_files([src])
    rels = organize.add_files(measured, items)
    assert rels == [f"{organize.INCOMING_FOLDER}/new one.wav"]
    assert src.read_bytes() == data                    # the original stays
    assert (library_root / rels[0]).read_bytes() == data
    sample = measured.by_rel[rels[0]]
    assert sample.problems and sample.features is not None      # measured once, not again
    assert measured.index_rows[rels[0]]["source_paths"] == str(src)
    organize.undo(measured)
    assert not (library_root / rels[0]).exists()
    assert (tmp_path / "trash" / "new one.wav").read_bytes() == data
    assert rels[0] not in measured.index_rows


def test_added_file_can_then_be_filed_keeping_its_origin(measured, library_root, tmp_path):
    src = save(tmp_path / "new one.wav", rich_sound())
    items, _ = D.measure_files([src])
    rel = organize.add_files(measured, items)[0]
    moves = organize.retag(measured, [measured.by_rel[rel]], organize.Changes(main="Drums", sub="Kick", type="OneShot"))
    assert moves[0].dst.startswith("Drums/One-Shots/Kick/")
    assert measured.index_rows[moves[0].dst]["original_name"] == "new one.wav"
    assert not (library_root / organize.INCOMING_FOLDER).exists()


def test_add_never_overwrites_in_incoming(measured, library_root, tmp_path):
    first = save(tmp_path / "a/same name.wav", rich_sound(1))
    second = save(tmp_path / "b/same name.wav", rich_sound(2))
    for src in (first, second):
        items, _ = D.measure_files([src])
        organize.add_files(measured, items)
    assert incoming(library_root) == ["same name 2.wav", "same name.wav"]


def test_add_uses_an_incoming_folder_however_it_is_spelled(measured, library_root, tmp_path):
    (library_root / "incoming").mkdir()
    items, _ = D.measure_files([save(tmp_path / "x.wav", rich_sound())])
    rel = organize.add_files(measured, items)[0]
    assert rel == "incoming/x.wav" and measured.by_rel[rel].features is not None


def test_read_only_refuses_adding(library_root, tmp_path):
    lib = Library(library_root, read_only=True).scan()
    items, _ = D.measure_files([save(tmp_path / "x.wav", rich_sound())])
    with pytest.raises(PermissionError):
        organize.add_files(lib, items)


# ---------------------------------------------------------------------- sources

def test_expand_sources_handles_files_folders_and_zips(tmp_path):
    a = save(tmp_path / "loose.wav", rich_sound(1))
    save(tmp_path / "folder/inner/b.wav", rich_sound(2))
    (tmp_path / "folder/readme.txt").write_text("hi")
    (tmp_path / "folder/.hidden.wav").write_bytes(b"x")
    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.write(a, "drums/kick.wav")
        zf.write(a, "__MACOSX/drums/._kick.wav")
        zf.writestr("notes.txt", "hi")
    with D.expand_sources([a, tmp_path / "folder", pack, tmp_path / "missing", tmp_path / "folder/readme.txt"]) as sources:
        names = sorted(p.name for p in sources.files)
        assert names == ["b.wav", "kick.wav", "loose.wav"]
        assert all(p.exists() for p in sources.files)
        assert any(v == "pack.zip::drums/kick.wav" for v in sources.origins.values())
        assert any("not found" in p for p in sources.problems) and any("not an audio file" in p for p in sources.problems)
    assert not any(p.exists() for p in sources.files if p in sources.origins)     # temporary files are cleaned up


def test_zip_cannot_write_outside_its_folder(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        buf = save(tmp_path / "x.wav", rich_sound())
        zf.write(buf, "../../escaped.wav")
    with D.expand_sources([evil]) as sources:
        assert sources.files and all("crate-" in str(p) for p in sources.files)
    assert not (tmp_path.parent / "escaped.wav").exists()


def test_corrupt_zip_is_reported(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with D.expand_sources([bad]) as sources:
        assert sources.files == [] and "not a readable zip" in sources.problems[0]


# ---------------------------------------------------------------------- command line

def run(capsys, *args):
    code = cli.main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_add_recognizes_duplicates(library_root, tmp_path, capsys):
    source = tmp_path / "pack"
    source.mkdir()
    shutil.copy(library_root / KICK, source / "kick copy.wav")
    save(source / "brand new.wav", rich_sound())
    code, out, _ = run(capsys, "add", "-l", library_root, source)
    assert code == 0
    assert "Added: 1" in out and "Already in your library: 1" in out and KICK in out
    assert incoming(library_root) == ["brand new.wav"]
    assert (source / "kick copy.wav").exists()


def test_cli_add_dry_run_copies_nothing(library_root, tmp_path, capsys):
    src = save(tmp_path / "n.wav", rich_sound())
    code, out, _ = run(capsys, "add", "-l", library_root, src, "-n")
    assert code == 0 and "New: 1" in out and "dry run" in out
    assert incoming(library_root) == []


def test_cli_add_zip_records_where_it_came_from(library_root, tmp_path, capsys):
    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.write(save(tmp_path / "x.wav", rich_sound()), "drums/new.wav")
    assert run(capsys, "add", "-l", library_root, pack)[0] == 0
    lib = Library(library_root).scan()
    row = next(r for r in lib.index_rows.values() if r["original_name"] == "new.wav")
    assert row["source_paths"] == "pack.zip::drums/new.wav"


def test_cli_add_flags_for_including_duplicates(library_root, tmp_path, capsys):
    sound = rich_sound()
    save(library_root / "Drums/One-Shots/Tom/Drums__Tom__OneShot__NA__NA__1.50s__base.wav", sound)
    likely = save(tmp_path / "likely.wav", sound * 0.5)
    exact = tmp_path / "exact.wav"
    shutil.copy(library_root / KICK, exact)
    code, out, _ = run(capsys, "add", "-l", library_root, likely, exact)
    assert "Added: 0" in out and "Already in your library: 2" in out
    code, out, _ = run(capsys, "add", "-l", library_root, likely, exact, "--include-likely")
    assert "Added: 1" in out and incoming(library_root) == ["likely.wav"]
    run(capsys, "undo", "-l", library_root)
    code, out, _ = run(capsys, "add", "-l", library_root, likely, exact, "--allow-duplicates")
    assert "Added: 2" in out


def test_cli_add_errors(library_root, tmp_path, capsys):
    assert run(capsys, "add", "-l", library_root, tmp_path / "nope")[0] == 2
    src = save(tmp_path / "n.wav", rich_sound())
    assert run(capsys, "add", "-l", library_root, "--read-only", src)[0] == 2


def test_cli_dupes_on_the_library(library_root, capsys):
    code, out, _ = run(capsys, "dupes", "-l", library_root)
    assert code == 0 and "0 set(s) of identical audio" in out
    shutil.copy(library_root / KICK, library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__second.wav")
    code, out, _ = run(capsys, "dupes", "-l", library_root)
    assert code == 1 and "keep  " in out and "copy  " in out


def test_cli_dupes_never_touches_samples(library_root, tmp_path, capsys):
    folder = tmp_path / "outside"
    folder.mkdir()
    shutil.copy(library_root / KICK, folder / "c.wav")
    outside = snapshot(folder)
    samples = lambda: {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")}
    before = samples()
    run(capsys, "dupes", "-l", library_root)
    run(capsys, "dupes", "-l", library_root, folder)
    assert samples() == before and snapshot(folder) == outside


def test_cli_dupes_on_an_outside_folder(library_root, tmp_path, capsys):
    folder = tmp_path / "f"
    folder.mkdir()
    shutil.copy(library_root / KICK, folder / "c.wav")
    code, out, _ = run(capsys, "dupes", "-l", library_root, folder)
    assert code == 1 and KICK in out and "c.wav" in out
    code, out, _ = run(capsys, "dupes", "-l", library_root, folder, "--no-library")
    assert code == 0


def test_cli_dupes_inside_one_folder_alone(library_root, tmp_path, capsys):
    sound = rich_sound()
    save(tmp_path / "f/a.wav", sound), save(tmp_path / "f/b.wav", sound)
    code, out, _ = run(capsys, "dupes", "-l", library_root, tmp_path / "f", "--no-library")
    assert code == 1 and "1 set(s) of identical audio" in out


def test_cli_dupes_on_a_library_subfolder_doesnt_match_files_to_themselves(library_root, capsys):
    code, out, _ = run(capsys, "dupes", "-l", library_root, library_root / "Drums")
    assert code == 0 and "0 set(s)" in out


def test_cli_filter_by_duplicates_runs_the_check_it_asks_for(library_root, capsys):
    shutil.copy(library_root / KICK, library_root / "second.wav")
    code, out, _ = run(capsys, "ls", "-l", library_root, "-f", "json", "is:dupe")
    assert code == 0 and len(json.loads(out)) == 2
    code, out, _ = run(capsys, "ls", "-l", library_root, "-f", "json", "is:copy")
    assert len(json.loads(out)) == 1


def test_other_commands_dont_run_a_check(library_root, capsys, monkeypatch):
    shutil.copy(library_root / KICK, library_root / "second.wav")
    monkeypatch.setattr(D, "scan_library", lambda *a, **k: pytest.fail("a check ran"))
    for args in (["ls", "kick"], ["stats"], ["lint"], ["analyze"]):
        run(capsys, *args[:1], "-l", library_root, *args[1:])


def test_cli_dupes_json(library_root, tmp_path, capsys):
    shutil.copy(library_root / KICK, library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__second.wav")
    code, out, _ = run(capsys, "dupes", "-l", library_root, "-f", "json")
    groups = json.loads(out)
    assert groups[0]["kind"] == D.SAME_AUDIO and sum(f["keep"] for f in groups[0]["files"]) == 1
