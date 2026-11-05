"""Auto-filing. Most of these rules exist because something got misfiled without them."""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import snapshot, write_wav
from samplecrate import cli, duplicates as D, features as fx, filing, naming, organize, query
from samplecrate.library import Library

SR = 44100
KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"


def measures(center=1000.0, decay=0.2, busy=4.0, movement=0.5, audible=8.0):
    return fx.Features(center, center / 2, center * 2, audible, decay, busy, movement, (0.0,) * 64, "hash")


SHOT = dict(busy=1.0, audible=0.3)
LOOP = dict(busy=5.0, audible=8.0)


def propose(filename, folder="", length=None, **kw):
    length = kw.get("audible", 8.0) if length is None else length
    return filing.propose(filename, folder, measures(**kw), length)


@pytest.mark.parametrize("filename,main,sub,type_", [
    ("Kick (22).wav", "Drums", "Kick", "OneShot"),
    ("BVKER_Organic_Trap_Snare_01_Wet.wav", "Drums", "Snare", "OneShot"),
    ("OXYGEN_LFSH_Kit03_Cymbal_One_Shot_Hihat_Loose.wav", "Drums", "Hat", "OneShot"),
    ("Hi-Hat Open  (21).wav", "Drums", "Hat-Open", "OneShot"),
    ("SMPL_UHT_Closed_Hat_One_Shot_Permit.wav", "Drums", "Hat-Closed", "OneShot"),
    ("Kit_05_Crash_One_Shot.wav", "Drums", "Crash", "OneShot"),
    ("AHS2_05_Bongo_Shot.wav", "Percs", "Bongo", "OneShot"),
    ("L_GB_One_Shot_808_Grande_E.wav", "Bass", "808", "OneShot"),
    ("VM_Vocal Phrase_One Shot_Find Yourself.wav", "Vocals", "Phrase", "OneShot"),
    ("SMPL_UHT_150_FX_Riser_Attack_D#.wav", "FX", "Riser", "OneShot"),
])
def test_files_one_shots(filename, main, sub, type_):
    name = propose(filename, **SHOT).name
    assert (name.main, name.sub, name.type) == (main, sub, type_)


@pytest.mark.parametrize("filename,main,sub", [
    ("WED_016_Full_Drum_Loop_85BPM.wav", "Drums", "Full"),
    ("SMPL_UHT_150_Drum_Loop_Gravity_Top.wav", "Drums", "Top"),
    ("SMPL_UHT_150_Drum_Loop_Affect_Kick.wav", "Drums", "Kick"),
    ("Beat_03_Full-No_Kick-125BPM.wav", "Drums", "Top"),
    ("PATZIK_Dreams_Music_Loop_17_keyG#m_130bpm.wav", "Melodic", "Music"),
    ("Ultra_Dance_Saxophones_08_120bpm_Am_WET.wav", "Instruments", "Sax"),
    ("LR_FellInLove_138_Piano_E_8bar.wav", "Instruments", "Piano"),
    ("MS_VUH_Bass_Loop_Chase_Bass_3_Dm_120bpm.wav", "Bass", "Bass"),
])
def test_files_loops(filename, main, sub):
    name = propose(filename, **LOOP).name
    assert (name.main, name.sub, name.type) == (main, sub, "Loop")


@pytest.mark.parametrize("filename,main,sub", [
    # Words after "loop" describe the sound. They don't name it.
    ("ESM_LHD_174_synth_pluck_loop_sharp_rhodes_sines_GMin.wav", "Synths", "Pluck"),
    ("ESM_LHD_174_drum_hat_loop_scratch_short_closed.wav", "Drums", "Hat"),
    ("ESM_LHD_174_synth_stab_loop_mango_single_voice_pitched_E.wav", "Synths", "Stab"),
    ("ESM_LHD_174_melodic_stack_loop_bird_slow_uplift_Bbmin.wav", "Melodic", "Stack"),
    ("PATZIK_UPLIFT_Keys_Loop_01_keyG#_80bpm.wav", "Instruments", "Keys"),       # a pack name, not a riser
    ("SHV_Kit01_Break_Chords_1_WET_128_Cm.wav", "Melodic", "Chords"),            # chords for the breakdown
    ("LANDR_Night Ride_Muted Trombone_keyAbmin_142bpm.wav", "Instruments", "Brass"),   # a song title, not a cymbal
])
def test_descriptors_and_titles_dont_decide_the_category(filename, main, sub):
    name = propose(filename, **LOOP).name
    assert (name.main, name.sub) == (main, sub)


def test_handle_doesnt_swallow_the_name():
    assert propose("@outroficial_808_Lamentos.wav", **SHOT).name.sub == "808"


@pytest.mark.parametrize("filename,expected", [
    ("WED_016_Full_Drum_Loop_85BPM.wav", 85),
    ("Drum Loop 7 - 140BPM.wav", 140),
    ("New_Jersy_V2_Drum_Bpm140_KeyFm.wav", 140),
    ("Fill_Dale_96pm_@soncity.wav", 96),
    ("AST_LM_75_drum_loop_peaches.wav", 75),                     # bare number that fits the loop's length
    ("OXYGEN_LFSH_Kit03_Drum_Loop_Kick_94.wav", 94),             # trailing number on a named loop
    ("wt_svc_vocal_adlib_loop_beach_Amaj_105.wav", 105),         # beside the key
])
def test_finds_tempo_in_the_name(filename, expected):
    length = 240 / expected * 4      # four bars
    assert propose(filename, length=length, busy=5.0, audible=length).name.bpm == expected


@pytest.mark.parametrize("filename", ["Ammo Vol 3 One-Shot 125 F.wav", "FS_COM_Computer_Glitch_SFX 156.wav", "SCY108_146_Bass_808_One_Shot_KeyC.wav"])
def test_numbers_on_one_shots_arent_tempos(filename):
    proposal = propose(filename, folder="wubs", **SHOT)
    assert (proposal.name.bpm if proposal.name else proposal.guesses.get("bpm")) in (None, "")


def test_impossible_tempo_is_ignored():
    assert propose("35_ChopNSoul2_Drums_999bpm.wav", busy=5.0).name.bpm is None


@pytest.mark.parametrize("filename,expected", [
    ("PATZIK_Dreams_Music_Loop_17_keyG#m_130bpm.wav", "G#min"),
    ("Kit 1 Reese Bass 128bpm KeyAmin.wav", "Amin"),
    ("Melody_Loop_37_Gm_85bpm.wav", "Gmin"),
    ("LANDR_Getaway_String & Flute_keyGbMaj_160bpm.wav", "F#maj"),
    ("3 - C minor - 125 bpm - Vocal Chop - Wet.wav", "Cmin"),
    ("Chords_Loop_10_Eb_dry_120BPM.wav", "D#"),
    ("L_GB_One_Shot_808_Grande_E.wav", "E"),
    ("Kick_Loop_41_G#_124BPM.wav", "G#"),
])
def test_finds_key_in_the_name(filename, expected):
    assert propose(filename, **LOOP).name.key == expected


@pytest.mark.parametrize("filename", [
    "Beat_09_A-125BPM.wav",                                 # variation A, not a key
    "VM_Vocal Phrase_One Shot_A Moment of Magic.wav",       # the word "A"
    "Ultra_Techno_Vocals_Rave_Like_A_Movie_138_BPM.wav",
    "Vocals - Chant 7 (Epic B).wav",
    "Ultra_G-House_Vocals_7_02_X6_100_BPM_WET.wav",
])
def test_stray_letters_arent_keys(filename):
    assert propose(filename, **LOOP).name.key is None


def test_folder_names_the_sound_when_the_filename_doesnt():
    assert propose("BS Skrillex.wav", folder="dubstep/wubs", **SHOT).name.sub == "Wub"
    assert propose("Sample 12.wav", folder="Pack/One_Shot/Percussion", **SHOT).name.main == "Percs"
    name = propose("Whatever 3.wav", folder="Pack/Loop/Synth", **LOOP).name
    assert (name.main, name.type) == ("Synths", "Loop")


def test_filename_beats_the_folder():
    assert propose("AHS2_05_Bongo_Shot.wav", folder="dubstep/wubs", **SHOT).name.sub == "Bongo"


def test_folder_decides_loop_or_one_shot_unless_the_audio_disagrees():
    assert propose("Lead 3.wav", folder="Pack/One_Shot/Synth", **SHOT).name.type == "OneShot"
    assert propose("Lead 3.wav", folder="Pack/Loop/Synth", busy=5.0, audible=8.0).name.type == "Loop"
    assert propose("Tiny Kick.wav", folder="Pack/Loop/Drums", busy=1.0, audible=0.2).name.type == "OneShot"


def test_length_is_the_real_file_length():
    assert propose("Kick 1.wav", length=1.234, busy=1.0, audible=0.3).name.length == 1.23


def test_extension_is_normalized():
    assert propose("Kick 1.WAV", **SHOT).name.ext == "wav"


@pytest.mark.parametrize("filename,folder,why", [
    ("Ammo - One Shot 44.wav", "", "Nothing in its name"),
    ("Untitled.wav", "Downloads", "Nothing in its name"),
])
def test_unknown_sounds_are_left_for_a_person(filename, folder, why):
    proposal = propose(filename, folder=folder, **SHOT)
    assert proposal.name is None and why in proposal.why
    assert proposal.guesses["type"] == "OneShot"     # what could be worked out is still offered


def test_a_hit_from_a_generic_drums_folder_is_left_for_a_person():
    proposal = propose("Sample 3.wav", folder="Pack/One_Shot/Drums", **SHOT)
    assert proposal.name is None and "which drum" in proposal.why


def test_silence_is_left_for_a_person():
    assert propose("Kick 1.wav", center=0.0, **SHOT).name is None


def test_partial_guesses_include_key_and_tempo():
    proposal = propose("Mystery 128bpm KeyAmin.wav", **LOOP)
    assert proposal.name is None
    assert proposal.guesses["bpm"] == "128" and proposal.guesses["key"] == "Amin"


def test_files_already_following_the_convention_keep_their_fields():
    proposal = propose("Synths__Pad__Loop__080bpm__Dmin__24.00s__haze.wav", length=24.0, **LOOP)
    assert proposal.name.relpath == "Synths/Loops/Pad/Synths__Pad__Loop__080bpm__Dmin__24.00s__haze.wav"


def test_every_filed_name_follows_the_convention():
    for filename in ("Kick (22).wav", "WED_016_Full_Drum_Loop_85BPM.wav", "PATZIK_Gone_Music_Loop_keyA#m_80bpm.wav", "Kit 1 Reese Bass 128bpm KeyAmin.wav", "Snare!!! (final) [v2].WAV"):
        name = propose(filename, **SHOT).name
        assert naming.check_name(name.filename) == [], filename


def test_description_keeps_the_meaningful_words():
    assert propose("ESM_LHD_174_synth_pad_loop_lime_dry_single_voices_Cmaj.wav", **LOOP).name.desc.startswith("lime-dry-single-voices")
    assert propose("VM_Vocal Phrase_One Shot_Find Yourself.wav", **SHOT).name.desc.startswith("find-yourself")


def test_description_says_how_it_compares_with_its_category():
    cohort = {("Drums", "OneShot", "Kick"): {"center": np.array([50.0, 60.0, 70.0, 80.0, 90.0]), "busy": np.array([1.0] * 5)}}
    dark = filing.propose("Kick 1.wav", "", measures(center=40, decay=0.2, busy=1, audible=0.3), 0.3, cohorts=cohort).name
    bright = filing.propose("Kick 2.wav", "", measures(center=100, decay=0.2, busy=1, audible=0.3), 0.3, cohorts=cohort).name
    assert "dark" in dark.desc and "bright" in bright.desc


# ---------------------------------------------------------------------- estimates from audio

def click_loop(bpm, beats=16, sr=SR):
    out = np.zeros(int(beats * 60 / bpm * sr), dtype="float32")
    for i in range(beats):
        start = int(i * 60 / bpm * sr)
        n = int(0.02 * sr)
        out[start:start + n] += (np.random.default_rng(i).normal(0, 0.6, n) * np.exp(-np.arange(n) / 150)).astype("float32")
    return out


@pytest.mark.parametrize("bpm", [90, 120, 140, 174])
def test_tempo_estimate_finds_a_click_loop_allowing_half_or_double(bpm):
    loop = click_loop(bpm)
    value, how = filing.estimate_bpm(loop, SR, len(loop) / SR)
    assert how == "est-grid"
    assert any(abs(value * f - bpm) <= 2 for f in (1, 2, 0.5))


def test_no_tempo_is_better_than_a_bad_one():
    held = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR * 8) / SR)).astype("float32")
    assert filing.estimate_bpm(held, SR, 8.0) == (None, None)
    uneven = click_loop(120)[: int(SR * 7.37)]       # doesn't fit a whole number of beats
    assert filing.estimate_bpm(uneven, SR, 7.37) == (None, None)


@pytest.mark.parametrize("hz,note", [(41.2, "E"), (55.0, "A"), (65.4, "C"), (98.0, "G"), (73.4, "D")])
def test_root_note_of_a_bass_hit(hz, note):
    t = np.arange(int(SR * 1.5)) / SR
    wave = ((np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(4 * np.pi * hz * t)) * np.exp(-1.5 * t) * 0.6).astype("float32")
    assert filing.root_note_from_audio(wave, SR) == note


def test_root_note_stays_quiet_on_silence_and_noise():
    assert filing.root_note_from_audio(np.zeros(SR, dtype="float32"), SR) is None
    noise = np.random.default_rng(0).normal(0, 0.3, SR * 2).astype("float32")
    assert filing.root_note_from_audio(noise, SR) is None


# ---------------------------------------------------------------------- uploading

def rich(seed, seconds=0.6):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * rng.uniform(50, 2500) * t) * rng.uniform(.1, .4) for _ in range(5))
    return ((tone + rng.normal(0, .05, len(t))) * np.exp(-3 * t) * .5).astype("float32")


def save(path, signal):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, SR)
    return path


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "New Pack"
    save(root / "Kick 7.wav", rich(1))
    save(root / "One_Shot/Synth/Glassy Thing.wav", rich(2))
    save(root / "Mystery 44.wav", rich(3))
    return root


def run(capsys, *args):
    code = cli.main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_upload_files_what_it_can_and_holds_the_rest(library_root, pack, capsys):
    code, out, _ = run(capsys, "add", "-l", library_root, pack)
    assert code == 0
    assert "filed automatically: 2" in out and "need you to file them: 1" in out and "Mystery 44.wav" in out
    lib = Library(library_root).scan()
    kick = next(s for s in lib.samples if s.index and s.index.get("original_name") == "Kick 7.wav")
    assert kick.sub == "Kick" and kick.type == "OneShot" and not kick.problems
    assert kick.rel.startswith("Drums/One-Shots/Kick/Drums__Kick__OneShot__")
    synth = next(s for s in lib.samples if s.index and s.index.get("original_name") == "Glassy Thing.wav")
    assert synth.main == "Synths" and synth.type == "OneShot"
    held = next(s for s in lib.samples if s.filename == "Mystery 44.wav")
    assert held.rel == f"{organize.INCOMING_FOLDER}/Mystery 44.wav" and held.problems


def test_upload_records_where_values_came_from(library_root, pack, capsys):
    run(capsys, "add", "-l", library_root, pack)
    lib = Library(library_root).scan()
    row = next(r for r in lib.index_rows.values() if r["original_name"] == "Glassy Thing.wav")
    assert row["category_source"] == "path" and row["source_paths"].endswith("Glassy Thing.wav")
    assert row["main"] == "Synths"


def test_upload_from_a_zip_uses_its_folders(library_root, tmp_path, capsys):
    archive = tmp_path / "pack.zip"
    source = save(tmp_path / "x.wav", rich(4))
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source, "Vol 1/Loop/Bass/Thing 3.wav")
    code, out, _ = run(capsys, "add", "-l", library_root, archive)
    assert "filed automatically: 1" in out
    lib = Library(library_root).scan()
    row = next(r for r in lib.index_rows.values() if r["original_name"] == "Thing 3.wav")
    assert row["main"] == "Bass" and row["source_paths"] == "pack.zip::Vol 1/Loop/Bass/Thing 3.wav"


def test_duplicates_are_left_out_before_filing(library_root, pack, capsys):
    shutil.copy(library_root / KICK, pack / "Kick copy.wav")
    code, out, _ = run(capsys, "add", "-l", library_root, pack)
    assert "Already in your library: 1" in out
    assert not any(s.filename == "Kick copy.wav" for s in Library(library_root).scan().samples)


def test_dry_run_shows_every_destination_and_moves_nothing(library_root, pack, capsys):
    before = {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")}
    code, out, _ = run(capsys, "add", "-l", library_root, pack, "-n")
    assert code == 0 and "-> Drums/One-Shots/Kick/" in out and "-> Incoming/ (needs you)" in out and "would be filed" in out
    assert {k: v for k, v in snapshot(library_root).items() if not k.startswith(".crate")} == before


def test_no_file_puts_everything_in_incoming(library_root, pack, capsys):
    run(capsys, "add", "-l", library_root, pack, "--no-file")
    lib = Library(library_root).scan()
    added = [s for s in lib.samples if s.is_new]
    assert len(added) == 3 and all(s.rel.startswith(organize.INCOMING_FOLDER) for s in added)


def test_one_undo_reverses_a_whole_upload(library_root, pack, tmp_path, capsys):
    before = set(snapshot(library_root)) - {p for p in snapshot(library_root) if p.startswith(".crate")}
    run(capsys, "add", "-l", library_root, pack)
    run(capsys, "undo", "-l", library_root)
    after = {p for p in snapshot(library_root) if not p.startswith(".crate")}
    assert after == before
    assert len(list((tmp_path / "trash").iterdir())) == 3
    assert (pack / "Kick 7.wav").exists()      # the originals were never touched


def test_upload_never_overwrites_an_existing_name(library_root, tmp_path, capsys):
    run(capsys, "add", "-l", library_root, save(tmp_path / "a/Kick 7.wav", rich(5)))
    run(capsys, "add", "-l", library_root, save(tmp_path / "b/Kick 7.wav", rich(6)))
    names = [s.filename for s in Library(library_root).scan().samples if s.index and s.index.get("original_name") == "Kick 7.wav"]
    assert len(names) == 2 and len(set(names)) == 2
    assert all(naming.parse_name(n) for n in names)


def test_is_new_lists_the_last_upload_and_follows_renames(library_root, pack, tmp_path, capsys):
    run(capsys, "add", "-l", library_root, pack)
    lib = Library(library_root).scan()
    new = query.parse("is:new").filter(lib.samples)
    assert len(new) == 3
    kick = next(s for s in new if s.sub == "Kick")
    moves = organize.retag(lib, [kick], organize.Changes(sub="Thump"))
    assert moves[0].dst in lib.last_upload and len(query.parse("is:new").filter(lib.samples)) == 3
    assert len(query.parse("is:new").filter(Library(library_root).scan().samples)) == 3
    run(capsys, "add", "-l", library_root, save(tmp_path / "later/Snare 9.wav", rich(7)))
    assert [s.index["original_name"] for s in query.parse("is:new").filter(Library(library_root).scan().samples)] == ["Snare 9.wav"]


def test_a_sound_repeated_in_an_upload_keeps_its_better_name(library_root, tmp_path, capsys):
    sound = rich(10)
    save(tmp_path / "p/Sample 0041.wav", sound)
    save(tmp_path / "p/Snare 41.wav", sound)
    code, out, _ = run(capsys, "add", "-l", library_root, tmp_path / "p")
    assert "Added: 1" in out and "Already in your library: 0" in out and "Repeated within this upload: 1" in out
    assert "Snare 41.wav in this upload" in out
    kept = [s for s in Library(library_root).scan().samples if s.is_new]
    assert len(kept) == 1 and kept[0].sub == "Snare"


def test_crate_file_files_what_was_dragged_in(library_root, capsys):
    save(library_root / "dragged/Snare 12.wav", rich(8))
    save(library_root / "dragged/No Idea.wav", rich(9))
    code, out, _ = run(capsys, "file", "-l", library_root, "-n")
    assert code == 0 and "-> Drums/One-Shots/Snare/" in out and "stays put" in out and "dry run" in out
    assert (library_root / "dragged/Snare 12.wav").exists()
    code, out, _ = run(capsys, "file", "-l", library_root)
    lib = Library(library_root).scan()
    filed = next(s for s in lib.samples if s.index and s.index.get("original_name") == "Snare 12.wav")
    assert filed.sub == "Snare" and not filed.problems
    assert (library_root / "dragged/No Idea.wav").exists()
    run(capsys, "undo", "-l", library_root)
    assert (library_root / "dragged/Snare 12.wav").exists()


def test_crate_file_with_nothing_to_do(tmp_path, capsys):
    root = tmp_path / "clean"
    write_wav(root / KICK, 0.2)
    code, out, _ = run(capsys, "file", "-l", root)
    assert code == 0 and "already follows" in out


def test_read_only_refuses_filing(library_root, pack, capsys):
    assert run(capsys, "add", "-l", library_root, "--read-only", pack)[0] == 2
    assert run(capsys, "file", "-l", library_root, "--read-only")[0] == 2
