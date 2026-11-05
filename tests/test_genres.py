import json

import pytest

from samplecrate import cli, genres, organize, query
from samplecrate.library import Library, sort_samples
from samplecrate.organize import Changes

KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"
PAD = "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__Amin__2.00s__warm.wav"


@pytest.mark.parametrize("typed,clean", [("house", "House"), ("  HOUSE ", "House"), ("dnb", "Drum & Bass"), ("drum and bass", "Drum & Bass"),
                                         ("lofi", "Lo-Fi"), ("LO-FI", "Lo-Fi"), ("hip-hop", "Hip Hop"), ("rnb", "R&B"), ("ukg", "UK Garage"),
                                         ("Future  Bass", "Future Bass")])
def test_clean_uses_standard_spellings(typed, clean):
    assert genres.clean(typed) == clean


@pytest.mark.parametrize("bad", ["", "   ", "!!!"])
def test_clean_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        genres.clean(bad)


def test_parse_list_removes_duplicates():
    assert genres.parse_list("House, techno, house,, HOUSE") == ["House", "Techno"]


def test_set_genres_add_remove_and_undo(library, library_root):
    sample = library.by_rel[KICK]
    assert organize.set_genres(library, [sample], ["house", "Techno"]) == 1
    assert sample.genres == ("House", "Techno")
    assert json.loads((library_root / ".crate/genres.json").read_text())[KICK] == ["House", "Techno"]
    assert organize.set_genres(library, [sample], [], ["house"]) == 1
    assert sample.genres == ("Techno",)
    organize.undo(library)
    assert library.by_rel[KICK].genres == ("House", "Techno")
    organize.undo(library)
    assert library.by_rel[KICK].genres == ()


def test_setting_the_same_tags_is_a_no_op(library):
    sample = library.by_rel[KICK]
    organize.set_genres(library, [sample], ["House"])
    depth = organize.undo_depth(library)
    assert organize.set_genres(library, [sample], ["house"]) == 0
    assert organize.undo_depth(library) == depth


def test_tags_dont_touch_the_audio_or_the_name(library, library_root):
    data = (library_root / KICK).read_bytes()
    organize.set_genres(library, [library.by_rel[KICK]], ["House"])
    assert (library_root / KICK).read_bytes() == data


def test_bulk_tagging_is_one_undo(library):
    samples = [library.by_rel[KICK], library.by_rel[PAD]]
    organize.set_genres(library, samples, ["House"])
    organize.undo(library)
    assert all(not library.by_rel[r].genres for r in (KICK, PAD))


def test_tags_survive_a_restart(library, library_root):
    organize.set_genres(library, [library.by_rel[KICK]], ["House"])
    assert Library(library_root).scan().by_rel[KICK].genres == ("House",)


def test_tags_follow_a_rename_and_its_undo(library):
    organize.set_genres(library, [library.by_rel[KICK]], ["House"])
    moves = organize.retag(library, [library.by_rel[KICK]], Changes(desc="other"))
    assert library.by_rel[moves[0].dst].genres == ("House",)
    organize.undo(library)
    assert library.by_rel[KICK].genres == ("House",)


def test_tags_follow_trash_and_undo(library):
    organize.set_genres(library, [library.by_rel[KICK]], ["House"])
    organize.trash(library, [library.by_rel[KICK]])
    assert KICK not in library.genres
    organize.undo(library)
    assert library.by_rel[KICK].genres == ("House",)


def test_read_only_refuses_tags(library_root):
    lib = Library(library_root, read_only=True).scan()
    with pytest.raises(PermissionError):
        organize.set_genres(lib, [lib.by_rel[KICK]], ["House"])
    assert not (library_root / ".crate").exists()


def test_corrupt_genre_file_is_ignored(library_root):
    (library_root / ".crate").mkdir()
    (library_root / ".crate/genres.json").write_text("[oops")
    assert Library(library_root).scan().by_rel[KICK].genres == ()


# ---------------------------------------------------------------------- filtering and sorting

@pytest.fixture
def tagged(library):
    organize.set_genres(library, [library.by_rel[KICK]], ["House", "Drum & Bass"])
    organize.set_genres(library, [library.by_rel[PAD]], ["Techno", "Hip Hop"])
    return library


def found(library, text):
    return sorted(s.desc for s in query.parse(text).filter(library.samples))


@pytest.mark.parametrize("text,expected", [
    ("genre:house", ["punchy"]), ("genre:HOUSE", ["punchy"]), ("genre:hou", ["punchy"]),
    ("genre:dnb", ["punchy"]), ("genre:drum-bass", ["punchy"]), ('genre:"Drum & Bass"', ["punchy"]),
    ("genre:hip-hop", ["warm"]), ("genre:hiphop", ["warm"]),
    ("genre:house,techno", ["punchy", "warm"]),       # comma means either
    ("genre:house genre:dnb", ["punchy"]),            # two terms mean both
    ("genre:house genre:techno", []),
    ("house", ["punchy"]),                            # free text sees tags too
])
def test_genre_filters(tagged, text, expected):
    assert found(tagged, text) == expected


def test_untagged_and_excluded(tagged):
    assert "punchy" not in found(tagged, "genre:na") and "jungle" in found(tagged, "genre:na")
    assert "punchy" not in found(tagged, "-genre:house") and "warm" in found(tagged, "-genre:house")


def test_sort_by_genre_keeps_untagged_last(tagged):
    for reverse in (False, True):
        rows = sort_samples(tagged.samples, "genres", reverse)
        assert {rows[0].desc, rows[1].desc} == {"punchy", "warm"}
        assert not rows[-1].genres


# ---------------------------------------------------------------------- suggestions

@pytest.mark.parametrize("source,genre", [("ToSort/dubstep/wubs/x.wav", "Dubstep"), ("ToSort/Liquid and Hard DnB.zip::a.wav", "Drum & Bass"),
                                          ("ToSort/PS1 Jungle/a.wav", "Jungle"), ("Premium Jersey Club/a.wav", "Jersey Club"),
                                          ("Old Skool Hip Hop Vol 2.zip::a", "Hip Hop"), ("Laidback Trap & Hiphop 2.zip::a", "Trap"),
                                          ("Haze Lo-Fi Vintage Samples.zip::a", "Lo-Fi")])
def test_suggest_finds_genres_in_sources(source, genre):
    assert genre in genres.suggest(source)


@pytest.mark.parametrize("source", ["drums/kick.wav", "strapped.wav", "warehouse.wav", "groundhouses.wav"])
def test_suggest_doesnt_match_inside_other_words(source):
    assert genres.suggest(source) == []


def run(capsys, *args):
    code = cli.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_tag_and_list(library_root, capsys):
    assert run(capsys, "tag", "-l", str(library_root), KICK, "--add", "House, Techno")[0] == 0
    code, out, _ = run(capsys, "genres", "-l", str(library_root))
    assert code == 0 and "House" in out and "1 of 12 samples are tagged" in out
    code, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "genre:techno")
    assert json.loads(out)[0]["genres"] == ["House", "Techno"]
    assert run(capsys, "tag", "-l", str(library_root), KICK, "--remove", "house")[0] == 0
    code, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "genre:house")
    assert json.loads(out) == []
    code, out, _ = run(capsys, "undo", "-l", str(library_root))
    assert "undid" in out


def test_cli_tag_where_and_dry_run(library_root, capsys):
    code, out, _ = run(capsys, "tag", "-l", str(library_root), "--where", "sub:kick type:shot", "--add", "House", "-n")
    assert code == 0 and "up to 2 file(s)" in out
    assert not (library_root / ".crate/genres.json").exists()


def test_cli_tag_needs_a_change_and_targets(library_root, capsys):
    assert run(capsys, "tag", "-l", str(library_root), KICK)[0] == 2
    assert run(capsys, "tag", "-l", str(library_root), "--add", "House")[0] == 2


def test_cli_suggest_changes_nothing_until_applied(library, library_root, capsys):
    rows = (library_root / "_index.csv").read_text().replace("ToSort/ORIG_punchy.wav", "ToSort/dubstep/kick/ORIG_punchy.wav")
    (library_root / "_index.csv").write_text(rows)
    code, out, _ = run(capsys, "genres", "-l", str(library_root), "--suggest")
    assert code == 0 and "Dubstep" in out and "nothing changed" in out
    assert not (library_root / ".crate/genres.json").exists()
    code, out, _ = run(capsys, "genres", "-l", str(library_root), "--suggest", "--apply")
    assert "applied" in out
    assert Library(library_root).scan().by_rel[KICK].genres == ("Dubstep",)
    run(capsys, "undo", "-l", str(library_root))
    assert Library(library_root).scan().by_rel[KICK].genres == ()


def test_cli_range_and_audible_filters_measure_on_demand(library_root, capsys):
    code, out, err = run(capsys, "ls", "-l", str(library_root), "-f", "json", "range:low", "audible:<0.6")
    assert code == 0 and "Measuring" in err
    records = json.loads(out)
    assert records and all(r["range"] == "Low" and r["audible"] < 0.6 for r in records)
    code, out, err = run(capsys, "ls", "-l", str(library_root), "-f", "json", "range:low")
    assert "Measuring" not in err       # cached now


def test_cli_analyze(library_root, capsys):
    code, out, _ = run(capsys, "analyze", "-l", str(library_root))
    assert code == 0 and "measured 12 sample(s)" in out
    code, out, _ = run(capsys, "analyze", "-l", str(library_root))
    assert "already measured" in out


@pytest.mark.parametrize("bad", ["range:top", "genre:,", "audible:x"])
def test_bad_new_terms_are_reported(library, bad):
    assert query.parse(bad).errors
