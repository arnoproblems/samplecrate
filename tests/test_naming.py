import pytest

from samplecrate import naming
from samplecrate.naming import SampleName, check_name, parse_name


@pytest.mark.parametrize("name, exported", [
    ("Drums__Kick__OneShot__NA__NA__0.25s__dark.wav", "Drums__Kick__OneShot__0.25s__dark.wav"),
    ("Bass__808__OneShot__NA__F#__1.16s__clean-dark.mp3", "Bass__808__OneShot__F#__1.16s__clean-dark.mp3"),
    ("Drums__Full__Loop__120bpm__NA__2.00s__busy.wav", "Drums__Full__Loop__120bpm__2.00s__busy.wav"),
    ("Synths__Pad__Loop__080bpm__Dmin__24.00s__haze.wav", "Synths__Pad__Loop__080bpm__Dmin__24.00s__haze.wav"),
    ("Drums__NA__OneShot__NA__NA__0.25s__dark.wav", "Drums__NA__OneShot__0.25s__dark.wav"),     # only the empty fields go
])
def test_export_filename_leaves_out_empty_fields(name, exported):
    assert parse_name(name).export_filename == exported


def test_parse_all_fields():
    n = parse_name("Synths__Pad__Loop__080bpm__Dmin__24.00s__haze-lowest.wav")
    assert (n.main, n.sub, n.type, n.bpm, n.key, n.length, n.desc, n.ext) == \
        ("Synths", "Pad", "Loop", 80, "Dmin", 24.0, "haze-lowest", "wav")


def test_na_fields_become_none():
    n = parse_name("Drums__Kick__OneShot__NA__NA__0.25s__dark.wav")
    assert n.bpm is None and n.key is None


def test_round_trip():
    name = "Bass__808__OneShot__NA__F#__1.16s__clean-dark.mp3"
    assert parse_name(name).filename == name


def test_folder_comes_from_type():
    assert parse_name("Drums__Kick__OneShot__NA__NA__0.25s__a.wav").folder == "Drums/One-Shots/Kick"
    assert parse_name("Drums__Kick__Loop__120bpm__NA__2.00s__a.wav").folder == "Drums/Loops/Kick"


def test_three_digit_bpm_formatting():
    n = SampleName("Drums", "Full", "Loop", 80, None, 1.0, "x", "wav")
    assert "__080bpm__" in n.filename


@pytest.mark.parametrize("bad,word", [
    ("kick.wav", "7 fields"),
    ("Drums__Kick__Shot__NA__NA__0.25s__a.wav", "type"),
    ("Drums__Kick__OneShot__80bpm__NA__0.25s__a.wav", "bpm"),
    ("Drums__Kick__OneShot__000bpm__NA__0.25s__a.wav", "bpm"),
    ("Drums__Kick__OneShot__NA__Bbm__0.25s__a.wav", "key"),
    ("Drums__Kick__OneShot__NA__E#__0.25s__a.wav", "key"),
    ("Drums__Kick__OneShot__NA__NA__0.3s__a.wav", "length"),
    ("Drums__Kick__OneShot__NA__NA__0.25s__A.wav", "description"),
    ("Drum2__Kick__OneShot__NA__NA__0.25s__a.wav", "main"),
    ("Drums__Ki_ck__OneShot__NA__NA__0.25s__a.wav", "sub"),
    ("Drums__Kick__OneShot__NA__NA__0.25s__a.WAV", "extension"),
    ("noextension", "extension"),
])
def test_check_name_explains_the_problem(bad, word):
    problems = check_name(bad)
    assert problems and any(word in p for p in problems)
    assert parse_name(bad) is None


@pytest.mark.parametrize("text,expected", [
    ("Am", "Amin"), ("am", "Amin"), ("A minor", "Amin"), ("A Minor", "Amin"), ("Amin", "Amin"),
    ("a#m", "A#min"), ("Bbm", "A#min"), ("Bbmin", "A#min"), ("bb", "A#"), ("BbMaj", "A#maj"),
    ("AM", "Amaj"), ("Cmaj", "Cmaj"), ("C major", "Cmaj"), ("F#", "F#"), ("f", "F"), ("Db", "C#"),
    ("E#", "F"), ("Cb", "B"), ("8A", "Amin"), ("8B", "Cmaj"), ("12a", "C#min"),
    ("", None), ("NA", None), ("none", None),
])
def test_normalize_key(text, expected):
    assert naming.normalize_key(text) == expected


@pytest.mark.parametrize("bad", ["H", "Amm", "13A", "A##", "kick"])
def test_normalize_key_rejects(bad):
    with pytest.raises(ValueError):
        naming.normalize_key(bad)


def test_camelot():
    assert naming.camelot("Amin") == "8A"
    assert naming.camelot("Cmaj") == "8B"
    assert naming.camelot("F#min") == "11A"
    assert naming.camelot("F") is None
    assert naming.camelot(None) is None


def test_compatible_keys():
    # Same key, relative major, and a step either way on the wheel
    assert naming.compatible_keys("Amin") == {"Amin", "Cmaj", "Emin", "Dmin"}
    # Wraps around 12 -> 1
    assert "G#min" in naming.compatible_keys("C#min")
    assert "F#min" in naming.compatible_keys("C#min")
    # A bare note could be either mode
    assert {"Fmin", "Fmaj"} <= naming.compatible_keys("F")


@pytest.mark.parametrize("value,expected", [
    (120, 120), ("120", 120), ("120.4", 120), ("120bpm", 120), ("080BPM", 80), ("", None), ("NA", None), (None, None)])
def test_normalize_bpm(value, expected):
    assert naming.normalize_bpm(value) == expected


@pytest.mark.parametrize("bad", ["abc", "0", "1000", "-5"])
def test_normalize_bpm_rejects(bad):
    with pytest.raises(ValueError):
        naming.normalize_bpm(bad)


def test_normalizers():
    assert naming.slugify("  My Dark_Kick!! 2 ") == "my-dark-kick-2"
    assert naming.normalize_sub("hat open") == "hat-open"
    assert naming.normalize_main("F X") == "FX"
    assert naming.normalize_type("one shot") == "OneShot"
    assert naming.normalize_type("shot") == "OneShot"
    assert naming.normalize_type("l") == "Loop"
    with pytest.raises(ValueError):
        naming.normalize_type("banana")


def test_with_version_replaces_existing_suffix():
    n = parse_name("Drums__Kick__OneShot__NA__NA__0.25s__dark-v2.wav")
    assert naming.with_version(n, 3).desc == "dark-v3"
