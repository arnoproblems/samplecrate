import pytest

from samplecrate import query
from samplecrate.library import sort_samples


def names(library, text, marked=()):
    return sorted(s.desc for s in query.parse(text, marked).filter(library.samples))


def test_empty_query_matches_everything(library):
    assert len(query.parse("").filter(library.samples)) == len(library.samples)


def test_free_text_is_and(library):
    assert names(library, "kick dark") == ["dark-tuned"]


def test_free_text_searches_original_name(library):
    assert names(library, "orig_warm") == ["warm"]


def test_negation(library):
    assert "punchy" not in names(library, "kick -punchy")
    assert names(library, "kick -punchy -incoming -new") == ["dark-tuned"]
    assert "punchy" in names(library, "drums -type:loop")


def test_main_and_sub_prefix(library):
    assert names(library, "main:bas") == ["long", "wide"]
    assert names(library, "main:bass,fx") == ["long", "whoosh", "wide"]


def test_sub_exact(tmp_path, library):
    assert names(library, "sub:hat") == ["bright"]
    assert names(library, "sub:=hat") == []
    assert names(library, "sub:=hat-open") == ["bright"]


@pytest.mark.parametrize("value", ["loop", "Loop", "l", "loops"])
def test_type_loop(library, value):
    found = query.parse(f"type:{value}").filter(library.samples)
    assert found and all(s.type == "Loop" for s in found)


@pytest.mark.parametrize("value", ["oneshot", "shot", "os", "one-shot"])
def test_type_shot(library, value):
    found = query.parse(f"type:{value}").filter(library.samples)
    assert found and all(s.type == "OneShot" for s in found)


def test_bpm_forms(library):
    assert names(library, "bpm:174") == ["jungle"]
    assert names(library, "174bpm") == ["jungle"]
    assert names(library, "bpm:85-90") == ["hello", "wide"]
    assert names(library, "bpm:>170") == ["jungle"]
    assert names(library, "bpm:>=174") == ["jungle"]
    assert names(library, "bpm:<88") == ["wide"]
    assert names(library, "bpm:<=87") == ["wide"]
    assert "punchy" in names(library, "bpm:na")
    assert "punchy" not in names(library, "bpm:60-200")


def test_bpm_tempo_compatible_includes_half_and_double(library):
    # 174 is double 87; 3% around 174 reaches 169..179
    assert names(library, "bpm:~174") == ["jungle", "wide"]
    # 120 +-3% reaches 122
    assert {"warm", "soft", "busy-break", "whoosh"} <= set(names(library, "bpm:~120"))
    assert "hello" not in names(library, "bpm:~120")


def test_key_forms(library):
    assert names(library, "key:Am") == ["warm", "wide"]
    assert names(library, "key:Amin") == ["warm", "wide"]
    assert names(library, "key:8A") == ["warm", "wide"]
    assert names(library, "key:Am,Cmaj") == ["soft", "warm", "wide"]
    assert "punchy" in names(library, "key:na")
    assert "punchy" not in names(library, "key:Am")


def test_key_bare_note_rules(library):
    # key:F matches any F; key:Fmin also matches a bare F, whose mode nobody stated
    assert names(library, "key:F") == ["dark-tuned", "long"]
    assert names(library, "key:Fmin") == ["dark-tuned", "long"]


def test_key_flats_are_understood(tmp_path):
    from samplecrate.library import Library
    from conftest import write_wav
    root = tmp_path / "k"
    write_wav(root / "Synths/Loops/Pad/Synths__Pad__Loop__120bpm__A#min__1.00s__x.wav", 1.0)
    lib = Library(root).scan()
    assert len(query.parse("key:Bbm").filter(lib.samples)) == 1


def test_key_harmonic(library):
    # Amin mixes with Amin, Cmaj, Emin and Dmin
    assert names(library, "key:~Am") == ["hello", "soft", "warm", "wide"]


def test_len(library):
    assert names(library, "len:<0.25") == ["punchy"]
    assert "busy-break" in names(library, "len:>=2")
    assert names(library, "len:0.4-0.5") == ["bright", "long"]
    assert names(library, "len:1.38") == ["jungle"]


def test_desc(library):
    assert names(library, "desc:bus") == ["busy-break"]


def test_in_folder_respects_boundaries(library):
    assert names(library, "in:Drums/One-Shots/Kick") == ["dark-tuned", "punchy"]
    assert "punchy" in names(library, "in:Drums")
    assert names(library, "in:Dru") == []


def test_is_flags(library):
    assert names(library, "is:invalid") == ["my new kick"]
    assert names(library, "is:fav") == []
    rel = library.samples[0].rel
    assert len(query.parse("is:marked", {rel}).filter(library.samples)) == 1


@pytest.mark.parametrize("bad", ["bpm:abc", "foo:bar", "key:H", "len:x", "is:weird", "type:banana", "bpm:", "key:~"])
def test_bad_terms_are_reported_and_ignored(library, bad):
    parsed = query.parse(f"kick {bad}")
    assert parsed.errors
    # The rest of the filter still applies
    assert len(parsed.filter(library.samples)) == len(query.parse("kick").filter(library.samples))


def test_half_typed_quote_does_not_crash(library):
    query.parse('desc:"dark').filter(library.samples)
    query.parse('"').filter(library.samples)


def test_quoted_value(library):
    assert names(library, 'desc:"busy-break"') == ["busy-break"]


def test_set_term_replaces_and_removes():
    assert query.set_term("kick bpm:120", "bpm", "~90") == "bpm:~90 kick"
    assert query.set_term("kick bpm:120", "bpm", None) == "kick"
    assert query.set_term("", "in", "Drums") == "in:Drums"
    assert query.set_term("kick", "bpm", None) == "kick"


def test_sort_keeps_missing_values_last_both_ways(library):
    for reverse in (False, True):
        rows = sort_samples(library.samples, "bpm", reverse)
        values = [s.bpm for s in rows]
        first_none = values.index(None)
        assert all(v is None for v in values[first_none:])
        known = values[:first_none]
        assert known == sorted(known, reverse=reverse)


def test_sort_by_key_uses_pitch_order(library):
    rows = sort_samples(library.samples, "key")
    known = [s.key for s in rows if s.key]
    assert known.index("Cmaj") < known.index("F") < known.index("Amin")
