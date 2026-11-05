"""docs/NAMING.md is a contract with anything that writes sample names, so keep it honest."""

import re
from pathlib import Path

from samplecrate.naming import check_name, parse_name

DOC = (Path(__file__).parent.parent / "docs" / "NAMING.md").read_text()


def fenced_after(marker: str) -> list[str]:
    match = re.search(rf"<!-- {marker} -->\n```[a-z]*\n(.*?)\n```", DOC, re.DOTALL)
    assert match, f"{marker} block is missing from NAMING.md"
    return match[1].splitlines()


def test_valid_examples_parse_and_round_trip():
    names = fenced_after("valid-names")
    assert len(names) >= 5
    for name in names:
        assert check_name(name) == [], name
        assert parse_name(name).filename == name


def test_invalid_examples_are_rejected():
    names = fenced_after("invalid-names")
    assert len(names) >= 10
    for name in names:
        assert parse_name(name) is None, name


def test_documented_regex_agrees_with_the_parser():
    rx = re.compile(fenced_after("name-regex")[0])
    for name in fenced_after("valid-names"):
        fields = rx.match(name)
        assert fields, name
        parsed = parse_name(name)
        assert fields["main"] == parsed.main and fields["desc"] == parsed.desc
        assert float(fields["length"]) == parsed.length
    for name in fenced_after("invalid-names"):
        assert not rx.match(name), name


def test_opening_example_is_valid():
    assert parse_name("Synths__Pad__Loop__080bpm__Dmin__24.00s__haze-lowest.wav") is not None
    assert "Synths__Pad__Loop__080bpm__Dmin__24.00s__haze-lowest.wav" in DOC
