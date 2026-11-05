"""Read-only checks against the actual sample library, skipped on machines that don't have it."""

from pathlib import Path

import pytest

from samplecrate import organize, query
from samplecrate.library import Library
from samplecrate.naming import parse_name

REAL = Path.home() / "Music" / "sorted"
pytestmark = pytest.mark.skipif(not REAL.is_dir(), reason="no sample library on this machine")


@pytest.fixture(scope="module")
def real():
    return Library(REAL, read_only=True).scan()


def test_every_real_file_conforms(real):
    assert real.samples
    assert [s.rel for s in real.samples if s.problems] == []


def test_every_real_name_round_trips(real):
    for sample in real.samples:
        assert parse_name(sample.filename).relpath == sample.rel


def test_index_covers_every_file(real):
    assert {s.rel for s in real.samples} == set(real.index_rows)


def test_lint_clean(real):
    assert organize.lint(real) == []


def test_filter_finds_sensible_results(real):
    kicks = query.parse("in:Drums/One-Shots/Kick").filter(real.samples)
    assert kicks and all(s.sub == "Kick" and s.type == "OneShot" for s in kicks)
    harmonic = query.parse("key:~Am type:loop").filter(real.samples)
    assert harmonic and {s.key for s in harmonic} <= {"Amin", "Cmaj", "Emin", "Dmin", "A", "C", "E", "D"}


def test_every_real_category_is_documented(real):
    doc = (Path(__file__).parent.parent / "docs" / "NAMING.md").read_text()
    for sample in real.samples:
        assert f"`{sample.main}`" in doc, sample.main
        assert f"`{sample.sub}`" in doc, sample.sub


def test_auto_filing_agrees_with_how_the_library_is_filed(real):
    """Treat every sample as new. This flatters the rules, which were written for these names, but catches regressions."""
    from samplecrate import features as fx
    from samplecrate import filing
    import os
    done = same_main = same_sub = same_type = total = 0
    for sample in real.samples[::7]:
        row = real.index_rows[sample.rel]
        features = fx.Features(1000.0, 500.0, 2000.0, float(row["length_s"]), 0.2, 3.0, 0.5, (0.0,) * 64, "x")
        folder = os.path.dirname(row["source_paths"].split(" | ")[0].split("::", 1)[-1])
        proposal = filing.propose(row["original_name"], folder, features, float(row["length_s"]))
        total += 1
        if proposal.name:
            done += 1
            same_main += proposal.name.main == sample.main
            same_sub += (proposal.name.main, proposal.name.sub) == (sample.main, sample.sub)
    assert done / total > 0.95
    assert same_main / done > 0.98 and same_sub / done > 0.97
