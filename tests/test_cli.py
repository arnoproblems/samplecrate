import csv
import io
import json
import sys
from pathlib import Path

import pytest

from conftest import UNSORTED, snapshot, write_wav
from samplecrate import cli

KICK = "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.20s__punchy.wav"


def run(capsys, *args):
    code = cli.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_ls_table(library_root, capsys):
    code, out, _ = run(capsys, "ls", "-l", str(library_root), "kick")
    assert code == 0
    assert "DESCRIPTION" in out and "punchy" in out and "warm" not in out


def test_ls_paths_are_real_files(library_root, capsys):
    _, out, _ = run(capsys, "ls", "-l", str(library_root), "--format", "paths", "sub:kick")
    paths = out.split("\n")[:-1]
    assert len(paths) == 2 and all(Path(p).exists() for p in paths)


def test_ls_json(library_root, capsys):
    _, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "key:Am")
    records = json.loads(out)
    assert {r["description"] for r in records} == {"warm", "wide"}
    assert records[0]["camelot"] == "8A"


def test_ls_csv(library_root, capsys):
    _, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "csv", "bpm:174")
    assert next(csv.DictReader(io.StringIO(out)))["description"] == "jungle"


def test_ls_sort_and_limit(library_root, capsys):
    _, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "--sort=-bpm", "-n", "2")
    assert [r["bpm"] for r in json.loads(out)] == [174, 122]


def test_ls_accepts_negated_terms_as_plain_arguments(library_root, capsys):
    _, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "kick", "-punchy", "-new")
    assert [r["description"] for r in json.loads(out)] == ["dark-tuned"]


@pytest.mark.parametrize("term,removed", [("-kick", "punchy"), ("-sub:kick", "punchy"), ("-loop", "warm"),
                                          ("-fx", "whoosh"), ("-snare", None), ("-type:loop", "warm"), ("-bpm:120", "warm")])
def test_negated_terms_starting_with_an_option_letter(library_root, capsys, term, removed):
    """-snare must not be read as "-s nare", or -fx as "-f x"."""
    code, out, err = run(capsys, "ls", "-l", str(library_root), "-f", "json", term)
    assert code == 0, err
    found = {r["description"] for r in json.loads(out)}
    assert found
    if removed:
        assert removed not in found


def test_real_options_still_work_next_to_negated_terms(library_root, capsys):
    code, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "-n1", "-s", "bpm", "-loop")
    assert code == 0 and len(json.loads(out)) == 1
    code, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "--sort=-bpm", "-n", "1")
    assert json.loads(out)[0]["bpm"] == 174
    code, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "-s", "-bpm", "-n", "1")
    assert code == 0 and json.loads(out)[0]["bpm"] == 174      # an option's value isn't a filter term


def test_whole_filter_can_be_quoted_as_one_argument(library_root, capsys):
    _, out, _ = run(capsys, "ls", "-l", str(library_root), "-f", "json", "sub:kick type:shot")
    assert len(json.loads(out)) == 2


def test_ls_bad_filter_is_an_error(library_root, capsys):
    code, _, err = run(capsys, "ls", "-l", str(library_root), "bpm:abc")
    assert code == 2 and "bad bpm" in err


def test_ls_bad_sort_is_an_error(library_root, capsys):
    code, _, err = run(capsys, "ls", "-l", str(library_root), "--sort", "nope")
    assert code == 2 and "cannot sort" in err


def test_missing_library_is_a_clean_error(tmp_path, capsys):
    code, _, err = run(capsys, "ls", "-l", str(tmp_path / "nope"))
    assert code == 2 and "not found" in err


def test_library_from_environment(library_root, capsys, monkeypatch):
    monkeypatch.setenv("CRATE_LIBRARY", str(library_root))
    code, out, _ = run(capsys, "ls", "punchy")
    assert code == 0 and "punchy" in out


def test_stats(library_root, capsys):
    code, out, _ = run(capsys, "stats", "-l", str(library_root))
    assert code == 0
    assert "12 samples: 6 loops, 5 one-shots, 1 need fixing" in out
    assert "Drums" in out and "BPM" in out


def test_show_by_path_and_query(library_root, capsys):
    code, out, _ = run(capsys, "show", "-l", str(library_root), str(library_root / KICK))
    assert code == 0 and "Kick" in out and "peak" in out and "original ORIG_punchy.wav" in out
    assert any(0x2800 <= ord(c) <= 0x28ff for c in out)     # a drawn waveform
    code, out, _ = run(capsys, "show", "-l", str(library_root), "jungle")
    assert code == 0 and "174" in out


def test_show_no_match_is_an_error(library_root, capsys):
    code, _, err = run(capsys, "show", "-l", str(library_root), "zzzz")
    assert code == 2 and "nothing matches" in err


def test_lint_reports_and_sets_exit_code(library_root, capsys):
    code, out, _ = run(capsys, "lint", "-l", str(library_root))
    assert code == 1 and "my new kick.wav" in out and "12 files checked" in out


def test_lint_clean_library_exits_zero(library_root, capsys):
    (library_root / UNSORTED).unlink()
    code, out, _ = run(capsys, "lint", "-l", str(library_root), "--deep")
    assert code == 0 and "0 problem(s)" in out


def test_lint_names_needs_no_library(capsys):
    code, out, _ = run(capsys, "lint", "Drums__Kick__OneShot__NA__NA__0.20s__a.wav")
    assert code == 0 and ": ok" in out
    code, out, _ = run(capsys, "lint", "bad name.wav")
    assert code == 1 and "7 fields" in out


def test_retag_dry_run_changes_nothing(library_root, capsys):
    before = snapshot(library_root)
    code, out, _ = run(capsys, "retag", "-l", str(library_root), KICK, "--sub", "Snare", "--dry-run")
    assert code == 0 and "-> Drums/One-Shots/Snare/" in out and "dry run" in out
    assert snapshot(library_root) == before


def test_retag_then_undo(library_root, capsys):
    code, out, _ = run(capsys, "retag", "-l", str(library_root), KICK, "--bpm", "120", "--key", "Bbm")
    assert code == 0
    new = "Drums/One-Shots/Kick/Drums__Kick__OneShot__120bpm__A#min__0.20s__punchy.wav"
    assert (library_root / new).exists()
    code, out, _ = run(capsys, "undo", "-l", str(library_root))
    assert code == 0 and "undid" in out
    assert (library_root / KICK).exists()


def test_retag_where_selects_by_filter(library_root, capsys):
    code, out, _ = run(capsys, "retag", "-l", str(library_root), "--where", "sub:kick type:shot", "--sub", "Thump", "-n")
    assert code == 0 and "would change 2 file(s)" in out


def test_retag_needs_targets_and_valid_values(library_root, capsys):
    assert run(capsys, "retag", "-l", str(library_root), "--sub", "X")[0] == 2
    assert run(capsys, "retag", "-l", str(library_root), KICK, "--key", "H")[0] == 2
    assert run(capsys, "retag", "-l", str(library_root), "not/a/file.wav", "--sub", "X")[0] == 2


def test_export(library_root, tmp_path, capsys):
    dest = tmp_path / "out"
    code, out, _ = run(capsys, "export", "-l", str(library_root), "sub:kick", "type:shot", "--to", str(dest))
    assert code == 0 and "copied 2" in out
    assert len(list(dest.iterdir())) == 2
    assert "converted" not in out


def test_export_reports_conversions(library_root, tmp_path, capsys):
    source = write_wav(library_root / "Drums/One-Shots/Kick/Drums__Kick__OneShot__NA__NA__0.50s__deep.wav", 0.5, "FLOAT")
    dest = tmp_path / "out"
    code, out, _ = run(capsys, "export", "-l", str(library_root), "desc:deep", "--to", str(dest))
    assert code == 0 and "converted 1 from float to 24-bit" in out
    assert (dest / "Drums__Kick__OneShot__0.50s__deep.wav").exists() and source.exists()


def test_export_everything_needs_all_flag(library_root, tmp_path, capsys):
    assert run(capsys, "export", "-l", str(library_root), "--to", str(tmp_path / "o"))[0] == 2
    assert run(capsys, "export", "-l", str(library_root), "--to", str(tmp_path / "o"), "--all")[0] == 0


@pytest.mark.parametrize("before_command", [True, False])
def test_read_only_flag_works_on_either_side_of_the_command(library_root, capsys, before_command):
    """Regression: a subparser default must not silently switch a safety flag back off."""
    state = snapshot(library_root)
    options = ["-l", str(library_root), "--read-only"]
    args = [*options, "retag", KICK, "--sub", "X"] if before_command else ["retag", KICK, "--sub", "X", *options]
    code, _, err = run(capsys, *args)
    assert code == 2 and "read-only" in err
    assert snapshot(library_root) == state


def test_library_flag_works_before_the_command(library_root, capsys):
    code, out, _ = run(capsys, "-l", str(library_root), "ls", "punchy")
    assert code == 0 and "punchy" in out


def test_unknown_option_is_rejected(library_root):
    with pytest.raises(SystemExit):
        cli.main(["retag", "-l", str(library_root), KICK, "--nonsense"])
    with pytest.raises(SystemExit):
        cli.main(["ls", "--nonsense"])


def test_broken_pipe_exits_quietly(library_root, monkeypatch):
    def boom(*a, **k):
        raise BrokenPipeError
    monkeypatch.setattr(cli, "print_table", boom)
    out = open("/dev/null", "w")
    monkeypatch.setattr(sys, "stdout", out)
    assert cli.main(["ls", "-l", str(library_root)]) == 0
