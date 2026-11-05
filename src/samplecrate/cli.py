"""Command line entry point. With no command, `crate` opens the TUI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from samplecrate import __version__, banks, duplicates, features as fx, filing, genres as genre_tags, naming, organize, query, viz
from samplecrate.library import PADS, Bank, Library, Sample, parse_sort_spec, resolve_library, sort_samples

QUERY_COMMANDS = ("tui", "ls", "stats", "show", "export", "bank")
VALUE_OPTIONS = {"-l", "--library", "-s", "--sort", "-n", "--limit", "-f", "--format", "-t", "--to"}


def build_parser() -> argparse.ArgumentParser:
    def global_options(parser, default):
        """Accepted before or after the command. Subparsers must suppress their defaults, or
        `crate --read-only ls` would have the flag silently reset by the subparser."""
        parser.add_argument("-l", "--library", default=default(None),
                            help="library folder (default: $CRATE_LIBRARY or ~/Music/sorted)")
        parser.add_argument("--read-only", action="store_true", default=default(False),
                            help="refuse any change to files in the library")
        parser.add_argument("--no-audio", action="store_true", default=default(False),
                            help="run the TUI without sound")

    parser = argparse.ArgumentParser(prog="crate", allow_abbrev=False,
                                     description="Browse, audition, filter and organize a sample library.")
    global_options(parser, lambda value: value)
    parser.add_argument("--version", action="version", version=f"crate {__version__}")
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    global_options(common, lambda value: argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", metavar="command")

    def add(name, help_text):
        return sub.add_parser(name, parents=[common], help=help_text, description=help_text, allow_abbrev=False)

    p = add("tui", "open the TUI, optionally with a starting filter")
    p.add_argument("query", nargs="*", help="filter, for example: bpm:120-130 key:~Am")

    p = add("ls", "list samples matching a filter")
    p.add_argument("query", nargs="*", help="filter, for example: kick dark -clap")
    p.add_argument("-s", "--sort", default="main", help="main, sub, type, bpm, key, len or desc; --sort=-bpm reverses")
    p.add_argument("-n", "--limit", type=int, help="show at most this many")
    p.add_argument("-f", "--format", choices=("table", "paths", "json", "csv"), default="table")

    p = add("stats", "summarize the library, or the samples matching a filter")
    p.add_argument("query", nargs="*")

    p = add("show", "print one sample's details and waveform")
    p.add_argument("target", nargs="+", help="a file path, or a filter (the first match is shown)")

    p = add("lint", "check files against the naming convention; exits 1 if any fail")
    p.add_argument("names", nargs="*", help="check these filenames instead of the library")
    p.add_argument("--deep", action="store_true", help="also compare the length in each name to the audio")
    p.add_argument("--fix", action="store_true", help="move misfiled files and correct wrong lengths (undoable)")

    p = add("retag", "change metadata, which renames and refiles the files (undoable)")
    p.add_argument("files", nargs="*", help="file paths, absolute or relative to the library")
    p.add_argument("-w", "--where", help="also select every sample matching this filter")
    p.add_argument("--main")
    p.add_argument("--sub")
    p.add_argument("--type", help="Loop or OneShot")
    p.add_argument("--bpm", help="a number, or NA to clear")
    p.add_argument("--key", help="for example Am, F#min, Bbmaj or 8A; NA to clear")
    p.add_argument("--desc")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would happen and change nothing")

    p = add("export", "copy matching samples into a folder")
    p.add_argument("query", nargs="*", help="filter; use --all to export everything")
    p.add_argument("-t", "--to", required=True, help="destination folder")
    p.add_argument("--all", action="store_true", help="allow exporting with no filter")
    p.add_argument("-n", "--dry-run", action="store_true")

    p = add("tag", "add or remove genre tags (undoable)")
    p.add_argument("files", nargs="*", help="file paths, absolute or relative to the library")
    p.add_argument("-w", "--where", help="also select every sample matching this filter")
    p.add_argument("--add", default="", help="genres to add, comma-separated: \"House, Techno\"")
    p.add_argument("--remove", default="", help="genres to remove, comma-separated")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would happen and change nothing")

    p = add("genres", "list the genres in use, or tag samples using their pack names")
    p.add_argument("--suggest", action="store_true", help="show genres implied by each untagged sample's source")
    p.add_argument("--apply", action="store_true", help="with --suggest, apply them (undoable)")

    p = add("add", "add files, folders or zips to the library, skipping samples you already have")
    p.add_argument("sources", nargs="+", help="files, folders or zips")
    p.add_argument("--include-likely", action="store_true", help="also add files that only probably match")
    p.add_argument("--allow-duplicates", action="store_true", help="add everything, even exact matches")
    p.add_argument("--no-file", action="store_true", help="put everything in Incoming, instead of filing it automatically")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would happen and copy nothing")

    p = add("file", "automatically name and file samples that are in the library but don't follow the convention yet")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would happen and move nothing")

    p = add("dupes", "find duplicate audio in the library, or in any folder; exits 1 if it finds some")
    p.add_argument("folder", nargs="?", help="a folder or zip to check, against itself and the library")
    p.add_argument("--no-library", action="store_true", help="check the folder only against itself")
    p.add_argument("--exact", action="store_true", help="only report identical audio")
    p.add_argument("-f", "--format", choices=("text", "json"), default="text")

    p = add("bank", "build 16-pad banks for the SP-404MK2; with no action, list them (changes are undoable)")
    actions = p.add_subparsers(dest="action", metavar="action")

    def act(name, help_text):
        return actions.add_parser(name, parents=[common], help=help_text, description=help_text, allow_abbrev=False)

    q = act("show", "print a bank's pads")
    q.add_argument("bank")
    q = act("new", "create an empty bank")
    q.add_argument("name")
    q = act("rename", "rename a bank")
    q.add_argument("bank")
    q.add_argument("name")
    q = act("delete", "delete a bank; its samples stay in the library")
    q.add_argument("bank")
    q = act("put", "put a sample on a pad, replacing what's there")
    q.add_argument("bank")
    q.add_argument("pad", type=int, help=f"1 to {PADS}")
    q.add_argument("target", nargs="+", help="a file path, or a filter (the first match is used)")
    q = act("clear", "empty a pad; its sample stays in the library")
    q.add_argument("bank")
    q.add_argument("pad", type=int, help=f"1 to {PADS}")
    q = act("export", "copy a bank into its own folder, with each file numbered by its pad")
    q.add_argument("bank")
    q.add_argument("-t", "--to", required=True, help="the folder to put it in, such as your SD card's IMPORT folder")
    q.add_argument("-n", "--dry-run", action="store_true")

    add("analyze", "measure every sample now, instead of waiting for the TUI to do it")
    add("undo", "reverse the most recent retag, genre change, bank change, trash or lint fix")
    return parser


def open_library(args, *, read_only: bool | None = None) -> Library:
    return Library(resolve_library(args.library), read_only=args.read_only if read_only is None else read_only).scan()


def split_negated_terms(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pull filter exclusions like -snare or -bpm:120 out of the arguments.

    argparse would read -snare as "-s nare" and -new as "-n ew". Anything longer than a
    single-letter option counts as a filter term, unless it's the value of an option or is
    an option with a number attached, like -n5.
    """
    if not any(a in QUERY_COMMANDS for a in argv):
        return argv, []
    kept, negated = [], []
    previous = None
    for arg in argv:
        if previous in VALUE_OPTIONS and arg.startswith("-"):
            # argparse won't take a value that starts with a dash, as in `--sort -bpm`, unless it's joined on
            kept[-1] = f"{previous}={arg}"
        elif re.fullmatch(r"-[A-Za-z][^\s=]+", arg) and not re.fullmatch(r"-[A-Za-z]\d+", arg):
            negated.append(arg)
        else:
            kept.append(arg)
        previous = arg
    return kept, negated


def ensure_features(library: Library) -> None:
    missing = sum(s.features is None for s in library.samples)
    if missing:
        print(f"Measuring {missing} sample(s)...", file=sys.stderr)
        fx.Analyzer(library).run()


def select(library: Library, words: list[str]) -> list[Sample]:
    text = " ".join(words)
    if query.needs_features(text):
        ensure_features(library)
    if query.needs_duplicates(text):
        # Asking for duplicates by name counts as asking for the check
        ensure_features(library)
        duplicates.flag_in_library(library, duplicates.scan_library(library))
    parsed = query.parse(text)
    if parsed.errors:
        raise ValueError("; ".join(parsed.errors))
    return parsed.filter(library.samples)


# ---------------------------------------------------------------------- output helpers

def as_record(sample: Sample) -> dict:
    return {
        "path": str(sample.path), "rel": sample.rel, "main": sample.main, "sub": sample.sub, "type": sample.type,
        "bpm": sample.bpm, "key": sample.key, "camelot": naming.camelot(sample.key), "length": sample.length,
        "description": sample.desc, "favorite": sample.favorite, "problems": sample.problems,
        "genres": list(sample.genres),
        "range": sample.features.range if sample.features else None,
        "audible": sample.features.audible if sample.features else None,
        "center_hz": sample.features.center_hz if sample.features else None,
    }


def print_table(samples: list[Sample]) -> None:
    header = ("MAIN", "SUB", "TYPE", "BPM", "KEY", "LENGTH", "DESCRIPTION")
    rows = [(s.main, s.sub, s.type or "-", str(s.bpm or "-"), s.key or "-",
             f"{s.length:.2f}s" if s.length is not None else "-", s.desc) for s in samples]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(header)]
    for row in [header, *rows]:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


def waveform_text(clip, width: int, rows: int = 5) -> str:
    return "\n".join(viz.braille_rows(viz.wave_dots(clip.mins, clip.maxs, width, rows)))


# ---------------------------------------------------------------------- commands

def cmd_tui(args, extra: list[str]) -> int:
    from samplecrate.tui.app import CrateApp
    library = open_library(args)
    words = [*(getattr(args, "query", None) or []), *extra]
    CrateApp(library, initial_query=" ".join(words), no_audio=args.no_audio).run()
    return 0


def cmd_ls(args, extra: list[str]) -> int:
    library = open_library(args)
    column, reverse = parse_sort_spec(args.sort)
    samples = sort_samples(select(library, args.query + extra), column, reverse)
    if args.limit is not None:
        samples = samples[:args.limit]
    if args.format == "paths":
        for s in samples:
            print(s.path)
    elif args.format == "json":
        json.dump([as_record(s) for s in samples], sys.stdout, indent=1)
        print()
    elif args.format == "csv":
        fields = ["rel", "main", "sub", "type", "bpm", "key", "camelot", "length", "description", "favorite",
                  "genres", "range", "audible", "path"]
        writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**as_record(s), "genres": "; ".join(s.genres)} for s in samples)
    else:
        print_table(samples)
    return 0


def cmd_stats(args, extra: list[str]) -> int:
    library = open_library(args)
    samples = select(library, args.query + extra)
    loops = sum(s.type == naming.LOOP for s in samples)
    shots = sum(s.type == naming.ONESHOT for s in samples)
    invalid = sum(bool(s.problems) for s in samples)
    print(f"{len(samples)} samples: {loops} loops, {shots} one-shots" + (f", {invalid} need fixing" if invalid else ""))
    print(f"library: {library.root}\n")

    by_main = Counter(s.main for s in samples)
    width = max((len(m) for m in by_main), default=4)
    print(f"{'':{width}}  {'total':>6} {'loops':>6} {'shots':>6}")
    for main, count in by_main.most_common():
        group = [s for s in samples if s.main == main]
        print(f"{main:{width}}  {count:>6} {sum(s.type == naming.LOOP for s in group):>6} "
              f"{sum(s.type == naming.ONESHOT for s in group):>6}")

    tempos = Counter(s.bpm // 10 * 10 for s in samples if s.bpm)
    if tempos:
        print("\nBPM")
        top = max(tempos.values())
        for start in sorted(tempos):
            print(f"{start:>4}s {tempos[start]:>5} {'█' * max(1, round(tempos[start] / top * 40))}")
    keys = Counter(s.key for s in samples if s.key)
    if keys:
        print("\nKey (most common)")
        print("  ".join(f"{k} {n}" for k, n in keys.most_common(12)))
    genres = Counter(tag for s in samples for tag in s.genres)
    if genres:
        print("\nGenre")
        print("  ".join(f"{k} {n}" for k, n in genres.most_common()))
    return 0


def _resolve_file(library: Library, text: str) -> Sample | None:
    path = Path(text).expanduser()
    rel = text
    if path.is_absolute() or path.exists():
        try:
            rel = path.resolve().relative_to(library.root).as_posix()
        except ValueError:
            return None
    return library.by_rel.get(rel)


def _resolve_target(library: Library, words: list[str]) -> Sample:
    """A file path, or else the first sample matching a filter."""
    sample = _resolve_file(library, words[0]) if len(words) == 1 else None
    if sample is None:
        matches = select(library, words)
        if not matches:
            raise ValueError(f"nothing matches: {' '.join(words)}")
        sample = matches[0]
    return sample


def cmd_show(args, extra: list[str]) -> int:
    from samplecrate import audio
    library = open_library(args)
    sample = _resolve_target(library, args.target + extra)
    width = min(shutil.get_terminal_size((100, 24)).columns, 120)
    clip = audio.load_clip(sample.path)
    print(sample.filename)
    print(f"  folder   {Path(sample.rel).parent}")
    print(f"  type     {sample.main} / {sample.sub} / {sample.type or '-'}")
    print(f"  tempo    {sample.bpm or '-'}" + (f"  ({sample.bars:.1f} bars)" if sample.bars else ""))
    code = naming.camelot(sample.key)
    print(f"  key      {sample.key or '-'}" + (f"  (Camelot {code})" if code else ""))
    print(f"  length   {clip.duration:.2f}s  {clip.sr} Hz  {clip.channels} ch  {clip.format}")
    print(f"  level    peak {clip.peak_db:.1f} dB, RMS {clip.rms_db:.1f} dB")
    features = sample.features or fx.measure(sample.path)
    print(f"  sound    {features.range} range, {features.audible:.2f}s audible")
    if sample.genres:
        print(f"  genres   {', '.join(sample.genres)}")
    if sample.index and sample.index.get("original_name"):
        print(f"  original {sample.index['original_name']}")
    for problem in sample.problems:
        print(f"  problem  {problem}")
    print()
    print(waveform_text(clip, width))
    return 0


def cmd_lint(args) -> int:
    if args.names:
        failed = 0
        for name in args.names:
            problems = naming.check_name(os.path.basename(name))
            if problems:
                failed += 1
                print(f"{name}:")
                for problem in problems:
                    print(f"  {problem}")
            else:
                print(f"{name}: ok")
        return 1 if failed else 0
    library = open_library(args)
    issues = organize.lint(library, deep=args.deep or args.fix)
    for issue in issues:
        print(f"{issue.rel}: {issue.problem}")
    print(f"{len(library.samples)} files checked, {len(issues)} problem(s)")
    if args.fix and issues:
        fixed = organize.fix_lint(library, issues)
        print(f"fixed {fixed} file(s); `crate undo` reverses this")
        return 1 if organize.lint(library) else 0
    return 1 if issues else 0


def cmd_retag(args) -> int:
    library = open_library(args)
    samples = []
    for text in args.files:
        sample = _resolve_file(library, text)
        if sample is None:
            raise ValueError(f"not in the library: {text}")
        samples.append(sample)
    if args.where:
        samples += select(library, [args.where])
    samples = list({s.rel: s for s in samples}.values())
    if not samples:
        raise ValueError("name some files or use --where to select them")
    changes = organize.Changes.from_text(main=args.main, sub=args.sub, type=args.type,
                                         bpm=args.bpm, key=args.key, desc=args.desc)
    moves = organize.plan_retag(library, samples, changes)
    for move in moves:
        print(f"{move.src}\n  -> {move.dst}")
    if args.dry_run:
        print(f"dry run: would change {len(moves)} file(s)")
    else:
        organize.apply_moves(library, moves, f"retag {len(moves)} file(s)")
        print(f"changed {len(moves)} file(s); `crate undo` reverses this")
    return 0


def cmd_export(args, extra: list[str]) -> int:
    library = open_library(args)
    words = args.query + extra
    if not words and not args.all:
        raise ValueError("give a filter, or --all to export the whole library")
    samples = select(library, words)
    result = organize.export(library, samples, args.to, dry_run=args.dry_run)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {result.copied} file(s) to {args.to}" + (f"; skipped {result.skipped} already there" if result.skipped else ""))
    if result.conversion_note:
        print(result.conversion_note)
    return 0


def _pick(library: Library, files: list[str], where: str | None) -> list[Sample]:
    samples = []
    for text in files:
        sample = _resolve_file(library, text)
        if sample is None:
            raise ValueError(f"not in the library: {text}")
        samples.append(sample)
    if where:
        samples += select(library, [where])
    samples = list({s.rel: s for s in samples}.values())
    if not samples:
        raise ValueError("name some files or use --where to select them")
    return samples


def cmd_tag(args) -> int:
    library = open_library(args)
    samples = _pick(library, args.files, args.where)
    add, remove = genre_tags.parse_list(args.add), genre_tags.parse_list(args.remove)
    if not add and not remove:
        raise ValueError("say what to change with --add or --remove")
    if args.dry_run:
        print(f"dry run: would add {add or 'nothing'} and remove {remove or 'nothing'} on up to {len(samples)} file(s)")
        return 0
    changed = organize.set_genres(library, samples, add, remove)
    print(f"changed genres on {changed} file(s); `crate undo` reverses this")
    return 0


def cmd_genres(args) -> int:
    library = open_library(args)
    if not args.suggest:
        counts = Counter(tag for s in library.samples for tag in s.genres)
        for genre, count in counts.most_common():
            print(f"{count:>6}  {genre}")
        tagged = sum(bool(s.genres) for s in library.samples)
        print(f"{tagged} of {len(library.samples)} samples are tagged")
        return 0
    found = organize.suggest_genres(library)
    for genre, samples in sorted(found.items(), key=lambda item: -len(item[1])):
        print(f"{len(samples):>6}  {genre}")
    print(f"{len({s.rel for group in found.values() for s in group})} of {len(library.samples)} samples have a clue in their source")
    if args.apply:
        before = organize.undo_depth(library)
        for genre, samples in found.items():
            organize.set_genres(library, samples, [genre])
        organize.merge_recent(library, before, "tag genres from pack names")
        print("applied; `crate undo` reverses this")
    else:
        print("nothing changed; add --apply to tag them")
    return 0


def cmd_add(args) -> int:
    library = open_library(args)
    if not args.dry_run:
        library._require_writable()
    ensure_features(library)
    with duplicates.expand_sources(args.sources) as sources:
        for problem in sources.problems:
            print(problem, file=sys.stderr)
        if not sources.files:
            raise ValueError("no audio files found")
        print(f"Checking {len(sources.files)} file(s)...", file=sys.stderr)
        items, failed = duplicates.measure_files(sources.files)
        check = duplicates.check_upload(library, items, failed)
        to_add = list(check.new)
        kept = []
        for entry in check.duplicates:
            item, kind, _ = entry
            if args.allow_duplicates or (args.include_likely and kind == duplicates.LIKELY):
                to_add.append(item)
            else:
                kept.append(entry)
        check.duplicates = kept
        check.new = to_add
        proposals = None
        if not args.no_file:
            proposals = filing.propose_for_upload(library, to_add, sources)
        if args.dry_run:
            print("\n".join(duplicates.describe_upload(check, proposals=proposals)))
            if proposals:
                print()
                for item in to_add:
                    name = proposals[item.path].name
                    print(f"{item.path.name}\n  -> {name.relpath if name else 'Incoming/ (needs you)'}")
            print("dry run: nothing was copied")
            return 0
        added = organize.add_files(library, to_add, sources.origins, proposals)
    print("\n".join(duplicates.describe_upload(check, len(added), proposals=proposals)))
    if added:
        print("\nReview them in crate with the filter is:new. `crate undo` reverses this whole upload.")
    return 0


def cmd_file(args) -> int:
    library = open_library(args)
    if not args.dry_run:
        library._require_writable()
    samples = [s for s in library.samples if s.problems]
    if not samples:
        print("everything already follows the naming convention")
        return 0
    proposals = filing.propose_for_samples(library, samples)
    for sample in samples:
        proposal = proposals[sample.rel]
        print(f"{sample.rel}\n  -> {proposal.name.relpath if proposal.name else 'stays put: ' + proposal.why}")
    movable = sum(bool(p.name) for p in proposals.values())
    if args.dry_run:
        print(f"dry run: would file {movable} of {len(samples)}")
        return 0
    moves = organize.file_samples(library, samples, proposals)
    print(f"filed {len(moves)} of {len(samples)}; `crate undo` reverses this")
    return 0


def _group_record(group: duplicates.Group) -> dict:
    return {"kind": group.kind, "files": [{"path": str(i.path), "rel": i.rel, "keep": n == 0} for n, i in enumerate(group.items)]}


def cmd_dupes(args) -> int:
    library = open_library(args)
    if args.folder is None:
        ensure_features(library)
        groups = duplicates.scan_library(library, exact_only=args.exact)
        checked = len(library.samples)
    else:
        target = Path(args.folder).expanduser().resolve()
        if target == library.root or library.root in target.parents:
            # Already in the library: its measurements exist, and a file must not match itself
            ensure_features(library)
            prefix = "" if target == library.root else target.relative_to(library.root).as_posix() + "/"
            rels = [s.rel for s in library.samples if s.rel.startswith(prefix)]
            groups = duplicates.scan_library(library, only_rels=rels, exact_only=args.exact)
            checked = len(rels)
        else:
            with duplicates.expand_sources([target]) as sources:
                for problem in sources.problems:
                    print(problem, file=sys.stderr)
                if not sources.files:
                    raise ValueError("no audio files found")
                print(f"Checking {len(sources.files)} file(s)...", file=sys.stderr)
                items, failed = duplicates.measure_files(sources.files)
                pool = list(items)
                if not args.no_library:
                    ensure_features(library)
                    pool += duplicates.library_items(library)
                groups = duplicates.find_groups(pool, only=items, exact_only=args.exact)
                checked = len(items)
                for path in failed:
                    print(f"couldn't read {path}", file=sys.stderr)
    if args.format == "json":
        json.dump([_group_record(g) for g in groups], sys.stdout, indent=1)
        print()
    else:
        print("\n".join(duplicates.describe_groups(groups)), end="\n" if groups else "")
        same = sum(g.kind == duplicates.SAME_AUDIO for g in groups)
        print(f"{checked} file(s) checked: {same} set(s) of identical audio, {len(groups) - same} likely")
    return 1 if groups else 0


def cmd_analyze(args) -> int:
    library = open_library(args)
    missing = sum(s.features is None for s in library.samples)
    if not missing:
        print(f"all {len(library.samples)} samples are already measured")
        return 0
    analyzer = fx.Analyzer(library)
    def progress(done, total):
        if done % 100 == 0 or done == total:
            print(f"\rmeasured {done}/{total}", end="", file=sys.stderr, flush=True)
    analyzer.run(progress)
    print(file=sys.stderr)
    print(f"measured {analyzer.done - len(analyzer.failed)} sample(s)" + (f"; couldn't read {len(analyzer.failed)}" if analyzer.failed else ""))
    return 0


def _named_bank(library: Library, name: str) -> Bank:
    bank = library.bank_named(name)
    if bank is None:
        known = ", ".join(b.name for b in library.banks) or "none yet"
        raise ValueError(f"no bank called {name} (banks: {known})")
    return bank


def _pad_index(number: int) -> int:
    if not 1 <= number <= PADS:
        raise ValueError(f"pad must be 1 to {PADS}, not {number}")
    return number - 1


def _print_bank(library: Library, bank: Bank) -> None:
    reading = banks.read_bank(library, bank)
    facts = [f"{bank.filled} of {PADS} pads"]
    facts += [f"tempo {reading.tempo}"] if reading.tempo else []
    facts += [f"key {reading.key}"] if reading.key else []
    print(f"{bank.name}  ({', '.join(facts)})")
    for pad, rel in enumerate(bank.pads):
        notes = reading.notes.get(pad)
        print(f"{pad + 1:>3}  {rel or '-'}" + (f"  [{', '.join(notes)}]" if notes else ""))


def cmd_bank(args, extra: list[str]) -> int:
    library = open_library(args)
    action = args.action
    if action is None:
        for bank in library.banks:
            print(f"{bank.filled:>3} of {PADS}  {bank.name}")
        if not library.banks:
            print("no banks yet; press B in the app, or use `crate bank new NAME`")
    elif action == "new":
        print(f"created {organize.new_bank(library, args.name).name}")
    elif action == "show":
        _print_bank(library, _named_bank(library, args.bank))
    elif action == "rename":
        organize.rename_bank(library, _named_bank(library, args.bank), args.name)
        print(f"renamed to {' '.join(args.name.split())}")
    elif action == "delete":
        organize.delete_bank(library, _named_bank(library, args.bank))
        print("deleted; `crate undo` brings it back")
    elif action == "put":
        bank, pad = _named_bank(library, args.bank), _pad_index(args.pad)
        sample = _resolve_target(library, args.target + extra)
        organize.change_bank(library, bank, f"put {sample.desc} on pad {pad + 1} of {bank.name}", lambda b: b.put(pad, sample.rel))
        print(f"pad {pad + 1}: {sample.rel}")
    elif action == "clear":
        bank, pad = _named_bank(library, args.bank), _pad_index(args.pad)
        organize.change_bank(library, bank, f"clear pad {pad + 1} of {bank.name}", lambda b: b.put(pad, None))
        print(f"cleared pad {pad + 1}")
    elif action == "export":
        result = organize.export_bank(library, _named_bank(library, args.bank), args.to, dry_run=args.dry_run)
        print("\n".join(banks.describe_export(result)))
    return 0


def cmd_undo(args) -> int:
    library = open_library(args)
    label = organize.undo(library)
    print(f"undid: {label}" if label else "nothing to undo")
    return 0


def main(argv: list[str] | None = None) -> int:
    kept, extra = split_negated_terms(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(kept)
    try:
        match args.command:
            case None | "tui":
                return cmd_tui(args, extra)
            case "ls":
                return cmd_ls(args, extra)
            case "stats":
                return cmd_stats(args, extra)
            case "show":
                return cmd_show(args, extra)
            case "lint":
                return cmd_lint(args)
            case "retag":
                return cmd_retag(args)
            case "export":
                return cmd_export(args, extra)
            case "tag":
                return cmd_tag(args)
            case "genres":
                return cmd_genres(args)
            case "add":
                return cmd_add(args)
            case "file":
                return cmd_file(args)
            case "dupes":
                return cmd_dupes(args)
            case "analyze":
                return cmd_analyze(args)
            case "bank":
                return cmd_bank(args, extra)
            case "undo":
                return cmd_undo(args)
    except BrokenPipeError:
        # Reader went away, for example `crate ls | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (ValueError, OSError) as exc:
        print(f"crate: error: {exc}", file=sys.stderr)
        return 2
    return 0
