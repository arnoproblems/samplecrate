"""Everything that changes files: retagging, trashing, exporting, undo and lint fixes.

Every change goes through a journal in <library>/.crate/journal.jsonl so it can be
undone, including after a restart. Nothing is ever overwritten or permanently deleted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from samplecrate import audio, banks, features as fx, filing, genres as genre_tags, naming
from samplecrate.library import Bank, Library, Sample, is_audio_file
from samplecrate.naming import SampleName

JOURNAL_FILE = "journal.jsonl"
INCOMING_FOLDER = "Incoming"
LENGTH_TOLERANCE = 0.011    # filenames round to 0.01s


class _Keep:
    def __repr__(self):
        return "KEEP"


KEEP = _Keep()


@dataclass(frozen=True)
class Changes:
    """Fields to change. KEEP leaves a field alone; for bpm and key, None clears the value to NA."""
    main: str | _Keep = KEEP
    sub: str | _Keep = KEEP
    type: str | _Keep = KEEP
    bpm: int | None | _Keep = KEEP
    key: str | None | _Keep = KEEP
    desc: str | _Keep = KEEP

    @classmethod
    def from_text(cls, *, main=None, sub=None, type=None, bpm=None, key=None, desc=None) -> "Changes":
        """Build from user-typed strings. None or blank means KEEP; 'NA' clears bpm or key."""
        def given(value):
            return value is not None and str(value).strip() != ""
        return cls(
            main=naming.normalize_main(main) if given(main) else KEEP,
            sub=naming.normalize_sub(sub) if given(sub) else KEEP,
            type=naming.normalize_type(type) if given(type) else KEEP,
            bpm=naming.normalize_bpm(bpm) if given(bpm) else KEEP,
            key=naming.normalize_key(key) if given(key) else KEEP,
            desc=(naming.slugify(desc) or _raise("description needs at least one letter or digit")) if given(desc) else KEEP,
        )


def _raise(message: str):
    raise ValueError(message)


@dataclass(frozen=True)
class Move:
    src: str
    dst: str

    @property
    def renames_only(self) -> bool:
        return Path(self.src).parent == Path(self.dst).parent


# ---------------------------------------------------------------------- planning

def _target_name(sample: Sample, changes: Changes) -> SampleName:
    current = sample.name
    pick = lambda new, old: old if isinstance(new, _Keep) else new
    if current is None:
        missing = [f for f in ("main", "sub", "type") if isinstance(getattr(changes, f), _Keep)]
        if missing:
            raise ValueError(f"{sample.filename} has no valid name yet, so it needs: {', '.join(missing)}")
        return SampleName(
            main=changes.main,
            sub=changes.sub,
            type=changes.type,
            bpm=pick(changes.bpm, None),
            key=pick(changes.key, None),
            length=audio.probe_duration(sample.path),
            desc=pick(changes.desc, naming.slugify(sample.path.stem) or "untitled"),
            ext=sample.path.suffix.lstrip(".").lower(),
        )
    return SampleName(
        main=pick(changes.main, current.main),
        sub=pick(changes.sub, current.sub),
        type=pick(changes.type, current.type),
        bpm=pick(changes.bpm, current.bpm),
        key=pick(changes.key, current.key),
        length=current.length,
        desc=pick(changes.desc, current.desc),
        ext=current.ext,
    )


def _assign_paths(library: Library, wanted: list[tuple[Sample, SampleName]]) -> list[Move]:
    """Turn desired names into moves, adding -vN to any name that would collide."""
    moving = {sample.rel.lower() for sample, name in wanted if name.relpath != sample.rel}
    taken = {rel.lower() for rel in library.by_rel if rel.lower() not in moving}
    moves = []
    for sample, name in wanted:
        if name.relpath == sample.rel:
            taken.add(sample.rel.lower())
            continue
        candidate, version = name, 1
        while candidate.relpath.lower() in taken:
            version += 1
            candidate = naming.with_version(name, version)
        taken.add(candidate.relpath.lower())
        if candidate.relpath != sample.rel:
            moves.append(Move(sample.rel, candidate.relpath))
    return moves


def plan_retag(library: Library, samples: list[Sample], changes: Changes) -> list[Move]:
    return _assign_paths(library, [(s, _target_name(s, changes)) for s in samples])


def preview_name(library: Library, sample: Sample, changes: Changes) -> str:
    """The relative path a retag would produce. For live feedback, before anything is touched."""
    moves = plan_retag(library, [sample], changes)
    return moves[0].dst if moves else sample.rel


# ---------------------------------------------------------------------- applying

def _do_move(root: Path, src: str, dst: str) -> None:
    source, target = root / src, root / dst
    target.parent.mkdir(parents=True, exist_ok=True)
    # A case-only rename looks like a collision on a case-insensitive disk
    if target.exists() and not _same_file(source, target):
        raise FileExistsError(f"refusing to overwrite {dst}")
    os.rename(source, target)
    _prune_empty(source.parent, root)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _prune_empty(folder: Path, root: Path) -> None:
    """Remove folders left empty by a move. A lone Finder .DS_Store doesn't count as content."""
    while folder != root and root in folder.parents:
        try:
            entries = os.listdir(folder)
            if entries == [".DS_Store"]:
                (folder / ".DS_Store").unlink()
                entries = []
            if entries:
                return
            folder.rmdir()
        except OSError:
            return
        folder = folder.parent


def _updated_row(library: Library, sample: Sample, new: SampleName) -> dict | None:
    """The index row for a retagged sample, marking the fields a person changed as 'manual'."""
    if not library.index_fields:
        return None
    old = sample.index
    row = dict(old) if old else {f: "" for f in library.index_fields}
    if old is None:
        row["original_name"] = sample.filename
    row.update(path=new.relpath, main=new.main, sub=new.sub, type=new.type,
               bpm=new.bpm or "", key=new.key or "", length_s=f"{new.length:.2f}", description=new.desc)
    current = sample.name
    if current is None or current.bpm != new.bpm:
        row["bpm_source"] = "manual"
    if current is None or current.key != new.key:
        row["key_source"] = "manual"
    if current is None or (current.main, current.sub) != (new.main, new.sub):
        row["category_source"] = "manual"
    if current is None or current.type != new.type:
        row["type_source"] = "manual"
    return row


def apply_moves(library: Library, moves: list[Move], label: str) -> int:
    """Run a batch of moves as one undoable step. If any move fails, the finished ones are rolled back."""
    library._require_writable()
    if not moves:
        return 0
    done: list[Move] = []
    try:
        for move in moves:
            _do_move(library.root, move.src, move.dst)
            done.append(move)
    except OSError:
        for move in reversed(done):
            _do_move(library.root, move.dst, move.src)
        raise

    ops = []
    for move in moves:
        sample = library.by_rel[move.src]
        new = naming.parse_name(Path(move.dst).name)
        ops.append({"kind": "move", "src": move.src, "dst": move.dst,
                    "row": library.index_rows.get(move.src), "fav": move.src in library.favorites,
                    "genres": library.genres.get(move.src)})
        library.index_rows.pop(move.src, None)
        if move.src in library.genres:
            library.genres[move.dst] = library.genres.pop(move.src)
        library.rename_features(move.src, move.dst)
        library.rename_duplicate(move.src, move.dst)
        library.rename_in_banks(move.src, move.dst)
        row = _updated_row(library, sample, new)
        if row is not None:
            library.index_rows[move.dst] = row
        if move.src in library.favorites:
            library.favorites.discard(move.src)
            library.favorites.add(move.dst)
        if move.src in library.last_upload:
            library.last_upload.discard(move.src)
            library.last_upload.add(move.dst)
    _append_journal(library, label, ops)
    _persist(library)
    return len(moves)


def retag(library: Library, samples: list[Sample], changes: Changes) -> list[Move]:
    moves = plan_retag(library, samples, changes)
    apply_moves(library, moves, f"retag {len(moves)} file(s)")
    return moves


def _trash_dir() -> Path:
    return Path(os.environ.get("CRATE_TRASH_DIR") or Path.home() / ".Trash")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(2, 100_000):
        candidate = path.with_name(f"{path.stem} {n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"too many files named like {path}")


def trash(library: Library, samples: list[Sample]) -> int:
    """Move samples to the system Trash. Undoable."""
    library._require_writable()
    if not samples:
        return 0
    trash_dir = _trash_dir()
    trash_dir.mkdir(parents=True, exist_ok=True)
    ops = []
    try:
        for sample in samples:
            target = _unique_path(trash_dir / sample.filename)
            shutil.move(sample.path, target)
            _prune_empty(sample.path.parent, library.root)
            ops.append({"kind": "trash", "src": sample.rel, "trash": str(target),
                        "row": library.index_rows.get(sample.rel), "fav": sample.rel in library.favorites,
                        "genres": library.genres.get(sample.rel)})
            library.index_rows.pop(sample.rel, None)
            library.genres.pop(sample.rel, None)
            library.favorites.discard(sample.rel)
    finally:
        # Even if a file failed part-way, whatever did move must stay undoable
        if ops:
            _append_journal(library, f"trash {len(ops)} file(s)", ops)
            _persist(library)
    return len(ops)


def undo(library: Library) -> str | None:
    """Reverse the most recent step. Checks every file first, so it either fully applies or changes nothing."""
    library._require_writable()
    entries = _read_journal(library)
    if not entries:
        return None
    _trash_dir().mkdir(parents=True, exist_ok=True)
    entry = entries[-1]
    root = library.root
    for op in entry["ops"]:
        if op["kind"] in ("genres", "bank"):
            continue
        if op["kind"] == "add":
            if not (root / op["rel"]).exists():
                raise FileNotFoundError(f"can't undo: {op['rel']} is gone")
            continue
        current = Path(op["trash"]) if op["kind"] == "trash" else root / op["dst"]
        if not current.exists():
            raise FileNotFoundError(f"can't undo: {current} is gone")
        if (root / op["src"]).exists() and not _same_file(root / op["src"], current):
            raise FileExistsError(f"can't undo: {op['src']} already exists")
    for op in reversed(entry["ops"]):
        if op["kind"] == "genres":
            if op["before"]:
                library.genres[op["rel"]] = op["before"]
            else:
                library.genres.pop(op["rel"], None)
            continue
        if op["kind"] == "bank":
            library.restore_bank(op["name"], op["before"], op.get("index"))
            continue
        if op["kind"] == "add":
            # The file might hold changes made since, so it goes to the Trash rather than being deleted
            added = root / op["rel"]
            shutil.move(added, _unique_path(_trash_dir() / added.name))
            _prune_empty(added.parent, root)
            library.index_rows.pop(op["rel"], None)
            library.feature_cache.pop(op["rel"], None)
            library.favorites.discard(op["rel"])
            library.genres.pop(op["rel"], None)
            continue
        source = root / op["src"]
        source.parent.mkdir(parents=True, exist_ok=True)
        if op["kind"] == "trash":
            shutil.move(op["trash"], source)
        else:
            _do_move(root, op["dst"], op["src"])
            library.rename_features(op["dst"], op["src"])
            library.rename_in_banks(op["dst"], op["src"])
            if op["dst"] in library.last_upload:
                library.last_upload.discard(op["dst"])
                library.last_upload.add(op["src"])
            library.rename_duplicate(op["dst"], op["src"])
            library.index_rows.pop(op["dst"], None)
            library.favorites.discard(op["dst"])
            library.genres.pop(op["dst"], None)
        if op.get("row"):
            library.index_rows[op["src"]] = op["row"]
        if op.get("fav"):
            library.favorites.add(op["src"])
        if op.get("genres"):
            library.genres[op["src"]] = op["genres"]
    _write_journal(library, entries[:-1])
    _persist(library)
    return entry["label"]


def _persist(library: Library) -> None:
    library.write_index()
    library.write_favorites()
    library.write_genres()
    library.write_banks()
    library.write_features()
    library.write_last_upload()
    library.scan()


def _incoming_folder(root: Path) -> Path:
    """The Incoming folder, spelled as it is on disk. On a case-insensitive disk, "incoming" is the same folder."""
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.lower() == INCOMING_FOLDER.lower():
            return entry
    return root / INCOMING_FOLDER


def _free_name(library: Library, name: SampleName, taken: set[str]) -> SampleName:
    candidate, version = name, 1
    while candidate.relpath.lower() in taken or (library.root / candidate.relpath).exists():
        version += 1
        candidate = naming.with_version(name, version)
    taken.add(candidate.relpath.lower())
    return candidate


def _name_row(row: dict, name: SampleName, sources: dict[str, str]) -> None:
    row.update(path=name.relpath, main=name.main, sub=name.sub, type=name.type, bpm=name.bpm or "", key=name.key or "",
               length_s=f"{name.length:.2f}", description=name.desc)
    row.update(sources)


def add_files(library: Library, items, origins: dict[Path, str] | None = None,
              proposals: dict[Path, "filing.Proposal"] | None = None) -> list[str]:
    """Copy measured files into the library as one undoable step. The originals are left where they are.

    Files with a proposed name go straight to their proper place. The rest wait in Incoming to be
    filed by hand. Returns the new relative paths.
    """
    library._require_writable()
    if not items:
        return []
    incoming = _incoming_folder(library.root)
    taken = {rel.lower() for rel in library.by_rel}
    ops = []
    for item in items:
        proposal = (proposals or {}).get(item.path)
        filed = _free_name(library, proposal.name, taken) if proposal and proposal.name else None
        target = library.root / filed.relpath if filed else _unique_path(incoming / item.path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy(item.path, target)
        rel = target.relative_to(library.root).as_posix()
        library.feature_cache[rel] = fx.to_entry(item.features, target.stat())
        if library.index_fields:
            row = {name: "" for name in library.index_fields}
            row.update(path=rel, original_name=item.path.name,
                       source_paths=(origins or {}).get(item.path, str(item.path)))
            if filed:
                _name_row(row, filed, proposal.sources)
            library.index_rows[rel] = row
        ops.append({"kind": "add", "rel": rel})
    _append_journal(library, f"add {len(ops)} file(s)", ops)
    library.last_upload = {op["rel"] for op in ops}
    library.write_last_upload()
    _persist(library)
    return [op["rel"] for op in ops]


def file_samples(library: Library, samples: list[Sample], proposals: dict[str, "filing.Proposal"]) -> list[Move]:
    """Rename and refile samples already in the library, using worked-out names. Undoable as one step."""
    wanted = [(s, proposals[s.rel].name) for s in samples if proposals[s.rel].name]
    moves = _assign_paths(library, wanted)
    apply_moves(library, moves, f"file {len(moves)} sample(s)")
    for move in moves:
        row = library.index_rows.get(move.dst)
        proposal = proposals[move.src]
        if row is not None:
            row.update(proposal.sources)
    library.write_index()
    return moves


def set_genres(library: Library, samples: list[Sample], add=(), remove=()) -> int:
    """Add and remove genre tags as one undoable step. Returns how many samples changed."""
    library._require_writable()
    adding = [genre_tags.clean(name) for name in add]
    removing = {genre_tags.key(name) for name in remove}
    ops = []
    for sample in samples:
        before = list(sample.genres)
        merged: dict[str, str] = {genre_tags.key(tag): tag for tag in before if genre_tags.key(tag) not in removing}
        for name in adding:
            merged.setdefault(genre_tags.key(name), name)
        after = sorted(merged.values(), key=str.lower)
        if after != before:
            ops.append({"kind": "genres", "rel": sample.rel, "before": before})
            library.set_sample_genres(sample, after)
    if ops:
        _append_journal(library, f"change genres on {len(ops)} file(s)", ops)
        library.write_genres()
    return len(ops)


def suggest_genres(library: Library) -> dict[str, list[Sample]]:
    """Genres implied by where each untagged sample came from, using the provenance in _index.csv."""
    found: dict[str, list[Sample]] = {}
    for sample in library.samples:
        if sample.genres or not sample.index:
            continue
        for genre in genre_tags.suggest(sample.index.get("source_paths", ""), sample.index.get("original_name", "")):
            found.setdefault(genre, []).append(sample)
    return found


# ---------------------------------------------------------------------- banks

def _bank_name(library: Library, name: str, keeping: Bank | None = None) -> str:
    tidy = " ".join(name.split())
    if not naming.slugify(tidy):
        raise ValueError("a bank name needs at least one letter or number")
    existing = library.bank_named(tidy)
    if existing is not None and existing is not keeping:
        raise ValueError(f"there's already a bank called {existing.name}")
    return tidy


def new_bank(library: Library, name: str) -> Bank:
    library._require_writable()
    bank = Bank(_bank_name(library, name))
    library.banks.append(bank)
    _append_journal(library, f"create bank {bank.name}", [{"kind": "bank", "name": bank.name, "before": None}])
    library.write_banks()
    return bank


def next_bank_name(library: Library) -> str:
    return next(name for n in range(1, 100_000) if library.bank_named(name := f"Bank {n}") is None)


def change_bank(library: Library, bank: Bank, label: str, change) -> bool:
    """Edit a bank as one undoable step. Returns whether anything changed."""
    library._require_writable()
    before = bank.to_state()
    change(bank)
    if bank.to_state() == before:
        return False
    _append_journal(library, label, [{"kind": "bank", "name": bank.name, "before": before}])
    library.write_banks()
    return True


def rename_bank(library: Library, bank: Bank, name: str) -> bool:
    library._require_writable()
    tidy = _bank_name(library, name, keeping=bank)
    return change_bank(library, bank, f"rename bank {bank.name}", lambda b: setattr(b, "name", tidy))


def delete_bank(library: Library, bank: Bank) -> None:
    library._require_writable()
    op = {"kind": "bank", "name": None, "before": bank.to_state(), "index": library.banks.index(bank)}
    _append_journal(library, f"delete bank {bank.name}", [op])
    library.banks.remove(bank)
    library.write_banks()


@dataclass
class BankFill:
    placed: list[int] = field(default_factory=list)
    already: int = 0        # already in the bank, so left where they are
    left_over: int = 0      # no empty pad left for them


def fill_bank(library: Library, bank: Bank, samples: list[Sample]) -> BankFill:
    """Put samples on the empty pads in order, as one undoable step."""
    result = BankFill()

    def change(target: Bank) -> None:
        for sample in samples:
            pad = target.first_empty()
            if sample.rel in target.pads:
                result.already += 1
            elif pad is None:
                result.left_over += 1
            else:
                target.put(pad, sample.rel)
                result.placed.append(pad)
    change_bank(library, bank, f"put {len(samples)} sample{'s' if len(samples) != 1 else ''} on {bank.name}", change)
    return result


# ---------------------------------------------------------------------- journal

def _journal_path(library: Library) -> Path:
    return library.state_dir / JOURNAL_FILE


def _append_journal(library: Library, label: str, ops: list[dict]) -> None:
    library.state_dir.mkdir(exist_ok=True)
    with open(_journal_path(library), "a") as fh:
        fh.write(json.dumps({"label": label, "at": int(time.time()), "ops": ops}) + "\n")


def _read_journal(library: Library) -> list[dict]:
    try:
        lines = _journal_path(library).read_text().splitlines()
    except OSError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def _write_journal(library: Library, entries: list[dict]) -> None:
    path = _journal_path(library)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in entries))
    os.replace(tmp, path)


def can_undo(library: Library) -> bool:
    return bool(_read_journal(library))


def undo_depth(library: Library) -> int:
    return len(_read_journal(library))


def merge_recent(library: Library, since: int, label: str) -> None:
    """Combine every step after `since` into one, so one undo reverses the whole batch."""
    entries = _read_journal(library)
    recent = entries[since:]
    if len(recent) > 1:
        ops = [op for entry in recent for op in entry["ops"]]
        _write_journal(library, entries[:since] + [{"label": label, "at": recent[-1]["at"], "ops": ops}])


# ---------------------------------------------------------------------- export

@dataclass
class ExportResult:
    copied: int = 0
    skipped: int = 0
    converted: int = 0      # of those copied, how many were floating-point and are now 24-bit
    turned_down: int = 0    # of those converted, how many were past full scale

    @property
    def conversion_note(self) -> str:
        if not self.converted:
            return ""
        note = f"converted {self.converted} from float to 24-bit"
        return note + f", turning {self.turned_down} down to avoid clipping" if self.turned_down else note


def export(library: Library, samples: list[Sample], dest: str | os.PathLike, *, dry_run: bool = False) -> ExportResult:
    """Copy samples into a flat folder, ready for an SP-404MK2. Names lose their NA fields, and floating-point
    files, which it rejects as UNSUPPORTED FILE, become 24-bit. Existing files are skipped, never overwritten.
    Allowed on read-only libraries."""
    target = _export_target(library, dest)
    result = ExportResult()
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        _drop_mac_companion(target)
    for sample in samples:
        _export_one(sample, target / _export_name(sample), result, dry_run)
    return result


def _export_target(library: Library, dest: str | os.PathLike) -> Path:
    target = Path(dest).expanduser().resolve()
    if target == library.root or library.root in target.parents:
        raise ValueError("export folder can't be inside the library; retag files to move them around it")
    return target


def _export_name(sample: Sample) -> str:
    return sample.name.export_filename if sample.name else sample.filename


def _export_one(sample: Sample, out: Path, result: ExportResult, dry_run: bool) -> None:
    if out.exists():
        result.skipped += 1
        return
    if _is_float(sample.path):
        result.converted += 1
        if not dry_run and _write_as_pcm24(sample.path, out):
            result.turned_down += 1
    elif not dry_run:
        _copy(sample.path, out)
    if not dry_run:
        _drop_mac_companion(out)
    result.copied += 1


@dataclass
class BankExport(ExportResult):
    folder: Path | None = None
    replaces: str | None = None     # a folder left alone because it holds an older layout of this bank
    missing: list[int] = field(default_factory=list)
    short: list[int] = field(default_factory=list)
    dry_run: bool = False


def export_bank(library: Library, bank: Bank, dest: str | os.PathLike, *, dry_run: bool = False) -> BankExport:
    """Copy a bank into its own folder inside `dest`, with each file numbered by its pad, so the number is the
    first thing the SP-404MK2's screen shows. Allowed on read-only libraries."""
    parent = _export_target(library, dest)
    result = BankExport(dry_run=dry_run)
    planned: dict[str, Sample] = {}
    for pad, rel in enumerate(bank.pads):
        if rel is None:
            continue
        sample = library.by_rel.get(rel)
        if sample is None:
            result.missing.append(pad)
            continue
        planned[f"{pad + 1:02d}__{_export_name(sample)}"] = sample
        if banks.is_short(sample):
            result.short.append(pad)
    if not planned:
        raise ValueError(f"{bank.name} has no samples to export")
    slug = naming.slugify(bank.name)
    result.folder = _bank_folder(parent, slug, set(planned))
    if result.folder.name != slug:
        result.replaces = slug
    if not dry_run:
        result.folder.mkdir(parents=True, exist_ok=True)
        _drop_mac_companion(result.folder)
    for name, sample in planned.items():
        _export_one(sample, result.folder / name, result, dry_run)
    return result


def _bank_folder(parent: Path, slug: str, names: set[str]) -> Path:
    """Where a bank's export goes. A folder holding an older layout is left alone, because adding to it would put
    two samples on one pad number. Nothing on the card gets overwritten or deleted."""
    for n in range(1, 100_000):
        folder = parent / (slug if n == 1 else f"{slug}-v{n}")
        if not folder.exists() or {p.name for p in folder.iterdir() if is_audio_file(p.name) and not p.name.startswith(".")} <= names:
            return folder
    raise FileExistsError(f"too many folders named like {slug}")


def _drop_mac_companion(path: Path) -> None:
    """macOS tags files as it writes them, and on an SD card's FAT32 the tag becomes a stray ._ file beside each one."""
    path.with_name("._" + path.name).unlink(missing_ok=True)


def _is_float(path: Path) -> bool:
    try:
        return sf.info(str(path)).subtype in fx.FLOAT_SUBTYPES
    except (RuntimeError, sf.LibsndfileError):
        return False


def _write_as_pcm24(src: Path, dst: Path) -> bool:
    """Rewrite a floating-point file as 24-bit. Returns whether it was turned down: float can go past full scale,
    and turning the whole file down keeps those peaks where clipping would flatten them."""
    with sf.SoundFile(str(src)) as fh:
        container, rate = fh.format, fh.samplerate
        data = np.nan_to_num(fh.read(dtype="float64", always_2d=True), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.abs(data).max(initial=0.0))
    if peak > 1.0:
        data /= peak
    # Through a temporary name, so a full card can't leave half a file that later exports would skip
    part = dst.with_name(dst.name + ".part")
    try:
        sf.write(str(part), data, rate, subtype="PCM_24", format=container)
        part.replace(dst)
    finally:
        part.unlink(missing_ok=True)
    return peak > 1.0


def _copy(src: Path, dst: Path) -> None:
    # Copy-on-write clone where the disk supports it: instant, and no extra space
    if subprocess.run(["cp", "-c", "-p", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------- lint

@dataclass(frozen=True)
class LintIssue:
    rel: str
    problem: str
    fix: Move | None = None


def lint(library: Library, *, deep: bool = False) -> list[LintIssue]:
    """Report files that break the convention. deep also compares each name's length to the real audio."""
    issues = []
    for sample in library.samples:
        if sample.name is None:
            issues.extend(LintIssue(sample.rel, p) for p in sample.problems)
            continue
        wanted = sample.name
        problems = list(sample.problems)
        if deep:
            actual = _actual_length(sample)
            if actual is not None and abs(actual - wanted.length) > LENGTH_TOLERANCE:
                problems.append(f"name says {wanted.length:.2f}s but the audio is {actual:.2f}s")
                wanted = SampleName(wanted.main, wanted.sub, wanted.type, wanted.bpm, wanted.key,
                                    round(actual, 2), wanted.desc, wanted.ext)
        if problems:
            fix = Move(sample.rel, wanted.relpath) if wanted.relpath != sample.rel else None
            issues.extend(LintIssue(sample.rel, p, fix) for p in problems)
    return issues


def _actual_length(sample: Sample) -> float | None:
    try:
        return audio.probe_duration(sample.path)
    except Exception:
        return None


def fix_lint(library: Library, issues: list[LintIssue]) -> int:
    """Apply the safe fixes: moving misfiled files and correcting lengths. Names that don't parse need a person."""
    by_src = {i.fix.src: i.fix for i in issues if i.fix}
    wanted = [(library.by_rel[src], naming.parse_name(Path(fix.dst).name)) for src, fix in by_src.items()]
    moves = _assign_paths(library, wanted)
    return apply_moves(library, moves, f"lint fix {len(moves)} file(s)")
