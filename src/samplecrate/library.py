"""The sample library: scanning a folder, and the optional _index.csv and favorites beside it.

Filenames are the source of truth. _index.csv only adds provenance, such as the
original name and where a value came from, so the library works without it.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from samplecrate import features as fx
from samplecrate import genres as genre_tags
from samplecrate import naming
from samplecrate.naming import SampleName

INDEX_FILE = "_index.csv"
STATE_DIR = ".crate"
FAVORITES_FILE = "favorites.json"
GENRES_FILE = "genres.json"
BANKS_FILE = "banks.json"
LAST_UPLOAD_FILE = "last_upload.json"
PREFS_FILE = "prefs.json"
UNSORTED = "Unsorted"
PADS = 16

DEFAULT_LIBRARY = Path.home() / "Music" / "sorted"


def resolve_library(explicit: str | os.PathLike | None = None) -> Path:
    """Library path: the explicit argument, then $CRATE_LIBRARY, then ~/Music/sorted."""
    chosen = explicit or os.environ.get("CRATE_LIBRARY") or DEFAULT_LIBRARY
    return Path(chosen).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class DuplicateInfo:
    kind: str
    others: tuple[str, ...]     # the other files with this sound
    keeper: bool                # the copy we'd suggest keeping


@dataclass(slots=True, eq=False)
class Sample:
    rel: str
    path: Path
    name: SampleName | None
    problems: list[str] = field(default_factory=list)
    index: dict | None = None
    favorite: bool = False
    haystack: str = ""
    features: fx.Features | None = None
    genres: tuple[str, ...] = ()
    duplicate: DuplicateInfo | None = None
    is_new: bool = False        # part of the most recent upload

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def main(self) -> str:
        return self.name.main if self.name else UNSORTED

    @property
    def sub(self) -> str:
        return self.name.sub if self.name else ""

    @property
    def type(self) -> str | None:
        return self.name.type if self.name else None

    @property
    def bpm(self) -> int | None:
        return self.name.bpm if self.name else None

    @property
    def key(self) -> str | None:
        return self.name.key if self.name else None

    @property
    def length(self) -> float | None:
        return self.name.length if self.name else None

    @property
    def desc(self) -> str:
        return self.name.desc if self.name else self.path.stem

    @property
    def bars(self) -> float | None:
        """Length in 4/4 bars, when the tempo is known."""
        if self.length and self.bpm:
            return self.length * self.bpm / 240
        return None


@dataclass(slots=True, eq=False)
class Bank:
    """16 samples laid out like the SP-404MK2's pads. Index 0 is pad 1. A pad can name a file that's gone, such
    as one in the Trash, so undoing that brings the pad back with it."""
    name: str
    pads: list[str | None] = field(default_factory=lambda: [None] * PADS)

    @property
    def filled(self) -> int:
        return sum(rel is not None for rel in self.pads)

    def first_empty(self) -> int | None:
        return next((pad for pad, rel in enumerate(self.pads) if rel is None), None)

    def put(self, pad: int, rel: str | None) -> None:
        self.pads[pad] = rel

    def swap(self, a: int, b: int) -> None:
        self.pads[a], self.pads[b] = self.pads[b], self.pads[a]

    def to_state(self) -> dict:
        return {"name": self.name, "pads": {str(pad + 1): rel for pad, rel in enumerate(self.pads) if rel}}

    @classmethod
    def from_state(cls, state: dict) -> "Bank":
        bank = cls(state["name"])
        for number, rel in state["pads"].items():
            if 1 <= int(number) <= PADS:
                bank.pads[int(number) - 1] = rel
        return bank


def location_problems(rel: str, name: SampleName) -> list[str]:
    """Check that a parsed file sits in the folder its name says it should."""
    if rel != name.relpath:
        return [f"should be at {name.relpath}"]
    return []


def is_audio_file(filename: str) -> bool:
    return filename.rpartition(".")[2].lower() in naming.AUDIO_EXTENSIONS


class Library:
    def __init__(self, root: str | os.PathLike, *, read_only: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.read_only = read_only
        self.samples: list[Sample] = []
        self.by_rel: dict[str, Sample] = {}
        self.index_fields: list[str] = []
        self.index_rows: dict[str, dict] = {}
        self.favorites: set[str] = set()
        self.feature_cache: dict[str, dict] = {}
        self._feature_cache_loaded = False
        self.genres: dict[str, list[str]] = {}
        self.duplicates: dict[str, DuplicateInfo] = {}
        self.duplicates_checked = False     # whether a full check has run in this session
        self.similarity_scores: dict[str, float] | None = None
        self.last_upload: set[str] = set()
        self.banks: list[Bank] = []

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIR

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILE

    def scan(self) -> "Library":
        if not self.root.is_dir():
            raise FileNotFoundError(f"library folder not found: {self.root}")
        self._load_index()
        self._load_favorites()
        self._load_genres()
        self._load_banks()
        self._load_last_upload()
        if not self._feature_cache_loaded:
            self.feature_cache = fx.load_cache(self.state_dir)
            self._feature_cache_loaded = True
        samples = []
        for folder, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                if filename.startswith(".") or not is_audio_file(filename):
                    continue
                path = Path(folder) / filename
                samples.append(self._make_sample(path.relative_to(self.root).as_posix(), path))
        samples.sort(key=lambda s: s.rel)
        self.samples = samples
        self.by_rel = {s.rel: s for s in samples}
        self._tidy_duplicates()
        self.last_upload &= set(self.by_rel)
        for sample in samples:
            sample.duplicate = self.duplicates.get(sample.rel)
            sample.is_new = sample.rel in self.last_upload
        return self

    def _make_sample(self, rel: str, path: Path) -> Sample:
        filename = path.name
        name = naming.parse_name(filename)
        problems = location_problems(rel, name) if name else naming.check_name(filename)
        sample = Sample(rel, path, name, problems, self.index_rows.get(rel), rel in self.favorites)
        sample.genres = tuple(self.genres.get(rel, ()))
        sample.haystack = _haystack(sample)
        sample.features = self._cached_features(rel, path)
        return sample

    def _cached_features(self, rel: str, path: Path) -> fx.Features | None:
        entry = self.feature_cache.get(rel)
        if entry is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        if entry.get("size") != stat.st_size or entry.get("mtime") != stat.st_mtime_ns:
            return None
        return fx.from_entry(entry)

    def rename_features(self, src: str, dst: str) -> None:
        """A rename doesn't change the audio, so its measurements can follow it."""
        entry = self.feature_cache.pop(src, None)
        if entry is not None:
            self.feature_cache[dst] = entry

    def attach_features(self) -> None:
        """Give samples their newly cached measurements, without replacing the sample objects."""
        for sample in self.samples:
            if sample.features is None:
                sample.features = self._cached_features(sample.rel, sample.path)

    def write_features(self) -> None:
        if not self.read_only and self.feature_cache:
            fx.save_cache(self.state_dir, self.feature_cache)

    # ------------------------------------------------------------------ index

    def _load_index(self) -> None:
        self.index_fields, self.index_rows = [], {}
        if not self.index_path.exists():
            return
        with open(self.index_path, newline="") as fh:
            reader = csv.DictReader(fh)
            self.index_fields = list(reader.fieldnames or [])
            self.index_rows = {row["path"]: row for row in reader if row.get("path")}

    def write_index(self) -> None:
        """Atomically rewrite _index.csv. Does nothing if the library has no index."""
        if not self.index_fields:
            return
        tmp = self.index_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.index_fields, extrasaction="ignore")
            writer.writeheader()
            for rel in sorted(self.index_rows):
                writer.writerow(self.index_rows[rel])
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------------ favorites

    def _load_favorites(self) -> None:
        path = self.state_dir / FAVORITES_FILE
        try:
            self.favorites = set(json.loads(path.read_text()))
        except (OSError, ValueError):
            self.favorites = set()

    def write_favorites(self) -> None:
        self._write_state(FAVORITES_FILE, sorted(self.favorites))

    def toggle_favorite(self, sample: Sample) -> bool:
        self._require_writable()
        sample.favorite = not sample.favorite
        if sample.favorite:
            self.favorites.add(sample.rel)
        else:
            self.favorites.discard(sample.rel)
        self.write_favorites()
        return sample.favorite

    # ------------------------------------------------------------------ the most recent upload

    def _load_last_upload(self) -> None:
        try:
            self.last_upload = set(json.loads((self.state_dir / LAST_UPLOAD_FILE).read_text()))
        except (OSError, ValueError, TypeError):
            self.last_upload = set()

    def write_last_upload(self) -> None:
        if self.last_upload or (self.state_dir / LAST_UPLOAD_FILE).exists():
            self._write_state(LAST_UPLOAD_FILE, sorted(self.last_upload))

    # ------------------------------------------------------------------ duplicate check results

    def rename_duplicate(self, src: str, dst: str) -> None:
        """Results follow a renamed file, so they stay right without running the check again."""
        renamed = {}
        for rel, info in self.duplicates.items():
            others = tuple(dst if other == src else other for other in info.others)
            renamed[dst if rel == src else rel] = replace(info, others=others)
        self.duplicates = renamed

    def _tidy_duplicates(self) -> None:
        """Keep results accurate as files leave: a lone survivor is no longer a duplicate, and every set keeps a keeper.

        Without the keeper rule, trashing the suggested keeper would leave is:copy listing every remaining
        copy as safe to remove.
        """
        tidy = {}
        for rel, info in self.duplicates.items():
            others = tuple(other for other in info.others if other in self.by_rel)
            if rel in self.by_rel and others:
                tidy[rel] = replace(info, others=others)
        for rel, info in list(tidy.items()):
            members = sorted((rel, *info.others))
            if not any(tidy[m].keeper for m in members if m in tidy):
                tidy[members[0]] = replace(tidy[members[0]], keeper=True)
        self.duplicates = tidy

    # ------------------------------------------------------------------ genres

    def _load_genres(self) -> None:
        path = self.state_dir / GENRES_FILE
        try:
            self.genres = {rel: list(tags) for rel, tags in json.loads(path.read_text()).items() if tags}
        except (OSError, ValueError, AttributeError):
            self.genres = {}

    def write_genres(self) -> None:
        if self.genres or (self.state_dir / GENRES_FILE).exists():
            self._write_state(GENRES_FILE, {rel: self.genres[rel] for rel in sorted(self.genres)})

    def set_sample_genres(self, sample: Sample, tags: list[str]) -> None:
        if tags:
            self.genres[sample.rel] = tags
        else:
            self.genres.pop(sample.rel, None)
        sample.genres = tuple(tags)
        sample.haystack = _haystack(sample)

    def all_genres(self) -> list[str]:
        """Every genre in use, sorted."""
        return sorted({tag for tags in self.genres.values() for tag in tags}, key=str.lower)

    # ------------------------------------------------------------------ banks

    def _load_banks(self) -> None:
        try:
            self.banks = [Bank.from_state(state) for state in json.loads((self.state_dir / BANKS_FILE).read_text())]
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            self.banks = []

    def write_banks(self) -> None:
        if self.banks or (self.state_dir / BANKS_FILE).exists():
            self._write_state(BANKS_FILE, [bank.to_state() for bank in self.banks])

    def bank_named(self, name: str) -> Bank | None:
        """Names are compared the way they'd be exported, since two banks can't share a folder."""
        wanted = naming.slugify(name)
        return next((bank for bank in self.banks if naming.slugify(bank.name) == wanted), None)

    def rename_in_banks(self, src: str, dst: str) -> None:
        """Pads follow a renamed file."""
        for bank in self.banks:
            bank.pads = [dst if rel == src else rel for rel in bank.pads]

    def restore_bank(self, name: str | None, before: dict | None, index: int | None = None) -> None:
        """Reverse one bank change. `name` is what it's called now, and `before` is None if it didn't exist then."""
        current = self.bank_named(name) if name else None
        if before is None:
            self.banks = [bank for bank in self.banks if bank is not current]
        elif current is not None:
            self.banks[self.banks.index(current)] = Bank.from_state(before)
        else:
            self.banks.insert(len(self.banks) if index is None else index, Bank.from_state(before))

    def pads_holding(self, rel: str) -> list[tuple[str, int]]:
        """Every (bank name, pad) a sample is on."""
        return [(bank.name, pad) for bank in self.banks for pad, held in enumerate(bank.pads) if held == rel]

    # ------------------------------------------------------------------ preferences

    def load_prefs(self) -> dict:
        try:
            return json.loads((self.state_dir / PREFS_FILE).read_text())
        except (OSError, ValueError):
            return {}

    def save_prefs(self, prefs: dict) -> None:
        if not self.read_only:
            self._write_state(PREFS_FILE, prefs)

    def _write_state(self, filename: str, data) -> None:
        self.state_dir.mkdir(exist_ok=True)
        path = self.state_dir / filename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, path)

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("library was opened read-only")


def _haystack(sample: Sample) -> str:
    """Everything a free-text search looks at, including the pre-sort filename."""
    parts = [sample.main, sample.sub, sample.type or "", sample.key or "", sample.desc, *sample.genres]
    if sample.index:
        parts.append(sample.index.get("original_name", ""))
    return " ".join(parts).lower()


# ---------------------------------------------------------------------- sorting

SORT_KEYS = {
    "main": lambda s: s.main.lower(),
    "sub": lambda s: s.sub.lower(),
    "type": lambda s: s.type,
    "bpm": lambda s: s.bpm,
    "key": lambda s: _key_order(s.key),
    "len": lambda s: s.length,
    "genres": lambda s: s.genres[0].lower() if s.genres else None,
    "desc": lambda s: s.desc,
}


def _key_order(key: str | None):
    if not key:
        return None
    return (naming.NOTES.index(naming.key_root(key)), naming.key_mode(key) or "")


def sort_samples(samples: list[Sample], column: str = "main", reverse: bool = False) -> list[Sample]:
    """Sort by a column. Samples with no value stay last in both directions, whichever way you sort."""
    getter = SORT_KEYS[column]
    known, missing = [], []
    for sample in samples:
        (missing if getter(sample) is None else known).append(sample)
    known.sort(key=lambda s: s.rel)
    known.sort(key=getter, reverse=reverse)
    missing.sort(key=lambda s: s.rel)
    return known + missing


def parse_sort_spec(spec: str) -> tuple[str, bool]:
    """'bpm' sorts ascending and '-bpm' descending."""
    reverse = spec.startswith("-")
    column = spec.lstrip("-").lower()
    aliases = {"length": "len", "description": "desc", "genre": "genres"}
    column = aliases.get(column, column)
    if column not in SORT_KEYS:
        raise ValueError(f"cannot sort by '{column}'; choose from {', '.join(SORT_KEYS)}")
    return column, reverse


# ---------------------------------------------------------------------- facets

def folder_counts(samples: list[Sample]) -> Counter:
    """Count samples at each folder level, for the browse tree: 'Drums', 'Drums/Loops', 'Drums/Loops/Kick'."""
    counts: Counter = Counter()
    for sample in samples:
        parts = sample.rel.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            counts["/".join(parts[:depth])] += 1
    return counts
