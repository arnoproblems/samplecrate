"""Finding samples you already have, in two tiers.

"Same audio" means the decoded sound is identical, even if the metadata or format differs.
"Likely the same" catches what an exact match can't, such as a re-encoded or level-changed copy.
Files that measure alike are confirmed by comparing their actual waveforms, because many different
sounds share a length and a pitch center.

Nothing here deletes or changes anything. It only reports.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from samplecrate import features as fx
from samplecrate import filing
from samplecrate.features import Features
from samplecrate.library import DuplicateInfo, Library, is_audio_file

SAME_AUDIO = "Same audio"
LIKELY = "Likely the same"
CENTER_WINDOW = 0.10            # a low-bitrate encode can shift a bright sound's center by this much
LENGTH_SLACK_SECONDS = 0.15     # an encoder can add a little silence
LENGTH_SLACK_SHARE = 0.03
SIGNATURE_TOLERANCE = 0.20      # deliberately loose: the waveform comparison makes the real decision
# Real copies at any normal quality score at least 0.97. Different kicks and 808s reach 0.969, being nearly pure sine waves.
MIN_CORRELATION = 0.97
SCORES_FILE = "similarity.json"
COMPARE_SECONDS = 20
MEASURE_WORKERS = 4


@dataclass(frozen=True, eq=False)
class Item:
    path: Path
    features: Features
    rel: str | None = None      # set for files inside the library
    rank: int = 0               # lower is kept first

    @property
    def in_library(self) -> bool:
        return self.rel is not None

    @property
    def label(self) -> str:
        return self.rel or str(self.path)


@dataclass
class Group:
    kind: str
    items: list[Item]           # the one to keep comes first

    def contains(self, item: Item) -> bool:
        return any(i is item for i in self.items)


class _Waves:
    """Decoded audio for comparing, kept while a check runs so each file is only read once."""

    def __init__(self):
        self._cache: dict[Path, tuple[np.ndarray, int]] = {}

    def get(self, path: Path) -> tuple[np.ndarray, int]:
        cached = self._cache.get(path)
        if cached is None:
            with sf.SoundFile(str(path)) as fh:
                data = fh.read(frames=fh.samplerate * COMPARE_SECONDS, dtype="float32", always_2d=True)
                cached = data.mean(axis=1), fh.samplerate
            self._cache[path] = cached
        return cached


def _read_compare(waves: _Waves, path: Path) -> tuple[np.ndarray, int]:
    try:
        return waves.get(path)
    except (RuntimeError, sf.LibsndfileError):
        mono, sr = fx._read_mono(path)
        return mono[:sr * COMPARE_SECONDS], sr


def waveform_similarity(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
    """How alike two sounds are as waveforms, from 0 to 1, at the best alignment and ignoring level."""
    if sr_a != sr_b:
        target = np.linspace(0, len(b) / sr_b, int(len(b) * sr_a / sr_b), endpoint=False)
        b = np.interp(target, np.arange(len(b)) / sr_b, b).astype("float32")
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-9:
        return 0.0
    size = len(a) + len(b)
    fft_size = 1 << math.ceil(math.log2(size))
    correlation = np.fft.irfft(np.fft.rfft(a, fft_size) * np.conj(np.fft.rfft(b, fft_size)), fft_size)
    return float(np.abs(correlation).max() / norm)


def might_match(a: Features, b: Features) -> bool:
    """A cheap test for whether two files are worth comparing properly."""
    if a.center_hz <= 0 or b.center_hz <= 0:
        return False
    if abs(a.center_hz - b.center_hz) > CENTER_WINDOW * max(a.center_hz, b.center_hz):
        return False
    if abs(a.audible - b.audible) > LENGTH_SLACK_SECONDS + LENGTH_SLACK_SHARE * max(a.audible, b.audible):
        return False
    distance = float(np.abs(np.array(a.signature) - np.array(b.signature)).mean())
    return distance <= SIGNATURE_TOLERANCE


def _candidate_pairs(items: list[Item], only: set[int] | None) -> Iterator[tuple[Item, Item]]:
    """Pairs with close pitch centers. Sorting by center means most of the library is never compared at all."""
    ordered = sorted((i for i in items if i.features.center_hz > 0), key=lambda i: i.features.center_hz)
    start = 0
    for index, item in enumerate(ordered):
        while ordered[start].features.center_hz < item.features.center_hz * (1 - CENTER_WINDOW):
            start += 1
        for other in ordered[start:index]:
            if only is not None and id(item) not in only and id(other) not in only:
                continue
            yield other, item


def _score_key(a: Features, b: Features) -> str:
    return "|".join(sorted((a.audio_hash, b.audio_hash)))


def find_groups(items: list[Item], *, only: Iterable[Item] | None = None, exact_only: bool = False,
                stop=None, min_correlation: float = MIN_CORRELATION, on_compare=None,
                scores: dict[str, float] | None = None) -> list[Group]:
    """Group duplicates. With `only`, likely matches are looked for only where one of those items is involved.

    `scores` remembers waveform comparisons by audio hash, so later scans don't repeat them.
    """
    parent = {id(i): id(i) for i in items}
    by_id = {id(i): i for i in items}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a: Item, b: Item) -> None:
        parent[find(id(a))] = find(id(b))

    hashes: dict[str, Item] = {}
    for item in items:
        code = item.features.audio_hash
        if code:
            if code in hashes:
                join(hashes[code], item)
            else:
                hashes[code] = item

    if not exact_only:
        waves = _Waves()
        focus = {id(i) for i in only} if only is not None else None
        for a, b in _candidate_pairs(items, focus):
            if stop is not None and stop():
                break
            if find(id(a)) == find(id(b)) or not might_match(a.features, b.features):
                continue
            key = _score_key(a.features, b.features)
            score = scores.get(key) if scores is not None else None
            if score is None:
                try:
                    score = waveform_similarity(*_read_compare(waves, a.path), *_read_compare(waves, b.path))
                except Exception:
                    continue
                if scores is not None:
                    scores[key] = round(score, 4)
            if on_compare:
                on_compare(a, b, score)
            if score >= min_correlation:
                join(a, b)

    members: dict[int, list[Item]] = defaultdict(list)
    for key in parent:
        members[find(key)].append(by_id[key])
    groups = []
    for group in members.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda i: (not i.in_library, i.rank, len(i.label), i.label))
        same = len({i.features.audio_hash for i in group}) == 1 and bool(group[0].features.audio_hash)
        groups.append(Group(SAME_AUDIO if same else LIKELY, group))
    groups.sort(key=lambda g: (g.kind != SAME_AUDIO, g.items[0].label))
    return groups


def library_items(library: Library) -> list[Item]:
    """Measured library samples as comparison items. Favorites and tagged samples are preferred as the one to keep."""
    items = []
    for sample in library.samples:
        if sample.features is not None:
            rank = (0 if sample.favorite else 2) - (1 if sample.genres else 0) + (5 if sample.problems else 0)
            items.append(Item(sample.path, sample.features, sample.rel, rank))
    return items


def scan_library(library: Library, *, only_rels: Iterable[str] | None = None, exact_only: bool = False, stop=None) -> list[Group]:
    items = library_items(library)
    only = None
    if only_rels is not None:
        wanted = set(only_rels)
        only = [i for i in items if i.rel in wanted]
    scores = load_scores(library)
    known = len(scores)
    groups = find_groups(items, only=only, exact_only=exact_only, stop=stop, scores=scores)
    if len(scores) != known:
        save_scores(library, scores)
    return groups


def load_scores(library: Library) -> dict[str, float]:
    if library.similarity_scores is None:
        try:
            library.similarity_scores = json.loads((library.state_dir / SCORES_FILE).read_text())
        except (OSError, ValueError):
            library.similarity_scores = {}
    return library.similarity_scores


def save_scores(library: Library, scores: dict[str, float]) -> None:
    if not library.read_only:
        library._write_state(SCORES_FILE, scores)


def flag_in_library(library: Library, groups: list[Group], *, merge: bool = False) -> None:
    """Record which samples have duplicates, so `is:dupe` and the browse tree can show them.

    A full check replaces earlier results. `merge` adds to them, for a check of only some files.
    """
    if not merge:
        library.duplicates = {}
        library.duplicates_checked = True
    for group in groups:
        rels = [i.rel for i in group.items if i.rel]
        if len(rels) < 2:
            continue
        for rel in rels:
            info = DuplicateInfo(group.kind, tuple(r for r in rels if r != rel), rel == rels[0])
            library.duplicates[rel] = info
    for sample in library.samples:
        sample.duplicate = library.duplicates.get(sample.rel)


# ---------------------------------------------------------------------- files outside the library

@dataclass
class Sources:
    files: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    origins: dict[Path, str] = field(default_factory=dict)     # where unpacked files really came from
    folders: dict[Path, str] = field(default_factory=dict)     # each file's folder within its pack, which often names the sound


@contextmanager
def expand_sources(paths: Iterable[str | os.PathLike]) -> Iterator[Sources]:
    """Turn files, folders and zips into a flat list of audio files. Zips are unpacked to a temporary folder."""
    with tempfile.TemporaryDirectory(prefix="crate-") as tmp:
        sources = Sources()
        for number, raw in enumerate(paths):
            path = Path(raw).expanduser()
            if not path.exists():
                sources.problems.append(f"not found: {path}")
            elif path.is_dir():
                walked = _walk_audio(path)
                sources.files.extend(walked)
                for file in walked:
                    sources.folders[file] = (Path(path.name) / file.parent.relative_to(path)).as_posix()
            elif path.suffix.lower() == ".zip":
                dest = Path(tmp) / str(number)
                try:
                    unpacked = _unpack(path, dest)
                except zipfile.BadZipFile:
                    sources.problems.append(f"not a readable zip: {path}")
                    continue
                sources.files.extend(unpacked)
                for file in unpacked:
                    sources.origins[file] = f"{path.name}::{file.relative_to(dest).as_posix()}"
                    sources.folders[file] = (Path(path.stem) / file.parent.relative_to(dest)).as_posix()
            elif is_audio_file(path.name):
                sources.files.append(path)
                sources.folders[path] = path.parent.name
            else:
                sources.problems.append(f"not an audio file: {path}")
        yield sources


def _walk_audio(root: Path) -> list[Path]:
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        found.extend(Path(folder) / f for f in files if not f.startswith(".") and is_audio_file(f))
    return sorted(found)


def _unpack(archive: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith(".") or not is_audio_file(name):
                continue
            zf.extract(info, dest)     # extract() strips absolute paths and ".." components
    return _walk_audio(dest)


def measure_files(paths: list[Path], *, on_progress=None) -> tuple[list[Item], list[Path]]:
    """Measure files outside the library. Returns (items, files that couldn't be read)."""
    items: list[Item] = []
    failed: list[Path] = []
    done = [0]

    def work(path: Path) -> None:
        try:
            # Between two copies in an upload, keep the one whose name says what it is
            named = filing.classify_name(filing.norm(path.name))[0] is not None
            items.append(Item(path, fx.measure(path), rank=0 if named else 1))
        except Exception:
            failed.append(path)
        done[0] += 1
        if on_progress:
            on_progress(done[0], len(paths))

    with ThreadPoolExecutor(MEASURE_WORKERS) as pool:
        list(pool.map(work, paths))
    items.sort(key=lambda i: str(i.path))
    return items, failed


@dataclass
class UploadCheck:
    """What would happen if these files were added to the library."""
    new: list[Item] = field(default_factory=list)
    duplicates: list[tuple[Item, str, Item]] = field(default_factory=list)     # (file, kind, what it matches)
    failed: list[Path] = field(default_factory=list)


def check_upload(library: Library, items: list[Item], failed: list[Path] | None = None) -> UploadCheck:
    """Compare incoming files with the library and with each other. The first of any incoming copies counts as new."""
    scores = load_scores(library)
    known = len(scores)
    groups = find_groups(library_items(library) + items, only=items, scores=scores)
    if len(scores) != known:
        save_scores(library, scores)
    matched: dict[int, tuple[str, Item]] = {}
    for group in groups:
        keeper = group.items[0]
        for item in group.items[1:]:
            if not item.in_library:
                # A group can mix tiers, so say what this file itself matches, if anything matches exactly
                exact = next((o for o in group.items if o is not item and o.features.audio_hash == item.features.audio_hash
                              and (o.in_library or group.items.index(o) < group.items.index(item))), None)
                matched[id(item)] = (SAME_AUDIO, exact) if exact else (LIKELY, keeper)
    result = UploadCheck(failed=list(failed or []))
    for item in items:
        hit = matched.get(id(item))
        if hit:
            result.duplicates.append((item, hit[0], hit[1]))
        else:
            result.new.append(item)
    return result


def describe_upload(check: UploadCheck, added: int | None = None, shown: int = 40, proposals: dict | None = None) -> list[str]:
    """A plain report on an upload. Pass `added` once the files have been copied."""
    lines = [f"{'Added' if added is not None else 'New'}: {len(check.new) if added is None else added}"]
    held = []
    if proposals is not None:
        filed = [proposals[i.path].name for i in check.new if proposals[i.path].name]
        held = [i for i in check.new if not proposals[i.path].name]
        lines.append(f"  {'filed' if added is not None else 'would be filed'} automatically: {len(filed)}")
        lines.append(f"  need you to file them: {len(held)}")
    repeats = sum(not match.in_library for _, _, match in check.duplicates)
    lines.append(f"Already in your library: {len(check.duplicates) - repeats}")
    if repeats:
        lines.append(f"Repeated within this upload: {repeats}")
    if check.failed:
        lines.append(f"Couldn't read: {len(check.failed)}")
    if proposals is not None and filed:
        counts = Counter(f"{n.main}/{n.sub} {'loops' if n.type == 'Loop' else 'shots'}" for n in filed)
        lines.append("")
        lines.extend(f"{count:>5}  {place}" for place, count in counts.most_common())
    if held:
        lines.append("")
        lines.append("Need you:")
        for item in held[:shown]:
            lines.append(f"  {item.path.name}: {proposals[item.path].why}")
        if len(held) > shown:
            lines.append(f"  ...and {len(held) - shown} more")
    if check.duplicates:
        lines.append("")
        for item, kind, match in check.duplicates[:shown]:
            lines.append(f"{item.path.name}")
            lines.append(f"  {kind.lower()} as {match.label if match.in_library else match.path.name + ' in this upload'}")
        if len(check.duplicates) > shown:
            lines.append(f"...and {len(check.duplicates) - shown} more")
    for path in check.failed[:shown]:
        lines.append(f"Couldn't read {path.name}")
    return lines


def describe_groups(groups: list[Group]) -> list[str]:
    """A plain report, one block per group, with the suggested file to keep first."""
    lines: list[str] = []
    for group in groups:
        lines.append(f"{group.kind} ({len(group.items)} files)")
        for index, item in enumerate(group.items):
            lines.append(f"  {'keep' if index == 0 else 'copy'}  {item.label}")
        lines.append("")
    return lines
