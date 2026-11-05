"""The sample file naming convention: parsing, formatting and validation.

docs/NAMING.md is the human-readable spec. tests/test_docs.py checks that the
examples in that document still agree with this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

AUDIO_EXTENSIONS = {"wav", "aif", "aiff", "flac", "ogg", "mp3"}
FIELD_SEP = "__"
NA = "NA"
LOOP, ONESHOT = "Loop", "OneShot"
TYPE_FOLDERS = {LOOP: "Loops", ONESHOT: "One-Shots"}
NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_MAIN_RE = re.compile(r"[A-Za-z]+")
_SUB_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_BPM_RE = re.compile(r"(\d{3})bpm")
_KEY_RE = re.compile(r"([A-G]#?)(maj|min)?")
_LENGTH_RE = re.compile(r"(\d+\.\d{2})s")
_DESC_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
               "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
_CAMELOT_MINOR = {"A": 8, "E": 9, "B": 10, "F#": 11, "C#": 12, "G#": 1,
                  "D#": 2, "A#": 3, "F": 4, "C": 5, "G": 6, "D": 7}
_CAMELOT_MAJOR = {"C": 8, "G": 9, "D": 10, "A": 11, "E": 12, "B": 1,
                  "F#": 2, "C#": 3, "G#": 4, "D#": 5, "A#": 6, "F": 7}
_CAMELOT_LOOKUP = {**{f"{n}A": f"{k}min" for k, n in _CAMELOT_MINOR.items()},
                   **{f"{n}B": f"{k}maj" for k, n in _CAMELOT_MAJOR.items()}}


@dataclass(frozen=True, slots=True)
class SampleName:
    main: str
    sub: str
    type: str
    bpm: int | None
    key: str | None
    length: float
    desc: str
    ext: str

    @property
    def filename(self) -> str:
        bpm = f"{self.bpm:03d}bpm" if self.bpm else NA
        fields = (self.main, self.sub, self.type, bpm, self.key or NA, f"{self.length:.2f}s", self.desc)
        return f"{FIELD_SEP.join(fields)}.{self.ext}"

    @property
    def export_filename(self) -> str:
        """The name for a copy leaving the library, where an NA says nothing. parse_name can't read this form."""
        bpm = [f"{self.bpm:03d}bpm"] if self.bpm else []
        key = [self.key] if self.key else []
        fields = (self.main, self.sub, self.type, *bpm, *key, f"{self.length:.2f}s", self.desc)
        return f"{FIELD_SEP.join(fields)}.{self.ext}"

    @property
    def folder(self) -> str:
        return f"{self.main}/{TYPE_FOLDERS[self.type]}/{self.sub}"

    @property
    def relpath(self) -> str:
        return f"{self.folder}/{self.filename}"


def check_name(filename: str) -> list[str]:
    """Return every way a filename breaks the convention. An empty list means it conforms."""
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        return ["no file extension"]
    problems = []
    if ext not in AUDIO_EXTENSIONS:
        problems.append(f"extension '{ext}' must be one of {', '.join(sorted(AUDIO_EXTENSIONS))} (lowercase)")
    fields = stem.split(FIELD_SEP)
    if len(fields) != 7:
        problems.append(f"expected 7 fields separated by '{FIELD_SEP}', found {len(fields)}")
        return problems
    main, sub, type_, bpm, key, length, desc = fields
    if not _MAIN_RE.fullmatch(main):
        problems.append(f"main '{main}' must contain only letters")
    if not _SUB_RE.fullmatch(sub):
        problems.append(f"sub '{sub}' must be letters and digits, with single hyphens between words")
    if type_ not in TYPE_FOLDERS:
        problems.append(f"type '{type_}' must be {LOOP} or {ONESHOT}")
    if bpm != NA:
        m = _BPM_RE.fullmatch(bpm)
        if not m or int(m[1]) == 0:
            problems.append(f"bpm '{bpm}' must be three digits plus 'bpm' (080bpm), or {NA}")
    if key != NA and not _is_canonical_key(key):
        problems.append(f"key '{key}' must be a sharp-spelled note, optionally with maj or min (F#min), or {NA}")
    if not _LENGTH_RE.fullmatch(length):
        problems.append(f"length '{length}' must be seconds with two decimals (3.87s)")
    if not _DESC_RE.fullmatch(desc):
        problems.append(f"description '{desc}' must be lowercase letters and digits, with single hyphens between words")
    return problems


def _is_canonical_key(key: str) -> bool:
    m = _KEY_RE.fullmatch(key)
    return bool(m) and m[1] in NOTES


def parse_name(filename: str) -> SampleName | None:
    """Parse a conforming filename, or return None. Use check_name() to find out why it failed."""
    if check_name(filename):
        return None
    stem, _, ext = filename.rpartition(".")
    main, sub, type_, bpm, key, length, desc = stem.split(FIELD_SEP)
    return SampleName(
        main=main,
        sub=sub,
        type=type_,
        bpm=None if bpm == NA else int(bpm[:3]),
        key=None if key == NA else key,
        length=float(length[:-1]),
        desc=desc,
        ext=ext,
    )


def normalize_key(text: str) -> str | None:
    """Turn any common key spelling into the canonical form.

    Accepts Am, A minor, a#m, Bbmin, BbMaj, F#, f and Camelot codes such as 8A.
    Returns None for empty/NA/none. Raises ValueError for anything else.
    """
    raw = text.strip()
    if not raw or raw.lower() in ("na", "none", "-"):
        return None
    camelot_code = raw.upper()
    if camelot_code in _CAMELOT_LOOKUP:
        return _CAMELOT_LOOKUP[camelot_code]
    m = re.fullmatch(r"([A-Ga-g])([#b]?)[\s_-]*(major|minor|maj|min|m|M)?", raw, re.IGNORECASE)
    if not m:
        raise ValueError(f"not a key: {text!r}")
    note = m[1].upper() + m[2].lower()
    note = _ENHARMONIC.get(note, note)
    # A bare capital M means major (AM); anything else starting with m means minor
    suffix = m[3]
    if suffix is None:
        return note
    mode = "maj" if suffix == "M" or suffix.lower().startswith("maj") else "min"
    return note + mode


def key_root(key: str) -> str:
    return key[:-3] if key.endswith(("maj", "min")) else key


def key_mode(key: str) -> str | None:
    return key[-3:] if key.endswith(("maj", "min")) else None


def camelot(key: str | None) -> str | None:
    """Camelot wheel code for a full key. A bare note has no mode, so it has no code."""
    if not key or not key_mode(key):
        return None
    table, letter = (_CAMELOT_MINOR, "A") if key_mode(key) == "min" else (_CAMELOT_MAJOR, "B")
    return f"{table[key_root(key)]}{letter}"


def compatible_keys(key: str) -> set[str]:
    """Keys that mix harmonically with `key`: itself, its relative, and one step either way on the wheel."""
    code = camelot(key)
    if code is None:
        # Root only: treat as either mode
        return compatible_keys(key + "min") | compatible_keys(key + "maj")
    number, letter = int(code[:-1]), code[-1]
    other = "B" if letter == "A" else "A"
    neighbors = (f"{number}{letter}", f"{number}{other}",
                 f"{number % 12 + 1}{letter}", f"{(number - 2) % 12 + 1}{letter}")
    return {_CAMELOT_LOOKUP[c] for c in neighbors}


def normalize_bpm(text: str | int | float | None) -> int | None:
    """Accept 120, 120.4, '120bpm', '' or 'NA'. Raises ValueError outside 1-999."""
    if text is None:
        return None
    raw = str(text).strip().lower().removesuffix("bpm").strip()
    if raw in ("", "na", "none", "-"):
        return None
    try:
        value = round(float(raw))
    except ValueError:
        raise ValueError(f"not a BPM: {text!r}") from None
    if not 1 <= value <= 999:
        raise ValueError(f"BPM must be between 1 and 999: {text!r}")
    return value


def slugify(text: str) -> str:
    """Make free text safe for the description field."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalize_main(text: str) -> str:
    value = re.sub(r"[^A-Za-z]", "", text)
    if not value:
        raise ValueError("main category needs at least one letter")
    return value


def normalize_sub(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    if not value:
        raise ValueError("sub category needs at least one letter or digit")
    return value


def normalize_type(text: str) -> str:
    value = re.sub(r"[^a-z]", "", text.lower())
    # Any start of the word is enough, so a half-typed filter already works
    if value and "loops".startswith(value):
        return LOOP
    if value and ("oneshots".startswith(value) or "shots".startswith(value) or value == "os"):
        return ONESHOT
    raise ValueError(f"type must be {LOOP} or {ONESHOT}: {text!r}")


def format_length(seconds: float) -> str:
    return f"{seconds:.2f}s"


def with_version(name: SampleName, version: int) -> SampleName:
    """Return a copy with a -vN suffix, the convention's way of avoiding a filename collision."""
    base = re.sub(r"-v\d+$", "", name.desc)
    return SampleName(name.main, name.sub, name.type, name.bpm, name.key, name.length, f"{base}-v{version}", name.ext)
