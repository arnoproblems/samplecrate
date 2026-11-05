"""The filter language shared by the TUI search box and `crate ls`.

A query is a space-separated list of terms, all of which must match:

    kick dark                   free text
    -clap                       negation (works on any term)
    main:drums  sub:kick        category, by prefix; sub:=Hat for an exact match
    type:loop   type:shot
    bpm:120  bpm:120-130  bpm:>140  bpm:~120 (tempo-compatible)  bpm:na
    key:Am  key:Am,Cmaj  key:~Am (harmonically compatible)  key:8A  key:na
    len:<2  len:1-4  len:>10    seconds, of the file
    audible:<1                  seconds of actual sound
    range:low  range:mid  range:high
    desc:vinyl                  substring of the description
    genre:house  genre:house,techno (either)  genre:na (untagged)
    in:Drums/Loops/Kick         folder, including everything below it
    is:fav  is:marked  is:invalid
    is:dupe (every copy)  is:copy (all but the one to keep)
    is:new                      from the most recent upload

The search box and the browse tree edit one query string, so what you see in the
box is always exactly what is being applied.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

from samplecrate import features as fx
from samplecrate import genres as genre_tags
from samplecrate import naming
from samplecrate.library import Sample

Predicate = Callable[[Sample], bool]

TEMPO_TOLERANCE = 0.03
_NUMBER = r"\d+(?:\.\d+)?"
_RANGE_RE = re.compile(rf"({_NUMBER})\s*-\s*({_NUMBER})")
_COMPARE_RE = re.compile(rf"(>=|<=|>|<|=)?\s*({_NUMBER})s?")
_BARE_BPM_RE = re.compile(r"(\d{2,3})bpm", re.IGNORECASE)
_TOKEN_RE = re.compile(r"""(?:[^\s"]+(?:"[^"]*"?)?|"[^"]*"?)+""")


@dataclass
class Query:
    text: str = ""
    predicates: list[Predicate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def matches(self, sample: Sample) -> bool:
        return all(p(sample) for p in self.predicates)

    def filter(self, samples: list[Sample]) -> list[Sample]:
        if not self.predicates:
            return list(samples)
        return [s for s in samples if self.matches(s)]


def parse(text: str, marked: Collection[str] = ()) -> Query:
    """Parse a query. Bad terms are reported in Query.errors and otherwise ignored, so a half-typed filter is harmless."""
    query = Query(text=text)
    for token in tokenize(text):
        negate = token.startswith("-") and len(token) > 1
        body = token[1:] if negate else token
        try:
            predicate = _term(body, marked)
        except ValueError as exc:
            query.errors.append(str(exc))
            continue
        query.predicates.append((lambda p: lambda s: not p(s))(predicate) if negate else predicate)
    return query


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def needs_duplicates(text: str) -> bool:
    """Whether a query depends on the results of a duplicate check."""
    return any(t.lstrip("-").lower() in ("is:dupe", "is:duplicate", "is:copy") for t in tokenize(text))


def needs_features(text: str) -> bool:
    """Whether a query depends on measurements, not just on filenames."""
    return any(t.lstrip("-").lower().startswith(("range:", "audible:")) for t in tokenize(text))


def _term(token: str, marked: Collection[str]) -> Predicate:
    field_name, sep, raw = token.partition(":")
    if sep:
        handler = _FIELDS.get(field_name.lower())
        if handler is None:
            raise ValueError(f"unknown filter '{field_name}:'")
        value = raw.strip('"')
        if not value:
            raise ValueError(f"'{field_name}:' needs a value")
        return handler(value, marked) if field_name.lower() == "is" else handler(value)
    bare = _BARE_BPM_RE.fullmatch(token)
    if bare:
        return _bpm_term(bare[1])
    word = token.strip('"').lower()
    return lambda s: word in s.haystack


def _text_prefix(getter: Callable[[Sample], str]) -> Callable[[str], Predicate]:
    def build(value: str) -> Predicate:
        options = [v.lower() for v in value.split(",") if v]
        if value.startswith("="):
            exact = [o.lstrip("=") for o in options]
            return lambda s: getter(s).lower() in exact
        return lambda s: any(getter(s).lower().startswith(o) for o in options)
    return build


def _type_term(value: str) -> Predicate:
    try:
        wanted = naming.normalize_type(value)
    except ValueError as exc:
        raise ValueError(str(exc)) from None
    return lambda s: s.type == wanted


def _numeric(value: str, label: str, tolerance: float = 0.0) -> Callable[[float], bool]:
    """Build a test from 120, =120, >120, >=120, <120, <=120 or 120-130."""
    text = value.strip().lower()
    m = _RANGE_RE.fullmatch(text)
    if m:
        low, high = sorted((float(m[1]), float(m[2])))
        return lambda v: low <= v <= high
    m = _COMPARE_RE.fullmatch(text)
    if not m:
        raise ValueError(f"bad {label} '{value}': use a number or a range such as 120-130, >120 or <120")
    target = float(m[2])
    op = m[1] or "="
    return {
        ">": lambda v: v > target,
        ">=": lambda v: v >= target,
        "<": lambda v: v < target,
        "<=": lambda v: v <= target,
        "=": lambda v: abs(v - target) <= tolerance,
    }[op]


def tempo_compatible(sample_bpm: int, target: float) -> bool:
    """Within 3%, counting half and double time."""
    return any(abs(sample_bpm * factor - target) <= target * TEMPO_TOLERANCE for factor in (1, 2, 0.5))


def _bpm_term(value: str) -> Predicate:
    text = value.strip().lower()
    if text in ("na", "none"):
        return lambda s: s.bpm is None
    if text.startswith("~"):
        try:
            target = float(text[1:])
        except ValueError:
            raise ValueError(f"bad bpm '{value}': use ~120") from None
        return lambda s: s.bpm is not None and tempo_compatible(s.bpm, target)
    test = _numeric(text, "bpm", tolerance=0.5)
    return lambda s: s.bpm is not None and test(s.bpm)


def _len_term(value: str) -> Predicate:
    test = _numeric(value, "len", tolerance=0.005)
    return lambda s: s.length is not None and test(s.length)


def _key_term(value: str) -> Predicate:
    text = value.strip()
    if text.lower() in ("na", "none"):
        return lambda s: s.key is None
    if text.startswith("~"):
        allowed = naming.compatible_keys(_parse_key(text[1:]))
        return lambda s: s.key is not None and key_in(s.key, allowed)
    options = [_parse_key(part) for part in text.split(",") if part]
    return lambda s: s.key is not None and any(_key_matches(s.key, o) for o in options)


def _parse_key(text: str) -> str:
    try:
        key = naming.normalize_key(text)
    except ValueError:
        raise ValueError(f"bad key '{text}': try Am, F#min, Bbmaj or 8A") from None
    if key is None:
        raise ValueError(f"bad key '{text}'")
    return key


def _key_matches(sample_key: str, wanted: str) -> bool:
    """key:F matches any F. key:Fmin matches Fmin and a bare F, whose mode nobody stated."""
    if naming.key_mode(wanted) is None:
        return naming.key_root(sample_key) == wanted
    return sample_key == wanted or sample_key == naming.key_root(wanted)


def key_in(sample_key: str, allowed: set[str]) -> bool:
    if naming.key_mode(sample_key):
        return sample_key in allowed
    return f"{sample_key}min" in allowed or f"{sample_key}maj" in allowed


def _audible_term(value: str) -> Predicate:
    test = _numeric(value, "audible", tolerance=0.005)
    return lambda s: s.features is not None and test(s.features.audible)


def _range_term(value: str) -> Predicate:
    wanted = value.strip().lower()
    options = {name.lower(): name for name in (fx.LOW, fx.MID, fx.HIGH)}
    match = [name for key, name in options.items() if key.startswith(wanted)]
    if not wanted or len(match) != 1:
        raise ValueError(f"bad range '{value}': use range:low, range:mid or range:high")
    return lambda s: s.features is not None and s.features.range == match[0]


def _genre_term(value: str) -> Predicate:
    text = value.strip()
    if text.lower() in ("na", "none"):
        return lambda s: not s.genres
    wanted = [genre_tags.search_key(part) for part in text.split(",") if genre_tags.key(part)]
    if not wanted:
        raise ValueError(f"bad genre '{value}'")
    return lambda s: any(genre_tags.key(tag).startswith(w) for tag in s.genres for w in wanted)


def _desc_term(value: str) -> Predicate:
    word = value.lower()
    return lambda s: word in s.desc.lower()


def _in_term(value: str) -> Predicate:
    folder = value.strip("/")
    prefix = folder + "/"
    return lambda s: s.rel.startswith(prefix)


def _is_term(value: str, marked: Collection[str]) -> Predicate:
    flag = value.lower()
    if flag in ("fav", "favorite", "star"):
        return lambda s: s.favorite
    if flag == "marked":
        return lambda s: s.rel in marked
    if flag in ("invalid", "unsorted"):
        return lambda s: bool(s.problems)
    if flag in ("dupe", "duplicate"):
        return lambda s: s.duplicate is not None
    if flag == "copy":
        return lambda s: s.duplicate is not None and not s.duplicate.keeper
    if flag == "new":
        return lambda s: s.is_new
    raise ValueError(f"unknown 'is:{value}': use is:fav, is:marked, is:invalid, is:new, is:dupe or is:copy")


_FIELDS = {
    "main": _text_prefix(lambda s: s.main),
    "sub": _text_prefix(lambda s: s.sub),
    "type": _type_term,
    "bpm": _bpm_term,
    "key": _key_term,
    "len": _len_term,
    "length": _len_term,
    "desc": _desc_term,
    "audible": _audible_term,
    "range": _range_term,
    "genre": _genre_term,
    "in": _in_term,
    "is": _is_term,
}


# ---------------------------------------------------------------------- editing a query string

def set_term(text: str, field_name: str, value: str | None) -> str:
    """Replace a field's terms in a query string, or remove them when value is None. Other terms are kept."""
    prefix = field_name + ":"
    kept = [t for t in tokenize(text) if not t.lower().startswith(prefix)]
    if value is None:
        return " ".join(kept)
    return " ".join([f"{prefix}{value}", *kept])

