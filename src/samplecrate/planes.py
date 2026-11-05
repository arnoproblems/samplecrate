"""The two sound maps and the search behind the Finder. No UI here.

Character places a sample relative to others in its own category, by rank. It answers
"which kick do I want?", and a dot near the top means "the brightest of your kicks".

Footprint uses one fixed scale for everything: audible length against Low, Mid or High.
It answers "where does this sit, and what else is in that space?".
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from samplecrate import features as fx
from samplecrate import naming
from samplecrate.library import Sample

MIN_FINDER_SAMPLES = 5
CLOSE, FAIR, FAR = "Close", "Fair", "Far"
CLOSE_WITHIN = 0.10
FAIR_WITHIN = 0.20
SUSTAINED_SUBS = {"Pad", "Atmosphere"}

MIN_SECONDS, MAX_SECONDS = 0.03, 100.0
HIT_SECONDS, NOTE_SECONDS = 0.5, 4.0
BAND_LIMITS = ((30.0, fx.LOW_MAX_HZ), (fx.LOW_MAX_HZ, fx.HIGH_MIN_HZ), (fx.HIGH_MIN_HZ, 16000.0))
BAND_NAMES = (fx.LOW, fx.MID, fx.HIGH)

Point = tuple[float, float]
Category = tuple[str, str, str]

_HATS = ("Soft", "Ticky", "Washy", "Sizzly")
_DRUM_LOOPS = ("Heavy and open", "Airy", "Rolling", "Skittery")
# (dark + low end of the across axis, bright + low, dark + high, bright + high)
CORNER_WORDS: dict[Category, tuple[str, str, str, str]] = {
    ("Drums", naming.ONESHOT, "Kick"): ("Thuddy", "Clicky", "Boomy", "Ringing"),
    ("Drums", naming.ONESHOT, "Snare"): ("Woody", "Snappy", "Fat", "Splashy"),
    ("Drums", naming.ONESHOT, "Hat"): _HATS,
    ("Drums", naming.ONESHOT, "Hat-Open"): _HATS,
    ("Drums", naming.ONESHOT, "Hat-Closed"): _HATS,
    ("Bass", naming.ONESHOT, "808"): ("Thump", "Knock", "Deep", "Growly"),
    ("Drums", naming.LOOP, "Full"): _DRUM_LOOPS,
    ("Drums", naming.LOOP, "Top"): _DRUM_LOOPS,
    ("Drums", naming.LOOP, "Break"): _DRUM_LOOPS,
    ("Drums", naming.LOOP, "Stripped"): _DRUM_LOOPS,
    ("Drums", naming.LOOP, "Fill"): _DRUM_LOOPS,
}


def category_of(sample: Sample) -> Category | None:
    if sample.name is None:
        return None
    return sample.main, sample.type, sample.sub


def category_label(category: Category) -> str:
    _, type_, sub = category
    return f"{sub} {'loops' if type_ == naming.LOOP else 'shots'}"


@dataclass(frozen=True)
class PlaneSpec:
    kind: str                   # "character" or "footprint"
    title: str
    x_low: str = ""
    x_high: str = ""
    y_low: str = ""
    y_high: str = ""
    corners: tuple[str, str, str, str] | None = None
    category: Category | None = None


def _across_axis(category: Category) -> tuple[str, str, str]:
    """(measure, low label, high label) for a category's horizontal axis."""
    _, type_, sub = category
    if sub in SUSTAINED_SUBS:
        return "movement", "Static", "Moving"
    if type_ == naming.LOOP:
        return "busy", "Sparse", "Busy"
    return "decay", "Tight", "Long"


def character_spec(category: Category) -> PlaneSpec:
    _, low, high = _across_axis(category)
    return PlaneSpec("character", category_label(category), low, high, "Dark", "Bright",
                     CORNER_WORDS.get(category), category)


def footprint_spec() -> PlaneSpec:
    return PlaneSpec("footprint", "Footprint", "shorter", "longer", fx.LOW, fx.HIGH)


def percentiles(values: np.ndarray) -> np.ndarray:
    """Each value's position among the others, 0 to 1. Tied values share a position, rather than being spread by chance."""
    if len(values) == 1:
        return np.array([0.5])
    ordered = np.sort(values)
    below = np.searchsorted(ordered, values, side="left")
    through = np.searchsorted(ordered, values, side="right") - 1
    return (below + through) / 2 / (len(values) - 1)


def footprint_position(features: fx.Features) -> Point:
    seconds = min(max(features.audible, MIN_SECONDS), MAX_SECONDS)
    x = (math.log(seconds) - math.log(MIN_SECONDS)) / (math.log(MAX_SECONDS) - math.log(MIN_SECONDS))
    return x, frequency_position(features.center_hz)


def frequency_position(hz: float) -> float:
    """Height on the Footprint plane. The three bands get equal space, with true log position inside each."""
    band = BAND_NAMES.index(fx.range_of(hz))
    low, high = BAND_LIMITS[band]
    within = (math.log(min(max(hz, low), high)) - math.log(low)) / (math.log(high) - math.log(low))
    return (band + within) / 3


class SoundSpace:
    """Every measured sample's position on both maps. Rebuild it when the library or its measurements change."""

    def __init__(self, samples: list[Sample]):
        self.footprint: dict[str, Point] = {}
        self.character: dict[str, Point] = {}
        self.categories: dict[Category, list[Sample]] = {}
        groups: dict[Category, list[Sample]] = defaultdict(list)
        for sample in samples:
            if sample.features is None:
                continue
            self.footprint[sample.rel] = footprint_position(sample.features)
            category = category_of(sample)
            if category:
                groups[category].append(sample)
        for category, group in groups.items():
            self.categories[category] = group
            measure = _across_axis(category)[0]
            across = percentiles(np.array([getattr(s.features, measure) for s in group]))
            up = percentiles(np.array([s.features.center_hz for s in group]))
            for sample, x, y in zip(group, across, up):
                self.character[sample.rel] = (float(x), float(y))

    def position(self, kind: str, sample: Sample) -> Point | None:
        return (self.character if kind == "character" else self.footprint).get(sample.rel)

    def finder_categories(self) -> list[tuple[Category, int]]:
        """Categories big enough for a map to mean something, with their sizes."""
        found = [(c, len(g)) for c, g in self.categories.items() if len(g) >= MIN_FINDER_SAMPLES]
        return sorted(found, key=lambda item: item[0])


_COMPARATIVES = {"Long": "longer", "Tight": "tighter", "Busy": "busier", "Sparse": "sparser",
                 "Moving": "more moving", "Static": "more static"}


def corner_word(spec: PlaneSpec, point: Point) -> str | None:
    if not spec.corners:
        return None
    x, y = point
    return spec.corners[(1 if y >= 0.5 else 0) + (2 if x >= 0.5 else 0)]


def describe_character(spec: PlaneSpec, point: Point) -> str:
    """Plain words for a position, such as: Boomy: longer than 71% of your Kick shots, darker than 64%."""
    x, y = point
    across = (f"{_COMPARATIVES[spec.x_high]} than {round(x * 100)}%" if x >= 0.5
              else f"{_COMPARATIVES[spec.x_low]} than {round((1 - x) * 100)}%")
    up = f"brighter than {round(y * 100)}%" if y >= 0.5 else f"darker than {round((1 - y) * 100)}%"
    text = f"{across} of your {spec.title}, {up}"
    word = corner_word(spec, point)
    return f"{word}: {text}" if word else text[0].upper() + text[1:]


def describe_footprint(sample: Sample) -> str:
    features = sample.features
    text = f"{features.range} range · {features.audible:.2f} s audible"
    if sample.length and features.audible < sample.length * 0.75:
        text += f" (file is {sample.length:.2f} s)"
    return text


# ---------------------------------------------------------------------- the Finder

def closeness(distance: float) -> str:
    if distance <= CLOSE_WITHIN:
        return CLOSE
    return FAIR if distance <= FAIR_WITHIN else FAR


@dataclass
class Match:
    sample: Sample
    position: Point
    distance: float
    hidden: int = 0         # near-duplicates of this sample left out of the list

    @property
    def closeness(self) -> str:
        return closeness(self.distance)


def nearest(space: SoundSpace, kind: str, pool: list[Sample], dot: Point, count: int) -> list[Match]:
    """The samples closest to a dot, skipping any that are effectively copies of a closer result."""
    ranked = []
    for sample in pool:
        position = space.position(kind, sample)
        if position is not None:
            ranked.append((math.hypot(position[0] - dot[0], position[1] - dot[1]), sample.rel, sample, position))
    ranked.sort(key=lambda item: item[:2])
    matches: list[Match] = []
    for distance, _, sample, position in ranked:
        original = next((m for m in matches if fx.are_near_duplicates(m.sample.features, sample.features)), None)
        if original is not None:
            original.hidden += 1
        elif len(matches) < count:
            matches.append(Match(sample, position, distance))
        else:
            break
    return matches


def summarize_matches(matches: list[Match], asked: int) -> str:
    if not matches:
        return "No samples to search. Try a different category or filter."
    close = sum(m.closeness == CLOSE for m in matches)
    hidden = sum(m.hidden for m in matches)
    if matches[0].closeness == FAR:
        text = "Nothing in here is close to that spot."
    elif close == len(matches):
        text = f"{close} close match{'es' if close != 1 else ''}."
    elif close:
        text = f"{close} of {len(matches)} {'are' if close != 1 else 'is'} close."
    else:
        text = "No close matches. These are the nearest."
    if len(matches) < asked:
        text += f" Only {len(matches)} to choose from."
    if hidden:
        text += f" {hidden} near-duplicate{'s' if hidden != 1 else ''} hidden."
    return text
