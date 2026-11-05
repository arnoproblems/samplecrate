"""Genre tags. A sample can have any number of them.

They live in <library>/.crate/genres.json rather than in filenames: a tag edit shouldn't
rename a file, and a list of tags doesn't fit the convention's seven fixed fields.
"""

from __future__ import annotations

import re

VOCABULARY = (
    "Afrobeats", "Amapiano", "Ambient", "Boom Bap", "Cinematic", "Drum & Bass", "Dubstep", "Funk", "Grime",
    "Hip Hop", "House", "Hyperpop", "Jersey Club", "Jungle", "Lo-Fi", "Phonk", "Pop", "R&B", "Soul",
    "Techno", "Trance", "Trap", "UK Garage",
)
_ALIASES = {"dnb": "Drum & Bass", "drumandbass": "Drum & Bass", "drumnbass": "Drum & Bass", "ukg": "UK Garage",
            "garage": "UK Garage", "rnb": "R&B", "randb": "R&B", "afrobeat": "Afrobeats", "lofi": "Lo-Fi",
            "hiphop": "Hip Hop", "boombap": "Boom Bap"}

# Clues to a sample's genre in where it came from, for `crate genres --suggest`
_SUGGESTIONS = (
    (r"dubstep", "Dubstep"),
    (r"liquid and hard dnb|\bdnb\b|_dnb|drum.?n.?bass|_lhd_", "Drum & Bass"),
    (r"jungle|psj_", "Jungle"),
    (r"jersey|jersy", "Jersey Club"),
    (r"hip.?hop|old skool|boom.?bap", "Hip Hop"),
    (r"(?<![a-z])trap(?![a-z])", "Trap"),
    (r"lo.?fi", "Lo-Fi"),
    (r"techno", "Techno"),
    (r"(?<![a-z])house(?![a-z])|g-house", "House"),
    (r"(?<![a-z])uk.?g(?![a-z])", "UK Garage"),
    (r"afro", "Afrobeats"),
    (r"amapiano|logdrum", "Amapiano"),
    (r"phonk", "Phonk"),
    (r"hyperpop", "Hyperpop"),
    (r"(?<![a-z])grime(?![a-z])", "Grime"),
)


def key(name: str) -> str:
    """How genres are compared: House, house and HOUSE are the same tag, as are Lo-Fi and lofi."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_CANONICAL = {key(name): name for name in VOCABULARY}
_CANONICAL.update(_ALIASES)


def clean(name: str) -> str:
    """Normalize one genre name, using the standard spelling if it's a known genre."""
    tidy = re.sub(r"\s+", " ", name).strip()
    if not key(tidy):
        raise ValueError(f"not a genre: {name!r}")
    return _CANONICAL.get(key(tidy), tidy)


def parse_list(text: str) -> list[str]:
    """Split comma-separated text into clean, unique genres."""
    found: dict[str, str] = {}
    for part in text.split(","):
        if part.strip():
            name = clean(part)
            found.setdefault(key(name), name)
    return list(found.values())


def search_key(text: str) -> str:
    """The form to compare with when filtering, which also understands aliases such as dnb."""
    canonical = _CANONICAL.get(key(text))
    return key(canonical) if canonical else key(text)


def suggest(*sources: str) -> list[str]:
    """Genres implied by a sample's source paths and original filename."""
    text = " ".join(sources).lower()
    return [genre for pattern, genre in _SUGGESTIONS if re.search(pattern, text)]
