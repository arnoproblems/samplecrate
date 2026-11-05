"""Reading a bank: whether its samples belong together, and what to say about them."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from samplecrate import naming, query
from samplecrate.library import Bank, Library, Sample

SHORT_SECONDS = 0.1     # Roland says files under 100 ms may not import
MISSING = "file missing"
TONELESS_MAIN = "Drums"     # a kick's note rarely decides whether it fits a key


@dataclass(frozen=True)
class BankReading:
    tempo: int | None
    key: str | None
    notes: dict[int, list[str]]     # by pad: what to watch out for


def is_short(sample: Sample) -> bool:
    return sample.length is not None and sample.length < SHORT_SECONDS


def _common_tempo(samples: list[Sample]) -> int | None:
    """The tempo most loops can play at, counting half and double time."""
    tempos = [s.bpm for s in samples if s.type == naming.LOOP and s.bpm]
    if not tempos:
        return None
    counts = Counter(tempos)
    return min(counts, key=lambda t: (-sum(query.tempo_compatible(other, t) for other in tempos), -counts[t], t))


def _common_key(samples: list[Sample]) -> str | None:
    """The key most pitched samples mix with. A bare note could be either mode, so it only leads if nothing's more exact."""
    keys = [s.key for s in samples if s.key and s.main != TONELESS_MAIN]
    if not keys:
        return None
    counts = Counter(keys)
    candidates = [k for k in counts if naming.key_mode(k)] or list(counts)
    return min(candidates, key=lambda c: (-sum(query.key_in(k, naming.compatible_keys(c)) for k in keys), -counts[c], c))


def read_bank(library: Library, bank: Bank) -> BankReading:
    samples = {pad: library.by_rel.get(rel) for pad, rel in enumerate(bank.pads) if rel}
    present = [s for s in samples.values() if s is not None]
    tempo, key = _common_tempo(present), _common_key(present)
    allowed = naming.compatible_keys(key) if key else set()
    notes: dict[int, list[str]] = {}
    for pad, sample in samples.items():
        found = []
        if sample is None:
            found.append(MISSING)
        else:
            if is_short(sample):
                found.append("under 100 ms")
            if tempo and sample.type == naming.LOOP and sample.bpm and not query.tempo_compatible(sample.bpm, tempo):
                found.append(f"≠ {tempo} bpm")
            if key and sample.key and sample.main != TONELESS_MAIN and not query.key_in(sample.key, allowed):
                found.append(f"≠ {key}")
        if found:
            notes[pad] = found
    return BankReading(tempo, key, notes)


def _pads(pads: list[int]) -> str:
    numbers = [str(pad + 1) for pad in pads]
    text = numbers[0] if len(numbers) == 1 else ", ".join(numbers[:-1]) + f" and {numbers[-1]}"
    return f"Pad{'s' if len(pads) != 1 else ''} {text}"


def describe_export(result) -> list[str]:
    """What an export did, for the app's report and the command line."""
    verb = "Would copy" if result.dry_run else "Copied"
    lines = [f"{verb} {result.copied} sample{'s' if result.copied != 1 else ''} to {result.folder}, numbered by pad."]
    if result.skipped:
        lines.append(f"{result.skipped} {'were' if result.skipped != 1 else 'was'} already there.")
    if result.conversion_note:
        lines.append(result.conversion_note[0].upper() + result.conversion_note[1:] + ".")
    if result.replaces:
        lines.append(f"{result.replaces} holds an older layout of this bank, so it was left alone.")
    if result.short:
        lines.append(f"{_pads(result.short)} {'is' if len(result.short) == 1 else 'are'} under 100 ms and may not import.")
    if result.missing:
        lines.append(f"{_pads(result.missing)} {'was' if len(result.missing) == 1 else 'were'} left out because the file is gone.")
    lines += ["", "On the SP-404MK2: hold SHIFT and press pad 13, then choose IMPORT, IMPORT from SD-CARD, SAMPLE. "
                  "Press a pad, turn VALUE to the file with that pad's number, and press VALUE."]
    return lines
