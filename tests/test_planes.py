from types import SimpleNamespace

import numpy as np
import pytest

from samplecrate import features as fx
from samplecrate import naming, planes
from samplecrate.library import Sample
from samplecrate.naming import parse_name


def features(center=1000.0, decay=0.2, busy=1.0, movement=0.5, audible=1.0, signature=None):
    sig = signature if signature is not None else tuple(float(center % 97 + i) for i in range(4))
    return fx.Features(center, center * 0.5, center * 2, audible, decay, busy, movement, sig)


def sample(sub="Kick", type_="OneShot", main="Drums", desc="x", **kw):
    bpm = "120bpm" if type_ == "Loop" else "NA"
    name = parse_name(f"{main}__{sub}__{type_}__{bpm}__NA__1.00s__{desc}.wav")
    s = Sample(name.relpath, None, name)
    s.features = features(**kw)
    return s


def kicks(n=10):
    return [sample(desc=f"k{i}", center=40.0 + i * 10, decay=0.1 + i * 0.05) for i in range(n)]


def test_percentiles_spread_evenly():
    assert planes.percentiles(np.array([3.0, 1.0, 2.0])).tolist() == [1.0, 0.0, 0.5]
    assert planes.percentiles(np.array([5.0])).tolist() == [0.5]


def test_tied_values_share_a_position():
    """Otherwise identical sounds would land in different places by chance."""
    result = planes.percentiles(np.array([1.0, 2.0, 2.0, 3.0]))
    assert result[1] == result[2] == 0.5


def test_across_axis_depends_on_the_kind_of_sound():
    assert planes.character_spec(("Drums", "OneShot", "Kick")).x_high == "Long"
    assert planes.character_spec(("Drums", "Loop", "Full")).x_high == "Busy"
    assert planes.character_spec(("Synths", "Loop", "Pad")).x_high == "Moving"
    assert planes.character_spec(("FX", "Loop", "Atmosphere")).x_low == "Static"


def test_corner_words_agreed_with_the_user():
    assert planes.character_spec(("Drums", "OneShot", "Kick")).corners == ("Thuddy", "Clicky", "Boomy", "Ringing")
    assert planes.character_spec(("Drums", "OneShot", "Snare")).corners == ("Woody", "Snappy", "Fat", "Splashy")
    assert planes.character_spec(("Drums", "OneShot", "Hat-Open")).corners == ("Soft", "Ticky", "Washy", "Sizzly")
    assert planes.character_spec(("Bass", "OneShot", "808")).corners == ("Thump", "Knock", "Deep", "Growly")
    assert planes.character_spec(("Drums", "Loop", "Full")).corners == ("Heavy and open", "Airy", "Rolling", "Skittery")
    assert planes.character_spec(("Synths", "Loop", "Lead")).corners is None


@pytest.mark.parametrize("point,word", [((0.1, 0.1), "Thuddy"), ((0.1, 0.9), "Clicky"), ((0.9, 0.1), "Boomy"), ((0.9, 0.9), "Ringing")])
def test_corner_word_for_each_quadrant(point, word):
    assert planes.corner_word(planes.character_spec(("Drums", "OneShot", "Kick")), point) == word


def test_character_positions_are_relative_to_the_category():
    # A bright kick is far darker than a dark hat, yet each tops its own category
    kick_low, kick_high = sample(desc="a", center=50), sample(desc="b", center=90)
    hat_low, hat_high = sample(sub="Hat", desc="c", center=5000), sample(sub="Hat", desc="d", center=9000)
    space = planes.SoundSpace([kick_low, kick_high, hat_low, hat_high])
    assert space.character[kick_high.rel][1] == 1.0 and space.character[hat_low.rel][1] == 0.0


def test_loops_rank_by_busyness_and_pads_by_movement():
    sparse, busy = sample(sub="Full", type_="Loop", desc="a", busy=1), sample(sub="Full", type_="Loop", desc="b", busy=9)
    still, moving = sample(main="Synths", sub="Pad", type_="Loop", desc="c", movement=0), sample(main="Synths", sub="Pad", type_="Loop", desc="d", movement=3)
    space = planes.SoundSpace([sparse, busy, still, moving])
    assert space.character[busy.rel][0] == 1.0 and space.character[sparse.rel][0] == 0.0
    assert space.character[moving.rel][0] == 1.0 and space.character[still.rel][0] == 0.0


def test_footprint_uses_one_scale_for_everything():
    kick, hat = sample(desc="a", center=60, audible=0.3), sample(sub="Hat", desc="b", center=8000, audible=0.1)
    space = planes.SoundSpace([kick, hat])
    assert space.footprint[kick.rel][1] < 1 / 3 < 2 / 3 < space.footprint[hat.rel][1]
    assert space.footprint[hat.rel][0] < space.footprint[kick.rel][0]


@pytest.mark.parametrize("hz,low,high", [(30, 0, 1 / 3), (249, 0, 1 / 3), (250, 1 / 3, 2 / 3), (3999, 1 / 3, 2 / 3), (4000, 2 / 3, 1), (16000, 2 / 3, 1)])
def test_frequency_bands_get_equal_space(hz, low, high):
    assert low <= planes.frequency_position(hz) <= high


def test_positions_stay_on_the_plane_for_extreme_values():
    for f in (features(center=1, audible=0), features(center=50000, audible=9999)):
        x, y = planes.footprint_position(f)
        assert 0 <= x <= 1 and 0 <= y <= 1


def test_unmeasured_samples_are_left_off_the_maps():
    s = sample()
    s.features = None
    space = planes.SoundSpace([s])
    assert space.position("footprint", s) is None and space.position("character", s) is None


def test_finder_skips_categories_too_small_to_map():
    samples = kicks(5) + [sample(sub="Tom", desc=f"t{i}") for i in range(4)]
    assert planes.SoundSpace(samples).finder_categories() == [(("Drums", "OneShot", "Kick"), 5)]


def test_describe_character():
    spec = planes.character_spec(("Drums", "OneShot", "Kick"))
    assert planes.describe_character(spec, (0.71, 0.36)) == "Boomy: longer than 71% of your Kick shots, darker than 64%"
    spec = planes.character_spec(("Synths", "Loop", "Pad"))
    assert planes.describe_character(spec, (0.2, 0.9)).startswith("More static than 80% of your Pad loops, brighter than 90%")
    spec = planes.character_spec(("Drums", "Loop", "Full"))
    assert "busier than 80%" in planes.describe_character(spec, (0.8, 0.5))


def test_describe_footprint_mentions_padding_only_when_it_matters():
    s = sample(audible=0.3)
    assert planes.describe_footprint(s) == "Mid range · 0.30 s audible (file is 1.00 s)"
    assert "file is" not in planes.describe_footprint(sample(audible=0.95))


# ---------------------------------------------------------------------- nearest

def test_nearest_returns_closest_in_order():
    samples = kicks()
    space = planes.SoundSpace(samples)
    matches = planes.nearest(space, "character", samples, (0.0, 0.0), 3)
    assert [m.sample.desc for m in matches] == ["k0", "k1", "k2"]
    assert matches[0].distance < matches[1].distance < matches[2].distance


@pytest.mark.parametrize("count", [3, 5, 10])
def test_nearest_honors_the_count(count):
    samples = kicks(12)
    assert len(planes.nearest(planes.SoundSpace(samples), "character", samples, (0.5, 0.5), count)) == count


def test_nearest_returns_what_exists_when_pool_is_small():
    samples = kicks(2)
    assert len(planes.nearest(planes.SoundSpace(samples), "character", samples, (0.5, 0.5), 10)) == 2


def test_nearest_searches_only_the_pool():
    samples = kicks()
    space = planes.SoundSpace(samples)
    matches = planes.nearest(space, "character", samples[5:], (0.0, 0.0), 2)
    assert [m.sample.desc for m in matches] == ["k5", "k6"]


def test_near_duplicates_dont_use_up_result_slots():
    shared = tuple(float(i) for i in range(64))
    a = sample(desc="copy1", center=100, decay=0.1, signature=shared)
    b = sample(desc="copy2", center=100.5, decay=0.1, signature=shared)
    others = [sample(desc=f"o{i}", center=200 + i * 50, decay=0.1 + i * 0.1, signature=tuple(float(i + 1000 * j) for j in range(64))) for i in range(6)]
    pool = [a, b, *others]
    matches = planes.nearest(planes.SoundSpace(pool), "character", pool, (0.0, 0.0), 3)
    descs = [m.sample.desc for m in matches]
    assert len(matches) == 3 and sum(d.startswith("copy") for d in descs) == 1
    assert matches[0].hidden == 1


@pytest.mark.parametrize("distance,label", [(0.0, planes.CLOSE), (0.1, planes.CLOSE), (0.11, planes.FAIR), (0.2, planes.FAIR), (0.21, planes.FAR)])
def test_closeness(distance, label):
    assert planes.closeness(distance) == label


def make_match(distance, hidden=0):
    return planes.Match(sample(), (0, 0), distance, hidden)


def test_summary_is_honest_about_empty_spots():
    assert "Nothing in here is close" in planes.summarize_matches([make_match(0.5)], 3)
    assert planes.summarize_matches([make_match(0.05), make_match(0.06), make_match(0.07)], 3) == "3 close matches."
    assert "No close matches" in planes.summarize_matches([make_match(0.15), make_match(0.3)], 2)
    assert "1 of 2 is close" in planes.summarize_matches([make_match(0.05), make_match(0.3)], 2)
    assert "Only 1 to choose from" in planes.summarize_matches([make_match(0.05)], 5)
    assert "1 near-duplicate hidden" in planes.summarize_matches([make_match(0.05, hidden=1)], 1)
    assert "No samples" in planes.summarize_matches([], 3)
