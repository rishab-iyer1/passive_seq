import numpy as np
import pynapple as nap
import pytest

from nev_events import clean_codes, label, parse_events, photodiode_times, split_by

FRAME = 0.0083
SOA, LAG, ON = 0.6, 0.033, 0.283


def session(extra=()):
    """Cleaned codes for 2 blocks x 2 sequences x 4 images (+ any extra (t, code))."""
    ev = [(1.0, 5), (301.0, 1), (301.05, 1), (301.1, 1)]
    t, img = 302.0, 0
    for b in (1, 2):
        ev.append((t, 100 + b))
        for s in (1, 2):
            ev.append((t + 0.05, 1000 + s))
            for k in range(4):
                img += 1
                ev.append((t + 0.1 + k * SOA, 6000 + img))
            seq_end = t + 0.1 + 4 * SOA
            ev += [(seq_end, 3000 + s), (seq_end + 0.8, 2), (seq_end + 1.4, 3)]
            t = seq_end + 2.25
        ev.append((t - 0.1, 500 + b))
        t += 2.0
    ev.append((t, 4))
    ev = sorted(ev + list(extra))
    raw_t = np.ravel([[ti, ti + FRAME] for ti, _ in ev])        # each code, then its clear
    raw_d = np.ravel([[c, 0] for _, c in ev])
    return clean_codes(nap.Tsd(t=raw_t, d=raw_d))


def fake_photodiode(onsets, fs=10_000):
    t = np.arange(onsets[0] - 2, onsets[-1] + 2, 1 / fs)
    v = np.zeros_like(t)
    for o in onsets:
        v[(t >= o + LAG) & (t < o + LAG + ON)] = 1000
    return nap.Tsd(t=t, d=v)


def images_onsets():
    c = session()
    return c[c.d > 6000].t


def test_clean_codes_drops_clears_and_glitches():
    raw = nap.Tsd(t=np.array([1.0, 1.0001, 1.0083, 1.0084]),
                  d=np.array([6001 & 0xFF, 6001, 6001 & 0xFF00, 0]))
    assert list(clean_codes(raw).d) == [6001]


def test_phases():
    ev = parse_events(session())
    assert ev["baseline"].end[0] - ev["baseline"].start[0] == pytest.approx(300.0)
    assert list(ev["blocks"].block) == [1, 2]
    assert list(ev["sequences"].seq) == [1, 2, 1, 2]
    assert list(ev["reward"].block) == [1, 1, 2, 2]


def test_image_metadata():
    img = parse_events(session())["images"]
    assert list(img.image) == list(range(1, 17))
    assert list(img.block) == [1] * 8 + [2] * 8
    assert list(img.seq) == ([1] * 4 + [2] * 4) * 2
    assert list(img.pos) == [1, 2, 3, 4] * 4
    assert img.end - img.start == pytest.approx(np.full(16, ON), abs=1e-3)


def test_photodiode_sets_image_times():
    on = images_onsets()
    img = parse_events(session(), photodiode=fake_photodiode(on))["images"]
    assert img.start - on == pytest.approx(np.full(16, LAG), abs=2e-4)
    assert img.end - img.start == pytest.approx(np.full(16, ON), abs=2e-4)
    assert np.asarray(img.code_onset) == pytest.approx(on)


def test_photodiode_inverted_polarity():
    on = images_onsets()
    pd = fake_photodiode(on)
    start, _ = photodiode_times(nap.Tsd(t=pd.t, d=1000 - pd.d), on)
    assert start - on == pytest.approx(np.full(16, LAG), abs=2e-4)


def test_missing_photodiode_pulse_raises():
    on = images_onsets()
    with pytest.raises(AssertionError, match="no photodiode pulse"):
        photodiode_times(fake_photodiode(np.delete(on, 5)), on)


def test_duplicate_block_end_ignored():
    ev = parse_events(session(extra=[(999.0, 502)]))
    assert ev["blocks"].end[-1] < 999


def test_bad_codes_raise():
    c = session()
    with pytest.raises(AssertionError, match="block"):
        parse_events(c[c.d != 501])
    with pytest.raises(AssertionError, match="outside any sequence"):
        parse_events(session(extra=[(990.0, 6017)]))


def test_contexts_and_split():
    rules = {"odd": lambda x: x.block % 2 == 1}
    ev = parse_events(session(), contexts=rules)
    assert list(ev["images"].context) == ["odd"] * 8 + ["none"] * 8
    assert list(split_by(ev["images"], "context")) == ["odd"]
    assert list(split_by(ev["blocks"], "block", "block")) == ["block1", "block2"]

    spikes = nap.TsGroup({0: nap.Ts(t=np.linspace(300, 330, 3000))})
    resp = nap.compute_response_per_epoch(spikes, split_by(ev["images"], "pos", "pos"))
    assert list(resp.epochs.values) == ["pos1", "pos2", "pos3", "pos4"]


def test_overlapping_rules_raise():
    ev = parse_events(session())
    with pytest.raises(AssertionError, match="more than one rule"):
        label(ev["blocks"], "context", {"a": lambda x: x.block > 0, "b": lambda x: x.block == 1})