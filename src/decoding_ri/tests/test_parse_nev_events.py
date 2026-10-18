import numpy as np
import pynapple as nap
import pytest

from utils import clean_codes, parse_task_events

FRAME = 0.0083  # code is cleared one frame after being set


def stream(events):
    """Raw port stream from (time, code) pairs: each code followed by a clear."""
    t, d = [], []
    for ti, c in events:
        t += [ti, ti + FRAME]
        d += [c, 0]
    return nap.Tsd(t=np.array(t), d=np.array(d))


def session(extra=()):
    ev = [
        (1.0, 5),                                   # InitialStart
        (10.0, 1), (10.1, 1), (10.2, 1),            # StartTask x3
        (11.0, 101), (11.1, 1001),                  # block 1, seq 1
        (11.2, 6001), (11.8, 6002), (12.4, 3001),
        (12.6, 2), (12.8, 3), (13.0, 501),
        (14.0, 102), (14.1, 1001),                  # block 2, seq 1
        (14.2, 6003), (14.8, 6004), (15.4, 3001),
        (15.6, 2), (15.8, 3), (16.0, 502),
        (17.0, 4),                                  # EndTask
        *extra,
    ]
    return stream(sorted(ev))


def test_clean_removes_clears():
    c = clean_codes(session())
    assert np.all(c.d != 0)
    assert len(c) == 21


def test_clean_removes_rising_and_falling_glitches():
    raw = nap.Tsd(
        t=np.array([11.1999, 11.2, 11.2082, 11.2083]),
        d=np.array([6001 & 0xFF, 6001, 6001 & 0xFF00, 0]),  # low byte first, high byte last
    )
    c = clean_codes(raw)
    assert list(c.d) == [6001]
    assert c.t[0] == pytest.approx(11.2)


def test_parse_phases():
    ev = parse_task_events(clean_codes(session()))
    assert ev.baseline.start[0] == pytest.approx(1.0)
    assert ev.baseline.end[0] == pytest.approx(10.0)
    assert ev.task.end[0] == pytest.approx(17.0)
    assert len(ev.blocks) == 2 and list(ev.blocks.block) == [1, 2]
    assert len(ev.sequences) == 2 and list(ev.sequences.block) == [1, 2]
    assert len(ev.reward) == 2
    assert list(ev.images.d) == [1, 2, 3, 4]


def test_duplicate_block_end_ignored():
    ev = parse_task_events(clean_codes(session(extra=[(18.0, 502)])))
    assert ev.blocks.end[-1] == pytest.approx(16.0)


def test_missing_end_task_falls_back_to_last_code():
    raw = session()
    c = clean_codes(raw)
    c = c[c.d != 4]
    ev = parse_task_events(c)
    assert ev.task.end[0] == pytest.approx(16.0)


def test_mismatched_block_raises():
    c = clean_codes(session())
    c = c[c.d != 501]
    with pytest.raises(ValueError, match="blocks"):
        parse_task_events(c)