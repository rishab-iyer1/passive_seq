import warnings
from dataclasses import replace

import numpy as np
import pynapple as nap
import pytest

from nev_events import (
    DEFAULT_TIMING, check_timing, clean_codes, label, make_epochs, parse_task_events, split_by,
)

FRAME = 0.0083  # code is cleared one frame after being set


def stream(events):
    """Raw port stream from (time, code) pairs: each code followed by a clear."""
    t, d = [], []
    for ti, c in sorted(events):
        t += [ti, ti + FRAME]
        d += [c, 0]
    return nap.Tsd(t=np.array(t), d=np.array(d))


def session_events(timing=DEFAULT_TIMING, n_blocks=2, n_seqs=2, n_imgs=4):
    """(time, code) pairs for a session that follows `timing` exactly."""
    ev = [(1.0, 5)]
    t = 1.0 + timing.baseline
    ev += [(t, 1), (t + 0.05, 1), (t + 0.1, 1)]
    t += 0.5
    img = 0
    for b in range(1, n_blocks + 1):
        ev.append((t, 100 + b))
        t += 0.05
        for s in range(1, n_seqs + 1):
            ev.append((t, 1000 + s))
            for k in range(n_imgs):
                img += 1
                ev.append((t + 0.05 + k * timing.soa, 6000 + img))
            seq_end = t + n_imgs * timing.soa          # = sequence_duration with defaults
            ev.append((seq_end, 3000 + s))
            r = seq_end + np.mean(timing.reward_delay)
            ev += [(r, 2), (r + timing.juice_duration, 3)]
            t = seq_end + np.mean(timing.post_sequence)
        ev.append((t - 0.05, 500 + b))
        t += np.mean(timing.inter_block)
    ev.append((t, 4))
    return ev


def codes(events=None, **kw):
    return clean_codes(stream(session_events(**kw) if events is None else events))


def parse(c, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # fail on any timing warning
        return parse_task_events(c, **kw)


# ── cleaning ──
def test_clean_removes_clears():
    c = codes()
    assert np.all(c.d != 0)


def test_clean_removes_rising_and_falling_glitches():
    raw = nap.Tsd(
        t=np.array([11.1999, 11.2, 11.2082, 11.2083]),
        d=np.array([6001 & 0xFF, 6001, 6001 & 0xFF00, 0]),  # low byte first, high byte last
    )
    c = clean_codes(raw)
    assert list(c.d) == [6001]
    assert c.t[0] == pytest.approx(11.2)


# ── parsing ──
def test_parse_phases():
    ev = parse(codes())
    assert ev.baseline.end[0] - ev.baseline.start[0] == pytest.approx(300.0)
    assert list(ev.blocks.block) == [1, 2]
    assert list(ev.sequences.block) == [1, 1, 2, 2]
    assert list(ev.sequences.seq) == [1, 2, 1, 2]
    assert len(ev.reward) == 4


def test_images_metadata():
    img = parse(codes()).images
    assert list(img.image) == list(range(1, 17))
    assert list(img.block) == [1] * 8 + [2] * 8
    assert list(img.seq) == ([1] * 4 + [2] * 4) * 2
    assert list(img.pos) == [1, 2, 3, 4] * 4
    assert img.end - img.start == pytest.approx(np.full(16, 0.25))


def test_display_duration_override():
    ev = parse(codes(), timing=replace(DEFAULT_TIMING, display_duration=0.3,
                                       inter_stim_interval=0.1))
    assert ev.images.end - ev.images.start == pytest.approx(np.full(16, 0.3))


def test_duplicate_block_end_ignored():
    ev = parse(codes(session_events() + [(1000.0, 502)]))
    assert ev.blocks.end[-1] < 1000


def test_missing_end_task_falls_back_to_last_code():
    c = codes()
    ev = parse(c[c.d != 4])
    assert ev.task.end[0] == pytest.approx(c[c.d != 4].t[-1])


def test_mismatched_block_raises():
    c = codes()
    with pytest.raises(ValueError, match="blocks"):
        parse_task_events(c[c.d != 501])


def test_image_outside_sequence_raises():
    with pytest.raises(ValueError, match="outside any sequence"):
        parse_task_events(codes(session_events() + [(900.0, 6017)]))


# ── timing checks ──
def test_matching_timing_gives_no_messages():
    assert check_timing(parse(codes())) == []


def test_wrong_soa_warns():
    slow = replace(DEFAULT_TIMING, inter_stim_interval=0.35)    # 600 ms SOA in the data
    with pytest.warns(UserWarning, match="soa"):
        parse_task_events(codes(timing=slow))


# ── epoch helpers ──
RULES = {"odd": lambda x: x.block % 2 == 1}  # block 2 matches nothing


def test_contexts_applied_to_all_sets():
    rules = {"odd": lambda x: x.block % 2 == 1, "even": lambda x: x.block % 2 == 0}
    ev = parse(codes(), contexts=rules)
    assert list(ev.blocks.context) == ["odd", "even"]
    assert list(ev.sequences.context) == ["odd", "odd", "even", "even"]
    assert list(ev.images.context) == ["odd"] * 8 + ["even"] * 8


def test_epochs_dict_and_split_by():
    ev = parse(codes())
    assert set(ev.epochs()) == {"baseline", "task", "blocks", "sequences", "reward", "images"}
    per_block = split_by(ev.blocks, "block", "block")
    assert list(per_block) == ["block1", "block2"]

    spikes = nap.TsGroup({0: nap.Ts(t=np.linspace(0, 330, 2000))})
    resp = nap.compute_response_per_epoch(spikes, per_block)
    assert list(resp.epochs.values) == ["block1", "block2"]


def test_make_epochs():
    ep = make_epochs(parse(codes()).sequences, RULES)
    assert list(ep) == ["odd"]
    assert list(ep["odd"].block) == [1, 1]


def test_label_and_split():
    ev = parse(codes())
    label(ev.blocks, "context", RULES)
    assert list(ev.blocks.context) == ["odd", "none"]
    assert list(split_by(ev.blocks, "context")) == ["odd"]           # "none" dropped
    assert list(split_by(ev.blocks, "context", drop=())) == ["none", "odd"]


def test_label_overlapping_rules_raise():
    ev = parse(codes())
    with pytest.raises(ValueError, match="more than one rule"):
        label(ev.blocks, "context", {"a": lambda x: x.block > 0, "b": lambda x: x.block == 1})