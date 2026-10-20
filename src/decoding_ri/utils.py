"""Task events from a Blackrock .nev file, as a dict of pynapple IntervalSets.

    from nev_events import load_task_events, split_by

    CONTEXTS = {
        "context1": lambda x: (x.block <= 20) & (x.block % 2 == 1),
        "context2": lambda x: (x.block <= 20) & (x.block % 2 == 0),
    }
    ev = load_task_events(nev_path, photodiode=data["1:nsx4"][:, 3], contexts=CONTEXTS)

    img = ev["images"]                                   # one interval per image
    img[(img.context == "context1") & (img.pos == 1)]    # filter on any metadata
    nap.compute_response_per_epoch(spikes, split_by(img, "context"))
    nap.compute_perievent(spikes, nap.Ts(img.start), minmax=(-0.2, 0.6))

Keys: baseline, task, blocks, sequences, reward, images.
Metadata: block (all but baseline/task), seq (sequences, images), image and
pos (images; pos = 1st, 2nd, ... in its sequence), context (if given).
"""

import numpy as np
import pynapple as nap

# Encode table (from the task code)
START_TASK, REWARD_START, REWARD_END, END_TASK, INITIAL_START = 1, 2, 3, 4, 5
BLOCK_START, BLOCK_END = 100, 500    # + blockNum
SEQ_START, SEQ_END = 1000, 3000      # + seqId
IMAGE_ON = 6000                      # + imgNum

DISPLAY_DURATION = 0.283   # s on screen (17 frames at 60 Hz, measured by photodiode)
MAX_PHOTODIODE_LAG = 0.1   # s; photodiode must change this soon after the image code
MIN_HOLD = 0.001           # s; real codes stay on the port ~8 ms, glitches ~0.1 ms


def load_task_events(path, photodiode=None, contexts=None, display_duration=DISPLAY_DURATION):
    """Read a .nev file and return {name: IntervalSet}.

    photodiode       -- 1-channel pynapple time series; image start/end become the
                        photodiode on/off times. Without it, images start at the
                        code time and last `display_duration`.
    contexts         -- {name: rule}, see `label`; adds a `context` column.
    """
    return parse_events(read_codes(path), photodiode, contexts, display_duration)


def read_codes(path):
    """Digital-port codes as a Tsd, with port clears and half-written glitches removed."""
    from neo.rawio import BlackrockRawIO

    r = BlackrockRawIO(str(path))
    r.parse_header()
    ts, _, labels = r.get_event_timestamps(block_index=0, seg_index=0, event_channel_index=0)
    t = r.rescale_event_timestamp(ts, dtype="float64", event_channel_index=0)
    return clean_codes(nap.Tsd(t=t, d=np.asarray(labels).astype(int)))


def clean_codes(codes):
    """Keep values that stayed on the port longer than MIN_HOLD (drops 0s and glitches)."""
    held = np.r_[np.diff(codes.t), np.inf]
    return codes[(codes.d != 0) & (held > MIN_HOLD)]


def parse_events(codes, photodiode=None, contexts=None, display_duration=DISPLAY_DURATION):
    """Cleaned codes -> {name: IntervalSet}. See load_task_events."""

    def sel(lo, hi=None):
        return codes[(codes.d >= lo) & (codes.d <= (lo if hi is None else hi))]

    def first_each(ev):  # markers can be re-sent (e.g. last block end): keep the first
        return ev[np.sort(np.unique(ev.d, return_index=True)[1])]

    b_on, b_off = first_each(sel(BLOCK_START + 1, BLOCK_END - 1)), first_each(sel(BLOCK_END + 1, SEQ_START - 1))
    s_on, s_off = sel(SEQ_START + 1, SEQ_END - 1), sel(SEQ_END + 1, IMAGE_ON - 1)
    r_on, r_off = sel(REWARD_START), sel(REWARD_END)
    img = sel(IMAGE_ON + 1, 2**16 - 1)

    assert np.array_equal(b_on.d - BLOCK_START, b_off.d - BLOCK_END), "block start/end codes don't pair up"
    assert np.array_equal(s_on.d - SEQ_START, s_off.d - SEQ_END), "sequence start/end codes don't pair up"
    assert len(r_on) == len(r_off), "reward start/end counts differ"
    assert np.all(np.diff(img.d) == 1), "image codes are not sequential"

    blk = _containing(img.t, b_on.t, b_off.t)
    seq = _containing(img.t, s_on.t, s_off.t)
    assert np.all(seq >= 0), "some images fall outside any sequence"
    k = np.arange(len(seq))
    pos = k - np.maximum.accumulate(np.where(np.r_[True, np.diff(seq) != 0], k, 0)) + 1

    if photodiode is None:
        start, end = img.t, img.t + display_duration
    else:
        start, end = photodiode_times(photodiode, img.t)

    def block_of(t):
        return b_on.d[_containing(t, b_on.t, b_off.t)] - BLOCK_START

    t_end = sel(END_TASK).t[0] if len(sel(END_TASK)) else codes.t[-1]
    ev = {
        "baseline": nap.IntervalSet(sel(INITIAL_START).t[0], sel(START_TASK).t[0]),
        "task": nap.IntervalSet(sel(START_TASK).t[0], t_end),
        "blocks": nap.IntervalSet(b_on.t, b_off.t, metadata={"block": b_on.d - BLOCK_START}),
        "sequences": nap.IntervalSet(s_on.t, s_off.t, metadata={
            "block": block_of(s_on.t), "seq": s_on.d - SEQ_START}),
        "reward": nap.IntervalSet(r_on.t, r_off.t, metadata={"block": block_of(r_on.t)}),
        "images": nap.IntervalSet(start, end, metadata={
            "image": img.d - IMAGE_ON, "block": b_on.d[blk] - BLOCK_START,
            "seq": s_on.d[seq] - SEQ_START, "pos": pos, "code_onset": img.t}),
    }
    if contexts:
        for name in ("blocks", "sequences", "reward", "images"):
            label(ev[name], "context", contexts)
    return ev


def photodiode_times(photodiode, onsets):
    """(on, off) photodiode times for each image code time in `onsets`."""
    w = photodiode.get(onsets[0] - 1, onsets[-1] + 1)          # one contiguous read
    t, v = np.asarray(w.t), np.asarray(w.values, dtype=float).ravel()

    lo, hi = np.percentile(v, [5, 95])
    lit = np.abs(v - hi) < np.abs(v - lo)                      # closer to the high level
    before = np.searchsorted(t, onsets) - 1                    # sample just before each code
    if lit[before].mean() > 0.5:                               # screen patch is dark when on
        lit = ~lit

    rises = t[1:][~lit[:-1] & lit[1:]]
    falls = t[1:][lit[:-1] & ~lit[1:]]
    on = rises[np.searchsorted(rises, onsets).clip(max=len(rises) - 1)]
    off = falls[np.searchsorted(falls, on).clip(max=len(falls) - 1)]
    missing = (on < onsets) | (on - onsets > MAX_PHOTODIODE_LAG) | (off <= on)
    assert not missing.any(), f"{missing.sum()} images have no photodiode pulse"
    return on, off


def _containing(times, starts, ends):
    """Index of the interval containing each time (-1 if none)."""
    i = np.searchsorted(starts, times, side="right") - 1
    return np.where((i >= 0) & (times <= ends[i.clip(min=0)]), i, -1)


# ── grouping ────────────────────────────────────────────────────────
def label(iset, column, rules, default="none"):
    """Add a metadata column naming the rule each interval matches (in place).

    rules: {name: function(IntervalSet) -> boolean mask}. Unmatched intervals get
    `default`; an interval matching two rules raises.
    """
    names = np.full(len(iset), default, dtype=object)
    hits = np.zeros(len(iset), dtype=int)
    for name, rule in rules.items():
        mask = np.asarray(rule(iset))
        names[mask] = name
        hits += mask
    assert np.all(hits <= 1), f"'{column}': some intervals match more than one rule"
    iset.set_info(**{column: names})
    return iset


def split_by(iset, column, prefix="", drop=("none",)):
    """{value: IntervalSet}, one entry per value of a metadata column.

    split_by(ev["images"], "context")         -> {"context1": ..., "context2": ...}
    split_by(ev["blocks"], "block", "block")  -> {"block1": ..., "block2": ..., ...}
    """
    values = np.asarray(getattr(iset, column))
    return {f"{prefix}{v}": iset[values == v] for v in np.unique(values) if v not in drop}
    