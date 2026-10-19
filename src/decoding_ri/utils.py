"""Parse Blackrock .nev digital event codes into pynapple objects.

Typical use:
    from nev_events import load_task_events, split_by

    CONTEXTS = {
        "context1": lambda x: (x.block <= 20) & (x.block % 2 == 1),
        "context2": lambda x: (x.block <= 20) & (x.block % 2 == 0),
    }
    ev = load_task_events("day2.nev", contexts=CONTEXTS)
    print(ev.summary())

    img = ev.images                                     # one interval per image
    img[(img.context == "context1") & (img.pos == 1)]   # first image of each context1 sequence
    nap.compute_perievent(spikes, ev.image_onsets, minmax=(-0.2, 0.6))
    nap.compute_response_per_epoch(spikes, split_by(img, "context"))

Structure:
    read_event_codes   -- file I/O only (neo); returns the raw port values
    clean_codes        -- drops clears (0) and half-written glitch values
    parse_task_events  -- pure function: cleaned codes -> TaskEvents
    load_task_events   -- convenience wrapper chaining the three
    make_epochs / label / split_by -- turn IntervalSets into named epoch dicts

Only read_event_codes touches the file, so everything else can be unit tested
on synthetic nap.Tsd streams.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pynapple as nap

# ── Encode table (from the task code) ───────────────────────────────
START_TASK = 1      # sent nStartTaskRepeats times (default 3)
REWARD_START = 2
REWARD_END = 3
END_TASK = 4
INITIAL_START = 5
BLOCK_START = 100   # + blockNum
BLOCK_END = 500     # + blockNum
SEQ_START = 1000    # + seqId
SEQ_END = 3000      # + seqId
IMAGE_ON = 6000     # + imgNum
MAX_CODE = 2**16 - 1

# A real code stays on the port until it is cleared one frame later (~8 ms);
# half-written values (port bits changing non-simultaneously) last ~0.1 ms.
DEFAULT_MIN_HOLD = 0.05  # seconds


# ── Task timing (from the task code) ────────────────────────────────
@dataclass(frozen=True)
class Timing:
    """Fixed task timing, in seconds. Change here, or override per call:
    load_task_events(path, display_duration=0.3).

    display_duration sets how long each image interval lasts; everything else
    is only used by check_timing to warn when the recording doesn't match.
    """

    display_duration: float = 0.25          # icon on screen  (displayDuration)
    inter_stim_interval: float = 0.15       # blank between icons  (interStimInterval)
    sequence_duration: float = 1.6          # 4 icons × 400 ms SOA
    reward_delay: tuple = (0.6, 1.0)        # sequence end → juice  (rewardDelayRange)
    juice_duration: float = 1.0             # 5 drops × 200 ms
    post_sequence: tuple = (2.0, 2.5)       # sequence end → next sequence  (postSeqTotal)
    inter_block: tuple = (1.5, 2.5)         # block end → next block  (interBlockInterval)
    baseline: float = 300.0                 # pre-task baseline  (baselineDelay)
    tolerance: float = 0.005                # allowed mismatch before warning

    @property
    def soa(self) -> float:
        return self.display_duration + self.inter_stim_interval


DEFAULT_TIMING = Timing()

@dataclass
class TaskEvents:
    """Experiment phases as IntervalSets, one interval per item.

    blocks    -- metadata: block
    sequences -- metadata: block, seq
    reward    -- metadata: block
    images    -- metadata: image (imgNum), block, seq, pos (1st, 2nd, … in its sequence)
    Every set with a `block` column also gets `context` when contexts are given.

    Image starts are when the code was sent (just before the flip). There is no
    image-off code, so each image lasts Timing.display_duration.
    """

    baseline: nap.IntervalSet
    task: nap.IntervalSet
    blocks: nap.IntervalSet
    sequences: nap.IntervalSet
    reward: nap.IntervalSet
    images: nap.IntervalSet

    @property
    def image_onsets(self) -> nap.Ts:
        """Image onset times, e.g. for compute_perievent."""
        return nap.Ts(t=self.images.start)

    def epochs(self, names=("baseline", "task", "blocks", "sequences", "reward", "images")) -> dict:
        """Phases as a plain {name: IntervalSet} dict (e.g. for compute_response_per_epoch)."""
        return {name: getattr(self, name) for name in names}

    def summary(self) -> str:
        lines = []
        for name, iset in self.epochs().items():
            lines.append(f"{name:10s} n={len(iset):5d}  total={iset.tot_length():8.1f} s")
        return "\n".join(lines)


# ── I/O ─────────────────────────────────────────────────────────────
def read_event_codes(path: str | Path, event_channel_index: int = 0) -> nap.Tsd:
    """Raw digital-port values from a .nev file (includes clears and glitches)."""
    from neo.rawio import BlackrockRawIO  # imported here so tests don't need neo

    r = BlackrockRawIO(str(path))
    r.parse_header()
    ts, _, labels = r.get_event_timestamps(
        block_index=0, seg_index=0, event_channel_index=event_channel_index
    )
    t = r.rescale_event_timestamp(ts, dtype="float64", event_channel_index=event_channel_index)
    return nap.Tsd(t=t, d=np.asarray(labels).astype(int))


# ── Cleaning ────────────────────────────────────────────────────────
def clean_codes(codes: nap.Tsd, min_hold: float = DEFAULT_MIN_HOLD) -> nap.Tsd:
    """Keep only values that stayed on the port for at least `min_hold` seconds.

    Removes the clears (0) and the transient half-written values that appear
    when the 16-bit code changes in two steps (~0.1 ms before a code, or
    while it is being cleared).
    """
    if len(codes) == 0:
        return codes
    held = np.r_[np.diff(codes.t), np.inf]
    return codes[(codes.d != 0) & (held > min_hold)]


# ── Parsing ─────────────────────────────────────────────────────────
def _select(codes: nap.Tsd, lo: int, hi: int | None = None) -> nap.Tsd:
    hi = lo if hi is None else hi
    return codes[(codes.d >= lo) & (codes.d <= hi)]


def _first_each(ev: nap.Tsd) -> nap.Tsd:
    """First occurrence of each distinct code, in time order."""
    _, idx = np.unique(ev.d, return_index=True)
    return ev[np.sort(idx)]


def _check_pairs(starts: nap.Tsd, ends: nap.Tsd, start_base: int, end_base: int, name: str):
    s_id, e_id = starts.d - start_base, ends.d - end_base
    if len(s_id) != len(e_id) or not np.array_equal(s_id, e_id):
        raise ValueError(
            f"{name}: start/end codes don't pair up "
            f"({len(s_id)} starts, {len(e_id)} ends)"
        )
    if np.any(starts.t >= ends.t):
        raise ValueError(f"{name}: some intervals end before they start")


def _containing(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the interval containing each time (-1 if none)."""
    idx = np.searchsorted(starts, times, side="right") - 1
    inside = (idx >= 0) & (times <= ends[np.clip(idx, 0, None)])
    return np.where(inside, idx, -1)


def parse_task_events(
    codes: nap.Tsd,
    contexts: dict | None = None,
    timing: Timing = DEFAULT_TIMING,
    require_sequential_images: bool = True,
) -> TaskEvents:
    """Build phase IntervalSets from cleaned event codes (see clean_codes).

    contexts -- optional rules dict (see `label`), applied to every set
    timing   -- task timing; the recording is checked against it (see check_timing)
    """
    init, start = _select(codes, INITIAL_START), _select(codes, START_TASK)
    if len(init) == 0:
        raise ValueError("no InitialStart code found")
    if len(start) == 0:
        raise ValueError("no StartTask code found")
    t_init, t_start = init.t[0], start.t[0]

    end = _select(codes, END_TASK)
    t_end = end.t[0] if len(end) else codes.t[-1]  # EndTask may be missing

    # Block markers can be re-sent (e.g. final block end at shutdown): keep first of each.
    b_st = _first_each(_select(codes, BLOCK_START + 1, BLOCK_END - 1))
    b_en = _first_each(_select(codes, BLOCK_END + 1, SEQ_START - 1))
    _check_pairs(b_st, b_en, BLOCK_START, BLOCK_END, "blocks")

    s_st = _select(codes, SEQ_START + 1, SEQ_END - 1)
    s_en = _select(codes, SEQ_END + 1, IMAGE_ON - 1)
    _check_pairs(s_st, s_en, SEQ_START, SEQ_END, "sequences")

    r_st, r_en = _select(codes, REWARD_START), _select(codes, REWARD_END)
    if len(r_st) != len(r_en):
        raise ValueError(f"reward: {len(r_st)} starts, {len(r_en)} ends")

    # ── which block each sequence / reward / image falls in ──
    def block_of(tt):
        i = _containing(tt, b_st.t, b_en.t)
        return np.where(i >= 0, b_st.d[i] - BLOCK_START, -1)

    # ── images ──
    on = _select(codes, IMAGE_ON + 1, MAX_CODE)
    if require_sequential_images and np.any(np.diff(on.d) != 1):
        raise ValueError("image codes are not strictly sequential")

    seq_i = _containing(on.t, s_st.t, s_en.t)
    if np.any(seq_i < 0):
        raise ValueError(f"{np.sum(seq_i < 0)} images fall outside any sequence")
    # position within sequence: count up from 1, restarting at each new sequence
    k = np.arange(len(seq_i))
    seq_first = np.maximum.accumulate(np.where(np.r_[True, np.diff(seq_i) != 0], k, 0))
    pos = k - seq_first + 1

    sets = dict(
        baseline=nap.IntervalSet(t_init, t_start),
        task=nap.IntervalSet(t_start, t_end),
        blocks=nap.IntervalSet(b_st.t, b_en.t, metadata={"block": b_st.d - BLOCK_START}),
        sequences=nap.IntervalSet(
            s_st.t, s_en.t, metadata={"block": block_of(s_st.t), "seq": s_st.d - SEQ_START}
        ),
        reward=nap.IntervalSet(r_st.t, r_en.t, metadata={"block": block_of(r_st.t)}),
        images=nap.IntervalSet(
            on.t, on.t + timing.display_duration,
            metadata={
                "image": on.d - IMAGE_ON,
                "block": block_of(on.t),
                "seq": s_st.d[seq_i] - SEQ_START,
                "pos": pos,
            },
        ),
    )
    if contexts:
        for name in ("blocks", "sequences", "reward", "images"):
            label(sets[name], "context", contexts)
    ev = TaskEvents(**sets)

    for msg in check_timing(ev, timing):
        warnings.warn(msg, stacklevel=2)
    return ev


def check_timing(ev: TaskEvents, timing: Timing = DEFAULT_TIMING) -> list[str]:
    """Compare measured intervals with `timing`; one message per mismatch.

    Fixed values must match within timing.tolerance; ranges must contain every
    measured value (± tolerance).
    """
    img, seq, blk, rew = ev.images, ev.sequences, ev.blocks, ev.reward
    same_seq = (np.diff(img.block) == 0) & (np.diff(img.seq) == 0)
    same_blk = np.diff(seq.block) == 0

    # reward → the sequence that ended just before it
    i = np.searchsorted(seq.end, rew.start) - 1
    has_seq = i >= 0

    measured = {
        "baseline": (np.array([ev.baseline.end[0] - ev.baseline.start[0]]), timing.baseline),
        "soa": (np.diff(img.start)[same_seq], timing.soa),
        "sequence_duration": (seq.end - seq.start, timing.sequence_duration),
        "juice_duration": (rew.end - rew.start, timing.juice_duration),
        "reward_delay": (rew.start[has_seq] - seq.end[i[has_seq]], timing.reward_delay),
        "post_sequence": ((seq.start[1:] - seq.end[:-1])[same_blk], timing.post_sequence),
        "inter_block": (blk.start[1:] - blk.end[:-1], timing.inter_block),
    }

    tol, msgs = timing.tolerance, []
    for name, (vals, expected) in measured.items():
        if len(vals) == 0:
            continue
        if isinstance(expected, tuple):
            lo, hi = expected
            bad = (vals < lo - tol) | (vals > hi + tol)
            want = f"{lo:g}–{hi:g} s"
        else:
            bad = np.abs(vals - expected) > tol
            want = f"{expected:g} s"
        if bad.any():
            msgs.append(
                f"timing '{name}': expected {want}, measured median {np.median(vals):.3f} s "
                f"(range {vals.min():.3f}–{vals.max():.3f}), {bad.sum()}/{len(vals)} out of range"
            )
    return msgs


def load_task_events(
    path: str | Path,
    contexts: dict | None = None,
    event_channel_index: int = 0,
    min_hold: float = DEFAULT_MIN_HOLD,
    require_sequential_images: bool = True,
    **timing_overrides,
) -> TaskEvents:
    """Read, clean and parse a .nev file in one call.

    Timing fields can be overridden by name, e.g. display_duration=0.3.
    """
    timing = replace(DEFAULT_TIMING, **timing_overrides)
    codes = clean_codes(read_event_codes(path, event_channel_index), min_hold)
    return parse_task_events(codes, contexts, timing, require_sequential_images)


# ── Epoch helpers ───────────────────────────────────────────────────
# A "rules" dict maps a name to a filter: a function taking an IntervalSet
# and returning a boolean mask, e.g.
#     CONTEXTS = {
#         "context1": lambda x: (x.block <= 20) & (x.block % 2 == 1),
#         "context2": lambda x: (x.block <= 20) & (x.block % 2 == 0),
#     }


def make_epochs(iset: nap.IntervalSet, rules: dict) -> dict:
    """{name: IntervalSet} with the intervals matching each rule.

    make_epochs(ev.blocks, CONTEXTS)  ->  {"context1": ..., "context2": ...}
    """
    return {name: iset[np.asarray(rule(iset))] for name, rule in rules.items()}


def label(iset: nap.IntervalSet, column: str, rules: dict, default: str = "none") -> nap.IntervalSet:
    """Add a metadata column naming the rule each interval matches (in place).

    Intervals matching no rule get `default`; matching two rules raises.

    label(ev.blocks, "context", CONTEXTS)
    ev.blocks[ev.blocks.context == "context1"]
    """
    names = np.full(len(iset), default, dtype=object)
    n_hits = np.zeros(len(iset), dtype=int)
    for name, rule in rules.items():
        mask = np.asarray(rule(iset))
        names[mask] = name
        n_hits += mask
    if np.any(n_hits > 1):
        raise ValueError(f"'{column}': some intervals match more than one rule")
    iset.set_info(**{column: names})
    return iset


def split_by(iset: nap.IntervalSet, column: str, prefix: str = "", drop=("none",)) -> dict:
    """{value: IntervalSet}, one entry per value of a metadata column.

    split_by(ev.blocks, "context")            ->  {"context1": ..., "context2": ...}
    split_by(ev.blocks, "block", "block")     ->  {"block1": ..., "block2": ..., ...}
    Values in `drop` (default: the `label` default "none") are left out.
    """
    values = np.asarray(getattr(iset, column))
    return {f"{prefix}{v}": iset[values == v] for v in np.unique(values) if v not in drop}