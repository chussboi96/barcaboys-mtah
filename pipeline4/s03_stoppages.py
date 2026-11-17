"""Stage 03 -- dead-ball intervals (rules 4-7 and 13, with the goal and kickoff cases of 8-10).

Output: a frame-level dead-ball mask and a table of stoppages. Each stoppage is the half-open
interval [onset, restart): onset is the frame play stopped, restart the frame the ball is put back
into play -- the first live frame. Who restarts is stage 04's job.

How
---
1. Deadness score (rule 5). Every graded frame gets a log-odds that play is stopped, summed from
   stage 01/02 evidence (weights below, each with its reason): players collectively walking,
   ball beyond a line, ball lost for a long stretch, ball flying fast and passing sequences
   (against), a set-piece tag starting nearby (weak prior). Weights come from football reasoning,
   checked against pseudo-labels taken from the automatic data only (frames around Smart Tagging
   set-piece starts vs frames around completed-pass touches far from any set piece). No ground
   truth was used to set them.

2. Segmentation (rule 6). A two-state Viterbi pass per half turns the score into live/dead runs;
   the switching cost means a stoppage needs sustained evidence. Runs shorter than 3 s also need
   a strong mean score, unless a ball-out crossing or a set-piece tag backs them.

3. Anchors (rules 4, 7, 8, 9, 10):
     ball out      the ball crossing a line -- clearly (> 1 m), or vanishing right after -- stops
                   play. A dead run starting within 15 s after such a crossing, with no completed
                   pass in between, is extended back to the crossing; a crossing with no dead run
                   gets its own stoppage up to the next restart signature.
     half 2        dead from its first graded frame to the kickoff restart.
     goals         every 'Kick Off (After Goal)' tag: dead from the goal (the last goal-mouth
                   event -- crossing between the posts, or the ball lost at the goal mouth -- in
                   the matching GOALS tag; else the start of the dead run before the kickoff) to
                   the restart. The goal stands only if the ball is put back into play from the
                   centre spot; a line-up followed by a restart elsewhere is a disallowed goal
                   (VAR) and the stoppage runs to that free kick / goal kick instead. GOALS tags
                   without a kickoff are reported, not forced.
     set pieces    a graded set-piece tag no stoppage covers is rescued from the best dead-
                   leaning stretch around its start, if one lasts long enough.

4. Boundaries (rules 7, 13).
     onset    the ball-out crossing if there is one, else the peak of collective speed just
              before players slowed (the whistle), else the ball being lost.
     restart  the earliest restart signature in the last 5 s of the dead run (+2 s, because the
              2-s evidence windows make the run end late): a launch of a placed ball (still in
              the second before, then struck), the ball re-entering the pitch (throw-in), or the
              ball reappearing and moving after >= 2 s unseen. Kickoffs: a launch at the centre
              spot, else the moment players start moving from the kickoff line-up. The Smart
              Tagging end time is never used.

    python -m pipeline4.s03_stoppages

Outputs, in work/s03/:

    deadball.npz        per-frame arrays (see DEAD_ARRAYS)
    stoppages.csv       one row per stoppage
    goals.csv           every 'Kick Off (After Goal)' / GOALS tag and what was found
    tag_coverage.csv    every graded set-piece tag and the stoppage covering it
    s03_labels.csv      X outside the halves, D in stoppages, ? elsewhere
    p1d_labels.csv      diagnostic: D from here, H/A from pass 1 (s02) elsewhere
    meta.json           parameters, statistics and checks
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline4 import common as C  # noqa: E402
from pipeline4.s00_timeline import Checks, load_timeline  # noqa: E402
from pipeline4.s01_evidence import load_evidence, rolling_mean, runs  # noqa: E402
from pipeline4.s02_touches import is_local_max, load_contacts  # noqa: E402
from pipeline4.s01w_whistle import load_whistle  # noqa: E402

STAGE = "s03"
F = C.FPS

# --------------------------------------------------------------------------------------------
# parameters

# 1. deadness log-odds
BIAS = -0.5                               # live play is the more common state
SLOW_CENTRE, SLOW_WEIGHT = 0.6, 3.0       # share of slow players (2 s): 0.85 -> +1.5, 0.33 -> -1.6
SPEED_CENTRE, SPEED_WEIGHT = 1.6, 0.75     # mean player speed (2 s), m/s: 0.95 -> +1.0, 2.2 -> -0.9
OUT_CLEAR_SCORE = 3.0                     # ball > 1 m beyond a line
OUT_MARGINAL_SCORE = 1.5                  # ball beyond a line by less than that
GAP_LONG_S, GAP_LONG_SCORE = 10.0, 1.0    # ball lost for >= 10 s
GAP_MID_S, GAP_MID_SCORE = 4.0, 0.5       # ball lost 4-10 s (also happens in live play)
GAP_OPEN_SCORE = 0.0                      # ball lost up to the end of the half
FAST_BALL_DISP_M, FAST_BALL_SCORE = 8.0, -0.5   # ball travelled > 8 m in 1 s
PASS_SCORE, PASS_HALF_WINDOW = -1.0, 12         # within 0.5 s of a completed-pass touch
TAG_PRIOR_SCORE = 0.5                     # within [-10 s, +5 s] of a set-piece tag start
TAG_PRIOR_BEFORE, TAG_PRIOR_AFTER = 10 * F, 5 * F

# 1a. A ball in flight. A ball travelling at 6 m/s or more for at least half a second has
# just been struck: play is live around it. This is strong live evidence in the score, and the
# whistle proxy used for the stoppage onset cannot be placed before the end of such a flight.
FAST_FLIGHT_SPEED, FAST_FLIGHT_MIN_FRAMES, FAST_FLIGHT_SCORE = 6.0, F // 2, -1.0
# MTAH_FAST_FLIGHT=0 turns this off.
FAST_FLIGHT_ON = os.environ.get("MTAH_FAST_FLIGHT", "1") != "0"

# 1b. Live goalkeeper holds. A goalkeeper who collects a delivered ball (a shot, cross or
# back-pass arriving at >= 5 m/s, no ball-out crossing in the 10 s before) holds it in live play:
# the ball is in play until he releases it, however slowly everyone else walks meanwhile. The
# ball often vanishes under the keeper after the catch, so the live stretch starts at the last
# sighting of the fast ball. A hold that starts within 3 s of a live hold's release is the same
# keeper keeping the ball (drop, re-gather). A still ball held after a crossing, or one that was
# already slow, is a goal kick or free kick being prepared and stays with the normal score.
LIVE_HOLD_SCORE = -10.0
LIVE_HOLD_APPROACH_SPEED = 5.0          # m/s, same as the release speed in stage 02
LIVE_HOLD_APPROACH_WINDOW = 3 * F       # last 3 s of observed ball before the hold
LIVE_HOLD_LOOKBACK = 20 * F             # the fast ball must have been seen this recently
LIVE_HOLD_OUT_WINDOW = 10 * F           # no ball-out crossing this soon before the hold
LIVE_HOLD_CHAIN = 3 * F

# 1c. Players gone. With fewer than LOWCOUNT_MAX_PLAYERS tracked on the pitch for LOWCOUNT_MIN_FRAMES
# or more, no football is being played: the teams are walking off at half or full time, or are
# gathered off the pitch for a VAR check or a celebration. Those frames are scored as strongly dead
# as a confirmed keeper hold is scored live. Shorter dropouts happen in live play and are left alone.
# MTAH_LOWCOUNT=0 turns this off.
LOWCOUNT_ON = os.environ.get("MTAH_LOWCOUNT", "1") != "0"
LOWCOUNT_MAX_PLAYERS = 8
LOWCOUNT_MIN_FRAMES = 10 * F
LOWCOUNT_SCORE = 10.0

# 2. segmentation
# Live-ball evidence. A player carrying the ball, or a stretch between two same-side touches, is
# live play whatever the players' pace; a stoppage found only from walking pace needs a cause.
CARRY_SCORE = float(os.environ.get("MTAH_CARRY", "-0.5"))
PASSSPAN_SCORE = float(os.environ.get("MTAH_SPAN", "-0.5"))
NO_CAUSE_ON = os.environ.get("MTAH_NOCAUSE", "1") != "0"
PASSSPAN_MAX_GAP, PASSSPAN_MIN_SCORE, PASSSPAN_MIN_SEEN = 3 * F, 0.75, 0.7
CARRY_MIN_BALL_SPEED = 0.7
NO_CAUSE_MAX_BALL_GAP, NO_CAUSE_CARRY_SHARE = 0.2, 0.25
SWITCH_COST = 12.0             # log units to enter or leave a stoppage
START_STATE_COST = 5.0         # penalty for starting a half in the less likely state
MIN_DEAD_FRAMES = 2 * F        # a real stoppage lasts seconds
SHORT_RUN_FRAMES = 3 * F       # runs shorter than this ...
SHORT_RUN_MIN_SCORE = 1.0      # ... need this mean score unless a crossing or tag backs them

# 3. anchors
OUT_LOST_WITHIN = F // 2       # ball vanishing within 0.5 s of crossing a line ...
OUT_LOST_MIN_GAP = 2 * F       # ... for at least 2 s counts as out
OUT_LINK_FRAMES = 15 * F       # a dead run starting within 15 s after a crossing belongs to it
OUT_MAX_SEARCH = 60 * F        # restart search after an unlinked crossing
GOAL_HALF_WIDTH_M = 3.66 + 1.0 # half the goal mouth plus tracking tolerance
GOAL_LINE_NEAR_M = 3.0         # ball lost within 3 m of the goal line, inside the mouth
GOAL_TAG_MATCH = 60 * F        # a GOALS tag ending within 60 s of a kickoff tag is its goal
GOAL_MIN_BEFORE_KICKOFF = 10 * F   # a goal is at least 10 s before the kickoff
GOAL_MAX_BEFORE_KICKOFF = 5 * 60 * F
KICKOFF_CENTRE_M = 5.0         # a kickoff launch within 5 m of the centre spot (smoothed ball lags)
KICKOFF_BEFORE, KICKOFF_AFTER = 3 * F, 10 * F  # search around the kickoff tag window
KICKOFF_REAPPEAR_CENTRE_M = 20.0   # first sighting of the ball this close to the centre spot
KICKOFF_CONFIRM_SEARCH = 60 * F    # how long after a kickoff tag to look for the ball at all
RESCUE_BEFORE, RESCUE_AFTER = 15 * F, 10 * F
RESCUE_LEVEL = 0.0
RESCUE_MIN_FRAMES = 2 * F

# 4. boundaries
ONSET_BACK, ONSET_FWD = 2 * F, 1 * F
RESTART_BACK, RESTART_FWD = 5 * F, 2 * F
LAUNCH_MIN_DV = 3.0            # m/s change of velocity ...
LAUNCH_MAX_SPEED_BEFORE = 3.0  # ... of a ball moving at most this fast ...
LAUNCH_MIN_SPEED_AFTER = 5.0   # ... that then leaves at least this fast
PLACED_MAX_DISP_M = 2.5        # still in the second before: < 2.5 m net movement at some point
REAPPEAR_MIN_GAP = 2 * F
REAPPEAR_MIN_DISP = 3.0
SPEED_SMOOTH = 6
SET_PIECE_MATCH_BEFORE = 5 * F
SET_PIECE_MATCH_AFTER = 10 * F
RESTART_MOTION_WINDOW = F
RESTART_MOTION_MIN_DISP = 0.75  # ball movement after a restart touch, metres

DEAD_ARRAYS = {
    "dead_score": "(N,) float32 deadness log-odds, NaN outside the halves",
    "viterbi_dead": "(N,) bool  raw segmentation before anchors and boundary snapping",
    "launch": "(N,) bool        restart signature: a placed ball is struck",
    "dead": "(N,) bool          final dead-ball mask",
    "stoppage_id": "(N,) int32  row in stoppages.csv, -1 live or outside",
}


# --------------------------------------------------------------------------------------------
# 1. score

def set_piece_tags(t):
    ev = t.events
    return ev[(ev["family"] == "set_piece") & (ev["graded_frames"] > 0)]


def fast_flight_mask(e):
    """Frames inside a run of >= FAST_FLIGHT_MIN_FRAMES observed ball frames at >= FAST_FLIGHT_SPEED."""
    fast = e.ball_ok & (np.nan_to_num(e.ball_speed, nan=0.0) >= FAST_FLIGHT_SPEED)
    mask = np.zeros(e.n_frames, dtype=bool)
    for a, b in runs(fast):
        if b - a >= FAST_FLIGHT_MIN_FRAMES:
            mask[a:b] = True
    return mask


def live_hold_mask(e, c):
    """Frames inside a goalkeeper hold that began in live play (see LIVE_HOLD_*)."""
    n = e.n_frames
    mask = np.zeros(n, dtype=bool)
    holds = getattr(c, "holds", None)
    if holds is None or not len(holds):
        return mask
    prev_release, prev_half, prev_live = -10 ** 9, -1, False
    confirmed_col = "confirmed" in holds.columns
    if "virtual" in holds.columns:
        keep = ~holds["virtual"].astype(bool)
        if confirmed_col:
            keep = keep | holds["confirmed"].astype(bool)     # a confirmed hidden hold is live play
        holds = holds[keep]
    for row in holds.sort_values("start").itertuples(index=False):
        start, release, half = int(row.start), int(row.release), int(row.half)
        if confirmed_col and bool(row.confirmed):
            mask[start:release] = True
            prev_release, prev_half, prev_live = release, half, True
            continue
        chained = prev_live and prev_half == half and start - prev_release <= LIVE_HOLD_CHAIN
        lo = max(0, start - LIVE_HOLD_LOOKBACK)
        seen = np.flatnonzero(e.ball_ok[lo:start]) + lo
        recent_out = bool(e.ball_out[max(0, start - LIVE_HOLD_OUT_WINDOW):start].any())
        approach = np.nan
        last_seen = start
        if len(seen):
            last_seen = int(seen[-1])
            approach = float(np.nanmax(e.ball_speed[seen[-LIVE_HOLD_APPROACH_WINDOW:]]))
        collected = (not recent_out) and np.isfinite(approach) and approach >= LIVE_HOLD_APPROACH_SPEED
        live = collected or (chained and not recent_out)
        if live:
            first = prev_release if (chained and not collected) else min(start, last_seen)
            if e.half[first] != half:
                first = start
            mask[first:release] = True
        prev_release, prev_half, prev_live = release, half, live
    return mask


def live_ball_mask(e, c):
    """Frames where a player carries the ball (close control or slow carry, not a keeper hold) or
    that lie between two consecutive same-side touches with the ball mostly seen."""
    n = e.n_frames
    hold = getattr(c, "hold_side", np.full(n, -1, dtype=np.int8))
    carry = getattr(c, "carry_side", np.full(n, -1, dtype=np.int8))
    moving = np.nan_to_num(e.ball_speed_1s, nan=0.0) >= CARRY_MIN_BALL_SPEED   # a placed ball is not carried
    ctl = (((c.control_side >= 0) & (hold < 0)) | (carry >= 0)) & moving
    span = np.zeros(n, dtype=bool)
    tt = c.touches[c.touches["side_score"].abs() >= PASSSPAN_MIN_SCORE].sort_values("frame")
    fr, sd, hf = tt["frame"].to_numpy(), tt["side"].to_numpy(), tt["half"].to_numpy()
    for k in range(1, len(fr)):
        if sd[k] == sd[k - 1] and hf[k] == hf[k - 1] and 0 < fr[k] - fr[k - 1] <= PASSSPAN_MAX_GAP:
            if e.ball_ok[fr[k - 1]:fr[k] + 1].mean() >= PASSSPAN_MIN_SEEN:
                span[fr[k - 1]:fr[k] + 1] = True
    return ctl, span


def low_player_count_mask(e):
    """Frames inside a run of >= LOWCOUNT_MIN_FRAMES graded frames with < LOWCOUNT_MAX_PLAYERS tracked."""
    total = e.n_eligible.astype(np.int32).sum(axis=1)
    low = e.graded & (total < LOWCOUNT_MAX_PLAYERS)
    mask = np.zeros(e.n_frames, dtype=bool)
    for a, b in runs(low):
        if b - a >= LOWCOUNT_MIN_FRAMES:
            mask[a:b] = True
    return mask


def deadness_score(t, e, c):
    n = e.n_frames
    slow = rolling_mean(e.slow_share, F)
    speed = e.mean_speed_all_2s.astype(np.float64)
    score = np.full(n, BIAS, dtype=np.float64)
    score += np.nan_to_num(SLOW_WEIGHT * (slow - SLOW_CENTRE))
    score += np.nan_to_num(SPEED_WEIGHT * (SPEED_CENTRE - speed))
    score += np.where(e.ball_out_clear, OUT_CLEAR_SCORE, np.where(e.ball_out, OUT_MARGINAL_SCORE, 0.0))
    gap_s = e.ball_gap_len / float(F)
    missing = ~e.ball_ok
    score += np.where(missing & (gap_s >= GAP_LONG_S), GAP_LONG_SCORE,
                      np.where(missing & (gap_s >= GAP_MID_S), GAP_MID_SCORE,
                               np.where(missing & (e.ball_gap_len < 0), GAP_OPEN_SCORE, 0.0)))
    score += np.where(np.nan_to_num(e.ball_disp_1s) > FAST_BALL_DISP_M, FAST_BALL_SCORE, 0.0)
    score += np.where(pass_mask(c, n), PASS_SCORE, 0.0)
    prior = np.zeros(n, dtype=bool)
    for s in set_piece_tags(t)["start_frame"].to_numpy():
        prior[max(0, s - TAG_PRIOR_BEFORE):s + TAG_PRIOR_AFTER] = True
    score += np.where(prior, TAG_PRIOR_SCORE, 0.0)
    score += np.where(live_hold_mask(e, c), LIVE_HOLD_SCORE, 0.0)
    ctl_m, span_m = live_ball_mask(e, c)
    score += np.where(ctl_m, CARRY_SCORE, 0.0) + np.where(span_m & ~ctl_m, PASSSPAN_SCORE, 0.0)
    if FAST_FLIGHT_ON:
        score += np.where(fast_flight_mask(e), FAST_FLIGHT_SCORE, 0.0)
    if LOWCOUNT_ON:
        low = low_player_count_mask(e)
        score += np.where(low, LOWCOUNT_SCORE, 0.0)
        print("players gone: {} graded frames with < {} players for >= {} s".format(
            int(low.sum()), LOWCOUNT_MAX_PLAYERS, LOWCOUNT_MIN_FRAMES // F))
    score[~e.graded] = np.nan
    return score


def pass_mask(c, n):
    mask = np.zeros(n, dtype=bool)
    for f in c.touches.loc[c.touches["completes_pass"], "frame"].to_numpy():
        mask[max(0, f - PASS_HALF_WINDOW):f + PASS_HALF_WINDOW + 1] = True
    return mask


# --------------------------------------------------------------------------------------------
# 2. segmentation

def viterbi_two_state(score, start_dead):
    """Most likely live(0)/dead(1) path, emissions -/+ score/2, symmetric switch cost."""
    n = len(score)
    em = np.nan_to_num(score) / 2.0
    back = np.zeros((n, 2), dtype=np.int8)
    live = -em[0] - (START_STATE_COST if start_dead else 0.0)
    dead = em[0] - (0.0 if start_dead else START_STATE_COST)
    for i in range(1, n):
        a, b = live, dead - SWITCH_COST
        nl, back[i, 0] = (a, 0) if a >= b else (b, 1)
        a, b = dead, live - SWITCH_COST
        nd, back[i, 1] = (a, 1) if a >= b else (b, 0)
        live, dead = nl - em[i], nd + em[i]
    path = np.zeros(n, dtype=bool)
    state = 1 if dead > live else 0
    for i in range(n - 1, -1, -1):
        path[i] = state == 1
        state = back[i, state]
    return path


# --------------------------------------------------------------------------------------------
# signatures

def crossings(e, lo, hi, outward=True):
    f = np.arange(max(lo, 1), min(hi, e.n_frames))
    bd = e.ball_boundary_dist
    both = e.ball_ok[f] & e.ball_ok[f - 1]
    if outward:
        return f[both & (bd[f] < 0) & (bd[f - 1] >= 0)]
    return f[both & (bd[f] >= 0) & (bd[f - 1] < 0)]


def ball_out_events(e, first, last):
    """Crossings that really took the ball out: clearly beyond the line, or lost straight after."""
    out = []
    for c in crossings(e, first, last + 1, outward=True):
        look = slice(c, min(c + F, last + 1))
        bd = e.ball_boundary_dist[look]
        clear = bool(np.nanmin(np.where(np.isfinite(bd), bd, np.inf)) < -1.0)
        lost_at = np.flatnonzero(~e.ball_ok[c:c + OUT_LOST_WITHIN + 1])
        lost = len(lost_at) > 0 and e.ball_gap_len[c + lost_at[0]] >= OUT_LOST_MIN_GAP
        if clear or lost:
            if out and c - out[-1] < 2 * F:
                continue  # the same exit seen twice through tracking jitter
            out.append(int(c))
    return out


def launches(e, c):
    """Frames where a placed or slow ball is struck: restart signatures."""
    dv, sb, sa = c.ball_dv, c.ball_speed_before, c.ball_speed_after
    cand = (e.graded & np.isfinite(dv) & (dv >= LAUNCH_MIN_DV) &
            (sb <= LAUNCH_MAX_SPEED_BEFORE) & (sa >= LAUNCH_MIN_SPEED_AFTER))
    peak = cand & is_local_max(np.where(cand, dv, np.nan), 8)
    n = e.n_frames
    placed = np.zeros(n, dtype=bool)
    for f in np.flatnonzero(peak):
        prior = e.ball_disp_1s[max(0, f - F):max(0, f - 5)]
        unseen = e.ball_since_seen[max(0, f - F // 2)] > 0 if f >= F // 2 else False
        placed[f] = (np.isfinite(prior).any() and np.nanmin(prior) <= PLACED_MAX_DISP_M) or unseen
    return peak & placed


def restart_signatures(e, launch, lo, hi):
    """Sorted [(frame, kind)] in [lo, hi)."""
    lo, hi = max(lo, 1), min(hi, e.n_frames)
    if hi <= lo:
        return []
    out = [(int(f), "launch_placed_ball") for f in np.flatnonzero(launch[lo:hi]) + lo]
    out += [(int(f), "ball_reenters") for f in crossings(e, lo, hi, outward=False)]
    # movement after the ball reappears is measured *forward* from the first frame seen.
    # ball_disp_1s looks back one second, so it is empty on exactly the frames that matter and
    # the signature could never fire.
    f = np.arange(lo, hi)
    first_seen = f[e.ball_ok[f] & (e.ball_since_seen[f - 1] >= REAPPEAR_MIN_GAP)]
    for x in first_seen:
        w = slice(int(x), min(int(x) + RESTART_MOTION_WINDOW + 1, e.n_frames))
        ok = e.ball_ok[w]
        if ok.any() and np.nanmax(np.linalg.norm(e.ball_m[w][ok] - e.ball_m[x], axis=1)) >= REAPPEAR_MIN_DISP:
            out.append((int(x), "ball_reappears"))
    return sorted(out)


def set_piece_kind(tag_type):
    """Normalize Smart Tagging restart type; return None for non-restart tags."""
    value = str(tag_type or "").strip().lower()
    kinds = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick",
             "free kick": "free_kick", "kick off (start)": "kickoff",
             "kick off (after goal)": "kickoff"}
    return kinds.get(value)


def restart_touch_with_motion(e, c, lo, hi):
    """First touch followed by observed ball movement; avoids using tag time as restart time."""
    lo, hi = max(int(lo), 1), min(int(hi), e.n_frames)
    if hi <= lo:
        return None
    for f in np.flatnonzero(c.touch[lo:hi]) + lo:
        end = min(int(f) + RESTART_MOTION_WINDOW + 1, e.n_frames)
        movement = e.ball_disp_1s[int(f):end]
        speed = e.ball_speed[int(f):end]
        moved = (np.isfinite(movement).any() and np.nanmax(movement) >= RESTART_MOTION_MIN_DISP) or \
                (np.isfinite(speed).any() and np.nanmax(speed) >= LAUNCH_MIN_SPEED_AFTER)
        if e.ball_ok[f] and moved:
            return int(f)
    return None


def tag_for_interval(sp, onset, restart):
    """Choose the closest typed set-piece tag temporally associated with an interval."""
    if not len(sp):
        return None
    lo, hi = onset - SET_PIECE_MATCH_BEFORE, restart + SET_PIECE_MATCH_AFTER
    candidates = sp[(sp["start_frame"] <= hi) & (sp["end_frame"] >= lo)].copy()
    candidates = candidates[candidates["type"].map(set_piece_kind).notna()]
    if candidates.empty:
        return None
    # Prefer a tag overlapping the onset/restart span; otherwise choose the smallest gap.
    def distance(row):
        if row.start_frame <= restart and row.end_frame >= onset:
            return 0
        return min(abs(int(row.start_frame) - onset), abs(int(row.end_frame) - restart))
    return candidates.iloc[min(range(len(candidates)), key=lambda i: distance(candidates.iloc[i]))]


def refine_restart_from_tag(e, c, launch, interval, tag, block_last):
    """Use a nearby tag type to guide the restart search, never to set D at tag boundaries."""
    kind = set_piece_kind(tag["type"])
    if kind is None or kind == "kickoff":
        return interval
    onset, current = int(interval["onset"]), int(interval["restart"])
    lo = max(onset + 1, int(tag["start_frame"]) - F, current - RESTART_BACK)
    hi = min(block_last + 1, max(current + RESTART_FWD, int(tag["end_frame"]) + 2 * F))
    signatures = restart_signatures(e, launch, lo, hi)
    touch = restart_touch_with_motion(e, c, lo, hi)
    options = signatures + ([(touch, "tagged_{}_touch_motion".format(kind))] if touch is not None else [])
    if not options:
        return interval
    frame, how = min(options, key=lambda x: x[0])
    # The tag only guides the search. Reject a candidate implausibly far outside its interval.
    if frame < int(tag["start_frame"]) - 2 * F or frame > int(tag["end_frame"]) + 2 * F:
        return interval
    result = dict(interval)
    result["restart"], result["restart_how"] = int(frame), how
    result["source"] = str(interval["source"]) + "+typed_set_piece"
    return result


def players_move(e, lo, hi):
    """Frame players start moving after standing still (a kickoff line-up), or None.

    The floor is the lowest smoothed collective speed in [lo, hi); the rise is the first later
    frame 0.5 m/s above it; the restart is the last frame before the rise still within 0.15 m/s
    of the floor.
    """
    lo, hi = max(lo, 0), min(hi, e.n_frames)
    if hi - lo < F:
        return None
    sp = rolling_mean(e.mean_speed_all[lo:hi], SPEED_SMOOTH)
    if not np.isfinite(sp).any():
        return None
    k = int(np.nanargmin(sp))
    floor = sp[k]
    rise = np.flatnonzero(sp[k:] >= floor + 0.5)
    if not len(rise):
        return None
    r = k + int(rise[0])
    still = np.flatnonzero(sp[k:r] <= floor + 0.15)
    return lo + k + int(still[-1]) if len(still) else lo + k


def own_half_share(e, frame, home_attacks_positive_x):
    """Share of eligible players standing in their own half (1.0 = a kickoff line-up)."""
    h = int(e.half[frame])
    if h == 0:
        return float("nan")
    home_pos = bool(home_attacks_positive_x[str(h)])
    x = e.players_m[frame, :, 0]
    el = e.players_eligible[frame]
    home_own = (x < C.PITCH_LENGTH / 2) if home_pos else (x > C.PITCH_LENGTH / 2)
    own = np.where(e.slot_team == C.HOME, home_own, ~home_own) & el
    return float(own.sum()) / max(1, int(el.sum()))


def kickoff_restart(e, c, launch, tag_start, tag_end):
    """Locate and verify a kickoff. Returns (restart_frame, how, confirmed).

    Confirmed means the ball is put into play from the centre spot: a strike within
    KICKOFF_CENTRE_M of it, or -- when the ball is not tracked at the kick -- the first sighting
    of the ball after the tag within KICKOFF_REAPPEAR_CENTRE_M of it. A line-up for a kickoff that
    is then restarted elsewhere (a goal disallowed after review: the restart is a free kick or a
    goal kick) is NOT confirmed, and its restart is the first restart signature after the tag.
    """
    lo, hi = max(tag_start - KICKOFF_BEFORE, 0), min(tag_end + KICKOFF_AFTER, e.n_frames)
    f = np.arange(lo, hi)
    centre = np.nan_to_num(e.ball_centre_dist[f], nan=99)
    strike = f[(c.touch[f] | launch[f]) & (centre <= KICKOFF_CENTRE_M)]
    if len(strike):
        return int(strike[0]), "kickoff_strike", True

    # first time the ball is seen at or after the line-up
    far = np.arange(lo, min(tag_start + KICKOFF_CONFIRM_SEARCH, e.n_frames))
    seen = far[e.ball_ok[far]]
    if len(seen) and np.nan_to_num(e.ball_centre_dist[seen[0]], nan=99) <= KICKOFF_REAPPEAR_CENTRE_M:
        r = players_move(e, lo, int(seen[0]) + 1)
        if r is not None:
            return int(r), "kickoff_players_move", True
        return int(seen[0]), "kickoff_ball_seen", True
    if len(seen):
        sig = restart_signatures(e, launch, lo, int(seen[0]) + 2 * F)
        r = sig[-1][0] if sig else int(seen[0])
        return int(r), "restart_elsewhere_not_a_kickoff", False
    return int(tag_start), "kickoff_tag_start_unverified", False


# --------------------------------------------------------------------------------------------
# boundaries

def refine_onset(e, start, block_first, outs):
    """Ball-out crossing near the start, else the whistle (speed peak), else the ball lost."""
    lo, hi = max(block_first, start - ONSET_BACK), start + ONSET_FWD
    near = [o for o in outs if lo <= o < hi]
    if near:
        return near[0], "ball_out"
    # the whistle cannot come before the end of a ball flight just before the run
    flight = getattr(e, "_flight_mask", None)
    if flight is None:
        flight = fast_flight_mask(e) if FAST_FLIGHT_ON else np.zeros(e.n_frames, dtype=bool)
        e._flight_mask = flight
    seen = np.flatnonzero(flight[lo:start])
    if len(seen):
        lo = min(lo + int(seen[-1]) + 1, start)
    # nor before the last clean touch in the window (play was still being played)
    tf = getattr(e, "_touch_frames", None)
    if tf is not None and len(tf):
        k = tf[(tf >= lo) & (tf < start)]
        if len(k):
            lo = min(int(k[-1]) + 1, start)
    sp = rolling_mean(e.mean_speed_all[lo:hi], SPEED_SMOOTH)
    if np.isfinite(sp).any():
        return int(lo + np.nanargmax(sp)), "speed_peak"
    lost = np.flatnonzero(~e.ball_ok[lo:hi])
    if len(lost):
        return int(lo + lost[0]), "ball_lost"
    return int(start), "segmentation"


# Inferred ball-out. The track can stay on the pitch as a stale position while the real ball goes
# out (stage 01 flags it as ball_untrusted); the tracked crossing then only comes seconds later,
# when the ball is brought back to the line for the restart. A stoppage that starts at such a
# crossing is moved back to the start of the flagged stretch when the stretch begins within
# OUT_LINK_FRAMES of the crossing, the last trusted ball before it was within INFER_NEAR_LINE_M of
# the same line, and nobody has the ball under control in between. It never creates a stoppage.
# Off by default: the ball is labelled dead when play visibly stops, which can be well after the
# ball crosses the line unseen. MTAH_INFER_OUT=1 turns it on.
INFER_OUT_ON = os.environ.get("MTAH_INFER_OUT", "0") != "0"
INFER_NEAR_LINE_M = 3.0
INFER_NEAR_WINDOW = F
INFER_CONTROL_MIN_FRAMES = 10


def infer_ball_out_onsets(e, c, intervals):
    """Move ball-out onsets back to the flagged stretch that hid the real crossing. In place."""
    flagged = getattr(e, "ball_untrusted", None)
    if not INFER_OUT_ON or flagged is None or not flagged.any():
        return 0
    starts = np.array([a for a, _ in runs(flagged)], dtype=np.int64)
    ctl = c.control
    held = ctl[(ctl["frames"] >= INFER_CONTROL_MIN_FRAMES) | (ctl["kind"].astype(str) == "hold")]
    held_start = held["start"].to_numpy()
    bd, nb = e.ball_boundary_dist, e.ball_nearest_boundary
    moved = 0
    ordered = sorted(intervals, key=lambda d: (d["half"], d["onset"]))
    for k, d in enumerate(ordered):
        if d["onset_how"] != "ball_out":
            continue
        onset = int(d["onset"])
        line = int(nb[onset]) if nb[onset] >= 0 else int(nb[max(onset - 1, 0)])
        floor = onset - OUT_LINK_FRAMES
        if k and ordered[k - 1]["half"] == d["half"]:
            floor = max(floor, int(ordered[k - 1]["restart"]))
        for i in starts[(starts >= floor) & (starts < onset)]:
            i = int(i)
            if e.half[i] != e.half[onset] or not e.graded[i]:
                continue
            w = slice(max(i - INFER_NEAR_WINDOW, 0), i)
            near = e.ball_ok[w] & ~flagged[w] & (nb[w] == line) & (bd[w] >= 0) & (bd[w] <= INFER_NEAR_LINE_M)
            if not near.any():
                continue
            if ((held_start >= i) & (held_start < onset)).any():
                continue
            d["onset"], d["onset_how"] = i, "ball_out_inferred"
            moved += 1
            break
    return moved


def refine_restart(e, launch, onset, end, block_last):
    lo = max(onset + F, end - RESTART_BACK)
    sig = restart_signatures(e, launch, lo, min(end + RESTART_FWD, block_last + 1))
    if sig:
        return sig[0][0], sig[0][1]
    return int(end), "segmentation"


def merge(items):
    """Union of overlapping intervals in one half. Onset: the earliest. Restart: an anchored
    kickoff restart (goal, start of half) wins; otherwise the latest."""
    anchored = {"goal", "disallowed_goal", "kickoff_start_of_half"}
    items = sorted(items, key=lambda d: (d["half"], d["onset"], d["restart"]))
    merged = []
    for d in items:
        d = dict(d)
        d["sources"] = set(d["source"].split("+"))
        if merged and d["half"] == merged[-1]["half"] and d["onset"] < merged[-1]["restart"]:
            m = merged[-1]
            if d["onset"] < m["onset"]:
                m["onset"], m["onset_how"] = d["onset"], d["onset_how"]
            ma, da = bool(m["sources"] & anchored), bool(d["sources"] & anchored)
            if (da and not ma) or (da == ma and d["restart"] > m["restart"]):
                m["restart"], m["restart_how"] = d["restart"], d["restart_how"]
            m["sources"] |= d["sources"]
        else:
            merged.append(d)
    for m in merged:
        m["source"] = "+".join(sorted(m.pop("sources")))
    return [m for m in merged if m["restart"] > m["onset"]]


# --------------------------------------------------------------------------------------------
# goals

def goal_stoppages(t, e, c, launch, vit):
    """One row per 'Kick Off (After Goal)' tag (and per GOALS tag without one).

    A goal stands only if play restarts with a kickoff from the centre spot. A tagged kickoff
    whose restart happens elsewhere is a goal ruled out after the players had lined up (VAR): it
    is still a stoppage, from the goal to the real restart (a free kick or goal kick), and it is
    reported as disallowed. Nothing here knows the final score -- it is inferred and reported, so
    it can be checked against the real result rather than forced to match it.
    """
    ev = t.events[t.events["graded_frames"] > 0]
    goals = ev[ev["family"] == "goal"]
    kos = ev[(ev["family"] == "set_piece") & (ev["type"].str.lower() == "kick off (after goal)")]
    hp = t.meta["home_attacks_positive_x"]
    rows, intervals = [], []
    used_goals = set()
    for k_idx, ko in kos.iterrows():
        ks, ke = int(ko["start_frame"]), int(ko["end_frame"])
        restart, how, confirmed = kickoff_restart(e, c, launch, ks, ke)
        lineup = own_half_share(e, max(ks - F, 0), hp)
        match = goals[(goals["end_frame"] >= ks - GOAL_TAG_MATCH) & (goals["end_frame"] <= ke + GOAL_TAG_MATCH)]
        g_idx = int(match.index[0]) if len(match) else None
        lo = int(match["start_frame"].iloc[0]) - 5 * F if len(match) else restart - GOAL_MAX_BEFORE_KICKOFF
        hi = min(restart, ks) - GOAL_MIN_BEFORE_KICKOFF
        onset, onset_how = None, None
        if hi > lo:
            f = crossings(e, lo, hi, outward=True)
            mouth = f[(np.abs(e.ball_m[f, 1] - C.PITCH_WIDTH / 2) <= GOAL_HALF_WIDTH_M) &
                      (e.ball_nearest_boundary[f] <= 1)]
            fr = np.arange(lo, hi)
            lost = fr[e.ball_ok[fr] & ~e.ball_ok[np.minimum(fr + 1, e.n_frames - 1)] &
                      (np.abs(np.nan_to_num(e.ball_m[fr, 1], nan=-99) - C.PITCH_WIDTH / 2) <= GOAL_HALF_WIDTH_M) &
                      (np.nan_to_num(e.ball_goalline_dist[fr], nan=99) <= GOAL_LINE_NEAR_M)] + 1
            found = sorted([(int(x), "goal_crossing") for x in mouth] + [(int(x), "goal_ball_lost") for x in lost])
            if found:
                onset, onset_how = found[-1]
        if onset is None:
            j = min(restart, ks) - 1
            while j > 0 and not vit[j] and min(restart, ks) - j < 10 * F:
                j -= 1
            if vit[j]:
                while j > 0 and vit[j - 1]:
                    j -= 1
                onset, onset_how = j, "dead_run_before_kickoff"
        status = "goal" if confirmed else "disallowed_goal"
        rows.append({
            "kickoff_tag_row": k_idx, "kickoff_tag_start": ks, "lineup_own_half_share": round(lineup, 2),
            "restart": restart, "restart_how": how, "kickoff_confirmed": confirmed, "status": status,
            "goals_tag_row": g_idx, "goals_tag_side": goals.at[g_idx, "side"] if g_idx is not None else "",
            "goal_onset": onset, "goal_onset_how": onset_how,
            "ball_x_m_first_seen": round(float(e.ball_m[restart, 0]), 2) if e.ball_ok[restart] else None,
            "ball_y_m_first_seen": round(float(e.ball_m[restart, 1]), 2) if e.ball_ok[restart] else None,
        })
        if g_idx is not None:
            used_goals.add(g_idx)
        if onset is not None and restart > onset:
            intervals.append({"half": int(e.half[restart]), "onset": int(onset), "restart": int(restart),
                              "onset_how": onset_how, "restart_how": how, "source": status})
    for g_idx, g in goals.iterrows():
        if g_idx not in used_goals:
            rows.append({"goals_tag_row": g_idx, "goals_tag_side": g["side"], "status": "goal_tag_without_kickoff",
                         "kickoff_confirmed": False})
    return pd.DataFrame(rows), intervals


# One stoppage per goal. Nothing is live between a goal and the kickoff, but players sprint off to
# celebrate and the ball is played back to the centre spot, so the segmentation finds short live
# spells and the stoppage comes out in pieces. For a goal with a GOALS tag (awarded or ruled out),
# start from the stoppage that ends at the restart and keep absorbing the one before it while the
# live gap between them is at most GOAL_GAP_MAX_FRAMES, lies inside the tag window (less
# GOAL_TAG_LEAD), and holds no shot -- a ball flight of FAST_FLIGHT_SPEED or more for
# FAST_FLIGHT_MIN_FRAMES towards the goal, within GOAL_SHOT_ZONE_M of it. The scoring move itself is
# longer than the gap limit or contains the shot, so it stays live; so does a penalty.
# MTAH_GOAL_CHAIN=0 turns this off.
GOAL_CHAIN_ON = os.environ.get("MTAH_GOAL_CHAIN", "1") != "0"
GOAL_GAP_MAX_FRAMES = 5 * F
GOAL_TAG_LEAD = 5 * F
GOAL_SHOT_ZONE_M = 30.0


def shot_in(e, a, b, goal_x):
    """A fast ball flight towards the goal at goal_x, near it, inside [a, b)."""
    if b <= a:
        return False
    towards = -1.0 if goal_x == 0.0 else 1.0
    vx = np.nan_to_num(e.ball_v[a:b, 0].astype(np.float64), nan=0.0) * towards
    near = np.abs(np.nan_to_num(e.ball_m[a:b, 0].astype(np.float64), nan=1e9) - goal_x) <= GOAL_SHOT_ZONE_M
    fast = e.ball_ok[a:b] & near & (vx >= FAST_FLIGHT_SPEED)
    return any(hi - lo >= FAST_FLIGHT_MIN_FRAMES for lo, hi in runs(fast))


def merge_goal_chains(t, e, goals, intervals):
    """Absorb the pieces of each goal stoppage into one. Returns (intervals, n absorbed)."""
    if not GOAL_CHAIN_ON or goals is None or not len(goals) or "restart" not in goals.columns:
        return intervals, 0
    hp = t.meta["home_attacks_positive_x"]
    intervals = sorted(intervals, key=lambda d: (d["half"], d["onset"]))
    absorbed = 0
    for g_row, g in goals.iterrows():
        if g.get("status") not in ("goal", "disallowed_goal") or pd.isna(g.get("goals_tag_row")) \
                or g.get("goals_tag_side") not in ("H", "A") or pd.isna(g.get("restart")):
            continue
        restart = int(g["restart"])
        cur = next((d for d in intervals if d["onset"] < restart <= d["restart"]), None)
        if cur is None:
            continue
        half = int(cur["half"])
        tag_start = int(t.events.at[int(g["goals_tag_row"]), "start_frame"]) - GOAL_TAG_LEAD
        scorer_positive = bool(hp[str(half)]) == (g["goals_tag_side"] == "H")
        goal_x = float(C.PITCH_LENGTH) if scorer_positive else 0.0
        while True:
            before = [d for d in intervals if d["half"] == half and d["restart"] <= cur["onset"]]
            if not before:
                break
            prev = max(before, key=lambda d: d["restart"])
            a, b = int(prev["restart"]), int(cur["onset"])
            if b - a > GOAL_GAP_MAX_FRAMES or a < tag_start or shot_in(e, a, b, goal_x):
                break
            cur["onset"], cur["onset_how"] = prev["onset"], prev["onset_how"]
            cur["source"] = "+".join(sorted(set(cur["source"].split("+")) | set(prev["source"].split("+"))))
            intervals.remove(prev)
            absorbed += 1
        goals.at[g_row, "goal_onset"] = int(cur["onset"])
        goals.at[g_row, "goal_onset_how"] = cur["onset_how"]
    return intervals, absorbed


# --------------------------------------------------------------------------------------------
# main

# Still ball. A placed ball is not in play until it is kicked. Some restarts are placed where the
# players start moving or the ball reappears, while the ball is still lying on the spot. If, within
# STILLBALL_SETTLE_FRAMES of the restart, the ball is seen below STILLBALL_SPEED (rolling to rest,
# never leaving at the LAUNCH_MIN_SPEED_AFTER pace of a real restart), the stoppage stays open until
# the ball is struck: it reaches STILLBALL_STRIKE_SPEED or moves more than PLACED_MAX_DISP_M. The
# restart is left as it was if the ball is lost for more than STILLBALL_MAX_HOLE frames, nothing
# happens within STILLBALL_MAX_FRAMES, or the next stoppage or the end of the half comes first.
# Restarts only ever move later. MTAH_STILL_BALL=0 turns this off.
STILLBALL_ON = os.environ.get("MTAH_STILL_BALL", "1") != "0"
STILLBALL_SPEED = 1.0
STILLBALL_STRIKE_SPEED = 3.0
STILLBALL_SETTLE_FRAMES = 2 * F
STILLBALL_MAX_FRAMES = 20 * F
STILLBALL_MAX_HOLE = F // 2


def hold_open_on_still_ball(e, intervals, block_last_by_half):
    """Move restarts later to the strike of a ball seen lying still. In place; returns rows."""
    if not STILLBALL_ON:
        return []
    speed = np.nan_to_num(e.ball_speed, nan=0.0)
    ordered = sorted(intervals, key=lambda d: (d["half"], d["onset"]))
    moved = []
    for k, d in enumerate(ordered):
        r = int(d["restart"])
        last = int(block_last_by_half[int(d["half"])])
        if d["restart_how"] == "half_end" or r > last:
            continue
        limit = min(r + STILLBALL_MAX_FRAMES, last + 1)
        if k + 1 < len(ordered) and ordered[k + 1]["half"] == d["half"]:
            limit = min(limit, int(ordered[k + 1]["onset"]))
        settle = None
        for f in range(r, min(r + STILLBALL_SETTLE_FRAMES, limit)):
            if not e.ball_ok[f]:
                continue
            if speed[f] >= LAUNCH_MIN_SPEED_AFTER:
                break                         # leaving at restart pace: the restart was a strike
            if speed[f] < STILLBALL_SPEED:
                settle = f
                break
        if settle is None:
            continue
        rest = e.ball_m[settle].astype(np.float64)
        strike, hole = None, 0
        for f in range(settle + 1, limit):
            if not e.ball_ok[f]:
                hole += 1
                if hole > STILLBALL_MAX_HOLE:
                    break
                continue
            hole = 0
            if (speed[f] >= STILLBALL_STRIKE_SPEED or
                    float(np.hypot(*(e.ball_m[f].astype(np.float64) - rest))) > PLACED_MAX_DISP_M):
                strike = f
                break
        if strike is None or strike <= r:
            continue
        moved.append({"onset": int(d["onset"]), "old_restart": r, "new_restart": int(strike),
                      "old_how": d["restart_how"]})
        d["restart"], d["restart_how"] = int(strike), d["restart_how"] + "+still_ball"
    return moved


# Whistle onsets. A stoppage with no ball-out and no set-piece tag is a foul or some other referee
# call, and its start comes from the players' speed, which lags the whistle by seconds. Stage 01w
# measures whistle strength in the audio. For these stoppages only (not goals, kickoffs, tagged set
# pieces or tag rescues, and nothing with a set-piece tag near it, which rules out throw-ins,
# corners, goal kicks and tagged free kicks), the start moves to the strongest whistle of
# WHISTLE_MIN_DB or more in the WHISTLE_LOOKBACK frames before it. The whistle has to come at least
# WHISTLE_AFTER_RESTART after the previous restart and after the last completed pass, because play
# doesn't go on after a whistle. Starts only move earlier, no stoppage is created or removed, and
# without stage 01w output nothing changes. MTAH_WHISTLE=0 turns this off.
WHISTLE_ON = os.environ.get("MTAH_WHISTLE", "1") != "0"
WHISTLE_MIN_DB = 10.0               # above the running median of the band contrast (stage 01w)
WHISTLE_LOOKBACK = 20 * F
WHISTLE_AFTER_RESTART = 3 * F
WHISTLE_EXCLUDED_SOURCES = ("goal", "disallowed_goal", "kickoff_start_of_half", "typed_set_piece", "tag_rescue")


def whistle_onsets(e, sp_starts, intervals, blocks, strength, pass_frames=None):
    """Move untagged, no-ball-out stoppage onsets back to the referee's whistle. In place.

    pass_frames: sorted frames of completed passes. Play does not go on after the whistle: a team
    completing a pass shows the ball was still live, so the whistle must come after the last
    completed pass before the onset (a whistle heard earlier was crowd noise or another call).
    """
    if not WHISTLE_ON or strength is None:
        return []
    if pass_frames is None:
        pass_frames = np.zeros(0, dtype=np.int64)
    first_by_half = {h: a for h, a, _ in blocks}
    ordered = sorted(intervals, key=lambda d: (d["half"], d["onset"]))
    moved = []
    for k, d in enumerate(ordered):
        onset, restart, half = int(d["onset"]), int(d["restart"]), int(d["half"])
        if d["onset_how"] != "speed_peak":
            continue
        if any(src in WHISTLE_EXCLUDED_SOURCES for src in str(d["source"]).split("+")):
            continue
        if ((sp_starts >= onset - TAG_PRIOR_AFTER) & (sp_starts < restart + F)).any():
            continue
        lo = max(onset - WHISTLE_LOOKBACK, int(first_by_half[half]))
        if k and ordered[k - 1]["half"] == half:
            lo = max(lo, int(ordered[k - 1]["restart"]) + WHISTLE_AFTER_RESTART)
        played = pass_frames[(pass_frames >= lo) & (pass_frames < onset)]
        if len(played):
            lo = int(played[-1]) + 1
        if lo >= onset:
            continue
        w = strength[lo:onset]
        if not len(w) or float(w.max()) < WHISTLE_MIN_DB:
            continue
        peak = lo + int(np.argmax(w))
        start = peak
        while start - 1 >= lo and strength[start - 1] >= WHISTLE_MIN_DB:
            start -= 1
        if start >= onset:
            continue
        moved.append({"old_onset": onset, "new_onset": int(start), "restart": restart,
                      "peak_db": round(float(strength[peak]), 1)})
        d["onset"], d["onset_how"] = int(start), "whistle"
    return moved


def build(work_dir=None):
    t0 = time.time()
    checks = Checks()
    t = load_timeline(work_dir)
    e = load_evidence(work_dir)
    c = load_contacts(work_dir)
    n = e.n_frames
    g = e.graded
    blocks = C.games.half_video_blocks(C.GAME)
    print("loaded s00-s02")

    score = deadness_score(t, e, c)
    e._touch_frames = np.sort(c.touches.loc[(c.touches["side_score"].abs() >= 1.5) &
                                            (~c.touches["contested"].astype(bool)), "frame"].to_numpy())
    launch = launches(e, c)
    passes = pass_mask(c, n)
    _ctl, _span = live_ball_mask(e, c)
    carry_all = _ctl | _span
    sp = set_piece_tags(t)
    sp_starts = sp["start_frame"].to_numpy()
    ko_tags = t.events[(t.events["family"] == "set_piece") &
                       (t.events["type"].str.lower() == "kick off (start)")]

    vit = np.zeros(n, dtype=bool)
    intervals = []
    outs_all = []
    for h, first, last in blocks:
        hit = ko_tags[(ko_tags["start_frame"] <= first) & (ko_tags["end_frame"] > first)]
        at_kickoff = bool(len(hit))
        seg_score = score[first:last + 1].copy()
        if at_kickoff:
            # graded from the kickoff itself: the line-up is not a stoppage
            seg_score[:int(hit["end_frame"].max()) - first] = np.minimum(
                seg_score[:int(hit["end_frame"].max()) - first], -3.0)
        seg = viterbi_two_state(seg_score, start_dead=not at_kickoff)
        vit[first:last + 1] = seg
        outs = ball_out_events(e, first, last)
        outs_all += outs
        linked = set()

        for a, b in runs(seg):
            a, b = a + first, b + first
            backed = any(a - OUT_LINK_FRAMES <= o < b for o in outs) or \
                bool(((sp_starts >= a - TAG_PRIOR_BEFORE) & (sp_starts < b)).any())
            if b - a < SHORT_RUN_FRAMES and np.nanmean(score[a:b]) < SHORT_RUN_MIN_SCORE and not backed:
                continue
            # no ball-out, tag, ball gap or restart-like launch behind it, and the ball is carried
            # for a good share of it: slow live play, not a stoppage
            if (NO_CAUSE_ON and not backed and a != first and (1.0 - float(e.ball_ok[a:b].mean())) <= NO_CAUSE_MAX_BALL_GAP and
                    float(carry_all[a:b].mean()) >= NO_CAUSE_CARRY_SHARE):
                continue
            # a crossing up to 15 s before the run, with no completed pass since, starts it
            prior_outs = [o for o in outs if a - OUT_LINK_FRAMES <= o < a and not passes[o:a].any()]
            if a == first:
                onset, onset_how = first, "half_start"
            elif prior_outs:
                onset, onset_how = prior_outs[0], "ball_out"
            else:
                onset, onset_how = refine_onset(e, a, first, outs)
            linked.update(o for o in outs if onset <= o < b)
            if b > last:
                restart, restart_how = last + 1, "half_end"
            else:
                restart, restart_how = refine_restart(e, launch, onset, b, last)
            if restart - onset < MIN_DEAD_FRAMES and onset_how != "ball_out":
                continue
            intervals.append({"half": h, "onset": onset, "restart": restart, "onset_how": onset_how,
                              "restart_how": restart_how, "source": "segmentation"})

        # crossings no dead run picked up: dead until the next restart signature
        for o in outs:
            if o in linked:
                continue
            sig = restart_signatures(e, launch, o + F, min(o + OUT_MAX_SEARCH, last + 1))
            if sig:
                intervals.append({"half": h, "onset": o, "restart": sig[0][0], "onset_how": "ball_out",
                                  "restart_how": sig[0][1], "source": "ball_out"})

        # a half graded from before its kickoff is dead until the kickoff
        tags = ko_tags[(ko_tags["start_frame"] >= first) & (ko_tags["start_frame"] <= last)]
        if len(tags) and not at_kickoff:
            tag = tags.iloc[0]
            r, how, _ = kickoff_restart(e, c, launch, int(tag["start_frame"]), int(tag["end_frame"]))
            intervals.append({"half": h, "onset": first, "restart": r, "onset_how": "half_start",
                              "restart_how": how, "source": "kickoff_start_of_half"})

    goals, goal_iv = goal_stoppages(t, e, c, launch, vit)
    intervals = merge(intervals + goal_iv)
    intervals, n_goal_pieces = merge_goal_chains(t, e, goals, intervals)
    print("goal stoppage pieces absorbed: {}".format(n_goal_pieces))

    # A typed set-piece tag can narrow the restart search for a nearby stoppage. The
    # stoppage onset remains whatever the tracking/goal evidence found; tag timestamps
    # are never copied directly into the D interval boundaries.
    block_last_by_half = {hh: last for hh, _, last in blocks}
    for i, interval in enumerate(intervals):
        tag = tag_for_interval(sp, int(interval["onset"]), int(interval["restart"]))
        if tag is not None:
            intervals[i] = refine_restart_from_tag(
                e, c, launch, interval, tag, block_last_by_half[int(interval["half"])])

    # rescue set-piece tags that no stoppage covers
    smooth = rolling_mean(score, F // 2)
    for s in sp_starts:
        if any(d["onset"] <= s < d["restart"] + F for d in intervals):
            continue
        h = int(e.half[min(max(int(s), 0), n - 1)])
        if h == 0:
            continue
        first, last = [(a, b) for hh, a, b in blocks if hh == h][0]
        lo, hi = max(first, s - RESCUE_BEFORE), min(last + 1, s + RESCUE_AFTER)
        w = smooth[lo:hi]
        if not np.isfinite(w).any() or np.nanmax(w) < RESCUE_LEVEL:
            continue
        p = int(np.nanargmax(w))
        a = p
        while a > 0 and w[a - 1] >= RESCUE_LEVEL:
            a -= 1
        b = p
        while b + 1 < len(w) and w[b + 1] >= RESCUE_LEVEL:
            b += 1
        a, b = lo + a, lo + b + 1
        if b - a < RESCUE_MIN_FRAMES:
            continue
        onset, onset_how = refine_onset(e, a, first, outs_all)
        restart, restart_how = refine_restart(e, launch, onset, b, last)
        if restart - onset >= RESCUE_MIN_FRAMES:
            intervals.append({"half": h, "onset": onset, "restart": restart, "onset_how": onset_how,
                              "restart_how": restart_how, "source": "tag_rescue"})
    intervals = merge(intervals)

    # Apply one final duration floor to every interval source, including confirmed
    # ball-out events. Intervals are half-open [onset, restart), so require strictly
    # more than two seconds before any frame is labeled dead.
    pass_frames = np.sort(c.touches.loc[c.touches["completes_pass"].astype(bool), "frame"].to_numpy().astype(np.int64))
    moved_w = whistle_onsets(e, sp_starts, intervals, blocks, load_whistle(work_dir), pass_frames)
    print("whistle onsets: {} moved{}".format(
        len(moved_w), "" if load_whistle(work_dir) is not None else " (no stage 01w output: skipped)"))
    for m in moved_w:
        print("  stoppage to {}: onset {} -> {} ({} dB)".format(m["restart"], m["old_onset"], m["new_onset"],
                                                                 m["peak_db"]))
    intervals = [d for d in intervals if d["restart"] - d["onset"] > MIN_DEAD_FRAMES]
    still_moved = hold_open_on_still_ball(e, intervals, block_last_by_half)
    print("restarts held open on a still ball: {}".format(len(still_moved)))
    for m in still_moved:
        print("  stoppage from {}: restart {} -> {} ({})".format(m["onset"], m["old_restart"], m["new_restart"],
                                                                  m["old_how"]))
    print("ball-out onsets moved back to a flagged stretch: {}".format(infer_ball_out_onsets(e, c, intervals)))

    # -- per-frame outputs ----------------------------------------------------------------------
    dead = np.zeros(n, dtype=bool)
    sid = np.full(n, -1, dtype=np.int32)
    for i, d in enumerate(intervals):
        dead[d["onset"]:d["restart"]] = True
        sid[d["onset"]:d["restart"]] = i
    dead &= g
    sid[~g] = -1

    st = pd.DataFrame(intervals, columns=["half", "onset", "restart", "onset_how", "restart_how", "source"])
    if len(st):
        on = st["onset"].to_numpy()
        prev = np.clip(on - 1, 0, n - 1)
        st["duration_s"] = ((st["restart"] - st["onset"]) / F).round(2)
        st["mean_score"] = [round(float(np.nanmean(score[a:b])), 2) for a, b in zip(st["onset"], st["restart"])]
        st["last_contact_side"] = np.where(c.last_contact_side[prev] >= 0,
                                           np.array(["H", "A"])[np.clip(c.last_contact_side[prev], 0, 1)], "")
        st["last_contact_frame"] = c.last_contact_frame[prev]
        st["last_contact_contested"] = c.last_contact_contested[prev]
        st["ball_x_m_at_onset"] = np.round(e.ball_m[on, 0], 2)
        st["ball_y_m_at_onset"] = np.round(e.ball_m[on, 1], 2)
        names = np.array(["goal_line_x0", "goal_line_xL", "touchline_y0", "touchline_yW"])
        nb = e.ball_nearest_boundary[on]
        st["boundary_at_onset"] = np.where(st["onset_how"].isin(["ball_out", "ball_out_inferred"]) & (nb >= 0),
                                           names[np.clip(nb, 0, 3)], "")
        rs = np.clip(st["restart"].to_numpy(), 0, n - 1)
        st["ball_x_m_at_restart"] = np.round(e.ball_m[rs, 0], 2)
        st["ball_y_m_at_restart"] = np.round(e.ball_m[rs, 1], 2)
        st["set_piece_tags"] = ["|".join("{}:{}:{}".format(int(i), r["type"], r["side"] or "N/A")
                                         for i, r in sp[(sp["start_frame"] >= a - TAG_PRIOR_AFTER) &
                                                        (sp["start_frame"] < b + F)].iterrows())
                                for a, b in zip(st["onset"], st["restart"])]
    st.index.name = "stoppage"

    cov = []
    for idx, row in sp.iterrows():
        s = int(row["start_frame"])
        hit = [i for i, d in enumerate(intervals) if d["onset"] <= s < d["restart"] + F]
        cov.append({"tag_row": idx, "type": row["type"], "side": row["side"], "start_frame": s,
                    "stoppage": hit[0] if hit else -1})
    cov = pd.DataFrame(cov)

    labels = np.where(~g, "X", np.where(dead, "D", "?"))
    p1 = c.p1_side
    p1d = np.where(~g, "X", np.where(dead, "D", np.where(p1 == 0, "H", np.where(p1 == 1, "A", "X"))))

    # -- checks ---------------------------------------------------------------------------------
    checks.add("no dead frame outside the halves", not bool((dead & ~g).any()))
    checks.add("stoppages are ordered and do not overlap",
               all(a["restart"] <= b["onset"] for a, b in zip(intervals, intervals[1:]) if a["half"] == b["half"]))
    cov_pct = 100.0 * float((cov["stoppage"] >= 0).mean()) if len(cov) else 0.0
    checks.add("set-piece tags covered by a stoppage", cov_pct >= 80,
               "{:.0f}% of {} tags".format(cov_pct, len(cov)), severity="warning")
    n_goal = int((goals["status"] == "goal").sum()) if len(goals) else 0
    n_dis = int((goals["status"] == "disallowed_goal").sum()) if len(goals) else 0
    checks.add("goals confirmed by a centre-spot kickoff", True,
               "{} stand, {} disallowed (lined up, restarted elsewhere), {} GOALS tags without kickoff".format(
                   n_goal, n_dis, int((goals["status"] == "goal_tag_without_kickoff").sum()) if len(goals) else 0),
               severity="warning")
    dead_pct = 100.0 * float(dead[g].mean())
    checks.add("dead share of graded frames is plausible", 20 <= dead_pct <= 50,
               "{:.1f}% (top-flight football: the ball is out of play roughly 40-45% of the time)".format(dead_pct),
               severity="warning")
    print("\nchecks")
    checks.report()
    if checks.failed("error"):
        raise SystemExit("\ncritical check failed; nothing written.")

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    np.savez(os.path.join(out, "deadball.npz"), dead_score=score.astype(np.float32),
             viterbi_dead=vit, launch=launch, dead=dead, stoppage_id=sid)
    st.to_csv(os.path.join(out, "stoppages.csv"))
    goals.to_csv(os.path.join(out, "goals.csv"), index=False)
    cov.to_csv(os.path.join(out, "tag_coverage.csv"), index=False)
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(os.path.join(out, "s03_labels.csv"), index=False)
    pd.DataFrame({"frame": np.arange(n), "label": p1d}).to_csv(os.path.join(out, "p1d_labels.csv"), index=False)

    dur = st["duration_s"].to_numpy() if len(st) else np.array([0.0])
    stats = {
        "stoppages": int(len(st)),
        "dead_pct_graded": round(dead_pct, 2),
        "duration_median_s": round(float(np.median(dur)), 2),
        "duration_p10_p90_s": [round(float(np.percentile(dur, 10)), 2), round(float(np.percentile(dur, 90)), 2)],
        "ball_out_events": len(outs_all),
        "by_source": st["source"].value_counts().to_dict() if len(st) else {},
        "onset_how": st["onset_how"].value_counts().to_dict() if len(st) else {},
        "restart_how": st["restart_how"].value_counts().to_dict() if len(st) else {},
        "set_piece_tag_coverage_pct": round(cov_pct, 1),
        "goals_standing": n_goal, "goals_disallowed": n_dis,
    }
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE,
        "parameters": {k: v for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, tuple, str))},
        "stats_graded": stats, "arrays": DEAD_ARRAYS, "checks": checks.items,
        "note": "p1d_labels.csv is a diagnostic: stage-03 dead balls over pass-1 possession.",
    })
    print("\nstats")
    for k, v in stats.items():
        print("  {:<28} {}".format(k, v))
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


class DeadBall(object):
    """Read-only view of work/s03: d.dead, d.stoppage_id, d.stoppages, d.goals ..."""

    def __init__(self, work_dir=None):
        folder = os.path.join(work_dir or C.WORK_DIR, STAGE)
        path = os.path.join(folder, "deadball.npz")
        if not os.path.isfile(path):
            raise IOError("{} not found -- run: python -m pipeline4.s03_stoppages".format(path))
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.stoppages = pd.read_csv(os.path.join(folder, "stoppages.csv"), keep_default_na=False)
        self.goals = pd.read_csv(os.path.join(folder, "goals.csv"))
        self.tag_coverage = pd.read_csv(os.path.join(folder, "tag_coverage.csv"))
        self.meta = C.read_json(os.path.join(folder, "meta.json"))


def load_deadball(work_dir=None):
    return DeadBall(work_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 03: dead-ball intervals.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    args = parser.parse_args(argv)
    build(args.work_dir)


if __name__ == "__main__":
    main()
