"""Stage 05 -- the possession model inside live play (rules 15-19).

Every live stretch -- from a restart (stage 04) to the next stoppage (stage 03) -- is decoded on
its own with a two-state Viterbi model, H or A in possession. Stoppages are never crossed, so a
stoppage always separates two possessions (rule 21 is respected by construction).

Evidence, as per-frame log-odds for home (+) vs away (-):

  restart prior   (rule 19) the restarting side, as the starting state, with strength growing
                  with the restart's confidence. Evidence after the restart can overturn it, so a
                  mishit or intercepted restart is not forced to stay with the restarting side.
  touches         (rule 17) an impulse by a side's player: a trap/reception is strong, a kick
                  shows who had the ball, a redirect is weak; contested touches count a third,
                  and every touch is scaled by how sure stage 02 is about which side made it.
  close control   (rules 17, 18) every frame of a control spell (reception, dribble, shielding)
                  counts for that side; a touch completing a same-side pass adds a bonus.
  goalkeeper hold the ball held in a side's own penalty area (stage 02) counts more than close
                  control, and nearby opponents count for nothing while it lasts.
  proximity       a small per-frame vote for a side whose player is right at the ball with the
                  nearest opponent clearly further away.
  tag hint        a very small vote from Smart Tagging possession phases, enough to break ties
                  when the ball is not tracked and nothing else is known.

When the ball is not tracked there is no ball evidence at all, so the state simply carries on
(rule 15): a team is never switched because the ball disappeared. Loose balls, deflections and
contested touches carry little weight against the switching cost, so possession stays with the
previous team unless the other side clearly takes control (rule 16).

After a decoded switch, a confident, uncontested touch by the new side can move the switch back to
that touch when a later same-side event confirms it within the stage-02 window. A detected impulse
assigned to the old side can also anchor a switch when the new side was within reach and sustained
new-side control confirms it within six seconds; only a credible old-side touch blocks this case.

Weights are set from what each piece of evidence means in football, not fitted to any ground
truth; they are listed below with that reasoning.

    python -m pipeline4.s05_possession

Outputs, in work/s05/:

    possession.npz      per-frame arrays (see POSSESSION_ARRAYS)
    spells.csv          one row per possession spell
    turnovers.csv       one row per change of possession inside live play
    backdating.csv      decoded switches moved back to earlier confirmed touches
    s05_labels.csv      X outside the halves, D in stoppages, H/A in live play
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
from pipeline4.s00_timeline import Checks  # noqa: E402
from pipeline4.s01_evidence import load_evidence, runs  # noqa: E402
from pipeline4.s02_touches import (CONFIDENT_SCORE, CONFIRM_CONTROL_FRAMES, FOLLOW_WINDOW,
                                  load_contacts)  # noqa: E402
from pipeline4.s03_stoppages import load_deadball  # noqa: E402
from pipeline4.s04_restarts import load_restarts  # noqa: E402

STAGE = "s05"
F = C.FPS

# --------------------------------------------------------------------------------------------
# parameters (log-odds units)

# Switching possession costs this much evidence. Possession means *controlled* possession
# (CHALLENGE.md), and a possession can last as little as a second, so one controlled reception by
# the other side should be enough. A clean reception (trap, 3.0) plus a moment at the ball, or a close-control spell of
# 6+ frames (>= 3.6), clears 3.5; a lone kick (2.0), a redirect (1.0) or any contested touch
# (<= 1.0) does not, so deflections and loose balls keep the previous possessor (rule 16).
SWITCH_COST = 3.5

# Restart prior: base + confidence of the restarting-side vote (stage 04), capped.
PRIOR_BASE, PRIOR_CAP = 2.0, 6.0

# Touch impulses, applied on the touch frame.
TOUCH_WEIGHT = {"trap": 3.0, "kick": 2.0, "redirect": 1.0}

# A trap is a *reception* only if the receiver is at the ball and keeps it. (a) A trap
# whose nearest player is 3-4 m away is a touch at the edge of reach, not control: its weight
# falls linearly from 1 at RECEPTION_NEAR_M to RECEPTION_MIN_FACTOR at RECEPTION_FAR_M.
# (b) A trap whose very next confident event (confident touch or control spell of 6+ frames,
# within CONFIRM_LOOKAHEAD frames) belongs to the *other* side was not kept: weight x
# UNCONFIRMED_FACTOR. A completed pass still adds its bonus, which is its own confirmation.
RECEPTION_NEAR_M, RECEPTION_FAR_M, RECEPTION_MIN_FACTOR = 3.0, 4.0, 0.25
UNCONFIRMED_FACTOR = 0.5
CONFIRM_LOOKAHEAD = 150
CONFIRM_EVENT_MIN_SCORE = 0.75
CONFIRM_CONTROL_MIN_FRAMES = 6
# A trap anchors possession (see anchor_possession_boundaries) only if it passes the same test:
# the receiver is within the midpoint of the near/far band and the trap is not contradicted by
# the other side's next confident event.
RECEPTION_ANCHOR_MAX_M = 0.5 * (RECEPTION_NEAR_M + RECEPTION_FAR_M)


def confirmation_events(touches, control):
    """Sorted (frames, sides) of the confident events used to confirm a reception."""
    confident = touches[touches["side_score"].abs() >= CONFIRM_EVENT_MIN_SCORE]
    ctl = control[control["frames"] >= CONFIRM_CONTROL_MIN_FRAMES]
    frames = np.concatenate([confident["frame"].to_numpy(), ctl["start"].to_numpy()]).astype(np.int64)
    sides = np.concatenate([confident["side"].to_numpy(), ctl["side"].to_numpy()])
    order = np.argsort(frames, kind="stable")
    return frames[order], sides[order]


def contradicted(frame, side, ev_frames, ev_sides):
    """The very next confident event within CONFIRM_LOOKAHEAD frames is the other side's."""
    k = int(np.searchsorted(ev_frames, int(frame), side="right"))
    return bool(k < len(ev_frames) and ev_frames[k] - int(frame) <= CONFIRM_LOOKAHEAD
                and ev_sides[k] != side)
CONTESTED_FACTOR = 1.0 / 3.0
PASS_BONUS = 2.0              # touch that completes a same-side pass
# Stage 02 scores which side made each touch (side_score, + home). A touch counts in full once
# that score reaches ATTR_FULL_SCORE and proportionally less below it, so a touch whose side is
# uncertain cannot on its own hand possession to the wrong team.
ATTR_FULL_SCORE = 1.5
ATTR_MIN_FACTOR = 0.25

# Per-frame evidence.
CONTROL_PER_FRAME = 0.6       # inside a close-control spell
PROXIMITY_PER_FRAME = 0.1     # nearest player within PROX_DIST_M, opponent PROX_MARGIN_M further
PROX_DIST_M, PROX_MARGIN_M = 1.0, 1.0
TAG_HINT_PER_FRAME = 0.02     # Smart Tagging possession phase
# A goalkeeper hold (stage 02) is control that cannot be challenged: it counts on top of the
# close-control weight, and opponents' proximity is ignored while it lasts.
HOLD_EXTRA_PER_FRAME = 0.4

# Slow-carry evidence, softer proximity, and the tag hint only where nobody is near the ball
CARRY_PER_FRAME = 0.35
PROX2_DIST_M, PROX2_MARGIN_M, PROX2_PER_FRAME = 2.0, 1.5, 0.05
TAG_HINT_FREE_DIST_M = 0.0   # 0 disables the cap
# A clean touch or control spell anchors only when the next confident event confirms it; with no
# later event in the window it needs a strong score
ANCHOR_ALONE_MIN_SCORE = 2.0
KICK_CONTRADICT_FRAMES = 40
ANCHOR_CONTROL_LOOKAHEAD = 75
# A restart from a confident vote keeps the restarting side until the first opposing anchor; a
# virtual restart-taker touch stands in for a strike the tracker never saw; a switch with no evidence
# in between waits for the first event of the new side
RESTART_ANCHOR_MIN_CONF = 1.5
VIRTUAL_TOUCH_W = 2.0
DEFER_MIN_EV, DEFER_MAX = 0.3, 6 * F
# Weak first touch confirmed by a sustained same-side control spell
WEAK_TOUCH_MIN_SCORE, WEAK_CONFIRM_FRAMES, WEAK_LOOKBACK = 0.4, 75, 6 * F
# after a ball gap the first positions are stale: ease the reception band
POST_GAP_MIN, POST_GAP_WINDOW, POST_GAP_EASE_M = 2 * F, 3 * F, 0.0   # easing disabled: first sighting after a gap is often the flight arriving, not a reception

# Special retrospective window for an impulse where the new side was also close enough to
# contest the touch. This changes only an already-decoded switch; it does not change stage-02
# touch attribution or control labels.
AMBIGUOUS_IMPULSE_LOOKBACK = 6 * F

POSSESSION_ARRAYS = {
    "evidence": "(N,) float32  per-frame log-odds for home (+) vs away (-), 0 outside live play",
    "poss_side": "(N,) int8    0 home, 1 away in live play; -1 dead or outside the halves",
    "spell_id": "(N,) int32    row in spells.csv, -1 dead or outside",
    "turnover": "(N,) bool     possession changes side on this frame inside live play",
}


# --------------------------------------------------------------------------------------------
# evidence

def build_evidence(e, c, live):
    n = e.n_frames
    ev = np.zeros(n, dtype=np.float64)
    sign = lambda side: np.where(side == C.HOME, 1.0, np.where(side == C.AWAY, -1.0, 0.0))  # noqa: E731

    # close control and proximity (frame-level)
    ev += CONTROL_PER_FRAME * sign(c.control_side)
    hold = getattr(c, "hold_side", np.full(n, -1, dtype=np.int8))
    ev += HOLD_EXTRA_PER_FRAME * sign(hold)
    near = ((np.nan_to_num(e.nearest_dist, nan=99) <= PROX_DIST_M) &
            (np.nan_to_num(e.nearest_margin, nan=0) >= PROX_MARGIN_M) & (c.control_side < 0) & (hold < 0))
    ev += np.where(near, PROXIMITY_PER_FRAME * sign(e.nearest_side), 0.0)
    near2 = ((np.nan_to_num(e.nearest_dist, nan=99) <= PROX2_DIST_M) & ~near &
             (np.nan_to_num(e.nearest_margin, nan=0) >= PROX2_MARGIN_M) & (c.control_side < 0) & (hold < 0))
    ev += np.where(near2, PROX2_PER_FRAME * sign(e.nearest_side), 0.0)
    carry = getattr(c, "carry_side", np.full(n, -1, dtype=np.int8))
    ev += np.where((c.control_side < 0) & (hold < 0), CARRY_PER_FRAME * sign(carry), 0.0)

    # confident events used to confirm a reception : (frame, side) sorted by frame
    tt = c.touches
    ev_frames, ev_sides = confirmation_events(tt, c.control)

    # touches (impulses)
    for row in tt.itertuples(index=False):
        w = TOUCH_WEIGHT.get(row.kind, 1.0)
        if bool(row.contested):
            w *= CONTESTED_FACTOR
        if row.kind == "trap":
            far = (RECEPTION_FAR_M - float(row.dist_m)) / (RECEPTION_FAR_M - RECEPTION_NEAR_M)
            w *= max(RECEPTION_MIN_FACTOR, min(1.0, far))
            if contradicted(row.frame, row.side, ev_frames, ev_sides):
                w *= UNCONFIRMED_FACTOR
        if bool(row.completes_pass):
            w += PASS_BONUS
        w *= max(ATTR_MIN_FACTOR, min(1.0, abs(float(row.side_score)) / ATTR_FULL_SCORE))
        ev[int(row.frame)] += w if row.side == "H" else -w

    # Smart Tagging possession phases (both-sides frames carry nothing)
    hint = e.tag_poss_hint
    free = ~(np.nan_to_num(e.nearest_dist, nan=99) <= TAG_HINT_FREE_DIST_M)
    ev += np.where(free, np.where(hint == 0, TAG_HINT_PER_FRAME, np.where(hint == 1, -TAG_HINT_PER_FRAME, 0.0)), 0.0)

    ev[~live] = 0.0
    return ev


# --------------------------------------------------------------------------------------------
# decoding

def post_gap_mask(e):
    """True for the first POST_GAP_WINDOW frames of a seen stretch that follows a ball gap >= POST_GAP_MIN."""
    n = e.n_frames
    mask = np.zeros(n, dtype=bool)
    for a, b in runs(e.ball_ok):
        if a == 0:
            continue
        g = 0
        while a - g - 1 >= 0 and not e.ball_ok[a - g - 1] and g < 20 * F:
            g += 1
        if g >= POST_GAP_MIN:
            mask[a:min(b, a + POST_GAP_WINDOW)] = True
    return mask


# Stale possession. The decoder carries the last possessor through a ball gap however long it is,
# and a clean touch normally only anchors possession once the next confident event agrees with it.
# After STALE_GAP_FRAMES or more with the ball unseen, nobody knows who has it, so the first clean
# trap or kick within FOLLOW_WINDOW frames of the ball reappearing anchors its side straight away.
# It still has to be uncontested, confidently attributed and, for a trap, a real reception.
# STALE_GAP_FRAMES is the same 4 s that stage 03 uses for a missing ball. MTAH_STALE=0 turns this off.
STALE_ON = os.environ.get("MTAH_STALE", "1") != "0"
STALE_GAP_FRAMES = 4 * F


def stale_resets(e, a, b):
    """Frames in (a, b) where the ball becomes usable again after a long gap inside the stretch."""
    if not STALE_ON:
        return []
    unknown = ~e.ball_ok[a:b]
    return [int(a + hi) for lo, hi in runs(unknown)
            if hi - lo >= STALE_GAP_FRAMES and lo > 0 and hi < b - a]


def defer_switches(path, evidence):
    """Where the decoder switched with no evidence between the two events, hold the old side until
    the first frame of the new side's evidence (and never past an opposing frame of evidence)."""
    path = path.copy()
    trans = (np.flatnonzero(np.diff(path) != 0) + 1).tolist()
    for i, t in enumerate(trans):
        new = int(path[t])
        old = 1 - new
        sgn = 1.0 if new == C.HOME else -1.0
        tn = trans[i + 1] if i + 1 < len(trans) else len(path)
        end = min(tn, t + DEFER_MAX)
        seg = evidence[t:end] * sgn
        hit = np.flatnonzero(seg >= DEFER_MIN_EV)
        if not len(hit):
            continue
        opp = np.flatnonzero(seg <= -DEFER_MIN_EV)
        f = t + int(hit[0])
        if len(opp) and t + int(opp[0]) < f:
            continue
        if t < f < tn:
            path[t:f] = old
    return path


def viterbi_side(evidence, start_logodds):
    """Two states, H (0) and A (1). Emissions +/- evidence/2, symmetric switch cost."""
    n = len(evidence)
    em = evidence / 2.0
    back = np.zeros((n, 2), dtype=np.int8)
    h = start_logodds / 2.0 + em[0]
    a = -start_logodds / 2.0 - em[0]
    for i in range(1, n):
        sh, xh = h, a - SWITCH_COST
        nh, back[i, 0] = (sh, 0) if sh >= xh else (xh, 1)
        sa, xa = a, h - SWITCH_COST
        na, back[i, 1] = (sa, 1) if sa >= xa else (xa, 0)
        h, a = nh + em[i], na - em[i]
    path = np.zeros(n, dtype=np.int8)
    state = 0 if h >= a else 1
    for i in range(n - 1, -1, -1):
        path[i] = state
        state = back[i, state]
    return path


def backdate_confirmed_switches(path, frame_start, touches, control, hidden_holds=()):
    """Backdate a decoded switch to its earliest supported possession-establishing impulse.

    Ordinary candidate: stage 02 already attributed a reliable touch to the new side and a
    subsequent same-side event confirms it within FOLLOW_WINDOW.
    Special candidate: an earlier detected impulse was attributed to the old side, but the new
    side was also within reach. A later sustained new-side control confirms that contested
    impulse. Old-side control spells alone do not veto this case; a reliable old-side touch does.
    """
    path = path.copy()
    frame_end = frame_start + len(path)
    events = []

    # Keep every touch as an event so an intervening opponent touch blocks confirmation.
    for row in touches.itertuples(index=False):
        frame = int(row.frame)
        if frame_start <= frame < frame_end:
            events.append({
                "frame": frame,
                "side": C.HOME if row.side == "H" else C.AWAY,
                "kind": "touch",
                "row": row,
                "reliable": (not bool(row.contested) and
                             abs(float(row.side_score)) >= CONFIDENT_SCORE),
            })

    # Control starts can confirm a touch, but do not replace the touch as the switch boundary.
    for row in control.itertuples(index=False):
        frame = int(row.start)
        if frame_start <= frame < frame_end:
            events.append({
                "frame": frame,
                "side": C.HOME if row.side == "H" else C.AWAY,
                "kind": "control",
                "row": row,
                "reliable": False,
            })

    events.sort(key=lambda event: event["frame"])
    transitions = (np.flatnonzero(np.diff(path) != 0) + 1).tolist()
    previous_boundary = 0
    corrections = []

    for transition in transitions:
        old_side = int(path[transition - 1])
        new_side = int(path[transition])
        candidates = []
        special_candidates = []

        for i, event in enumerate(events):
            local_frame = event["frame"] - frame_start
            if (event["kind"] != "touch" or
                    not (previous_boundary <= local_frame < transition)):
                continue

            if (event["side"] == new_side and event["reliable"] and
                    str(event["row"].kind) in ("trap", "kick")):
                # Preserve the ordinary short-window rule as-is.
                next_events = [other for other in events[i + 1:]
                               if other["frame"] > event["frame"]]
                if next_events:
                    next_frame = next_events[0]["frame"]
                    next_sides = {other["side"] for other in next_events
                                  if other["frame"] == next_frame}
                    if (next_frame - event["frame"] <= FOLLOW_WINDOW and
                            next_sides == {new_side}):
                        candidates.append((local_frame, next_frame, "reliable_touch", event["side"],
                                           event["row"]))
                continue

            # Special ambiguous impulse: stage 02 detected the ball impulse but assigned it
            # to the old side; the new side was nevertheless within the touch's reach radius.
            row = event["row"]
            opp_dist = float(getattr(row, "opp_dist_m", np.nan))
            reach = float(getattr(row, "reach_m", np.nan))
            if (event["side"] != old_side or not np.isfinite(opp_dist) or
                    not np.isfinite(reach) or opp_dist > reach or
                    frame_start + transition - event["frame"] > AMBIGUOUS_IMPULSE_LOOKBACK):
                continue

            # A sustained new-side control spell must confirm the switch. A later reliable
            # old-side touch before that control blocks this special backdate. Old-side control
            # alone is deliberately not a blocker.
            confirmations = [other for other in events
                             if other["kind"] == "control" and other["side"] == new_side and
                             int(other["row"].end) - int(other["row"].start) + 1 >=
                             CONFIRM_CONTROL_FRAMES and
                             event["frame"] < other["frame"] <= frame_start + transition]
            for confirmation in confirmations:
                confirmed_at = confirmation["frame"]
                # A hidden goalkeeper hold is possession but not observed control at the
                # ball. It confirms that the earlier impulse was the keeper's side's only when
                # that side's player was the nearer one; otherwise the impulse was the shot or
                # pass the keeper then gathered, and the switch stays at the catch.
                if (int(confirmation["row"].start) in hidden_holds and
                        str(getattr(confirmation["row"], "kind", "")) == "hold" and
                        not opp_dist < float(row.dist_m)):
                    continue
                old_touches = [other for other in events
                               if other["kind"] == "touch" and other["side"] == old_side and
                               other["reliable"] and
                               event["frame"] < other["frame"] < confirmed_at]
                if not old_touches:
                    special_candidates.append((local_frame, confirmed_at, "ambiguous_impulse",
                                               event["side"], event["row"]))
                    break

        # weak first touch (airborne / near-tie) of the new side confirmed by sustained new-side control
        for i, event in enumerate(events):
            local_frame = event["frame"] - frame_start
            row = event["row"]
            if (event["kind"] != "touch" or event["side"] != new_side or event["reliable"] or
                    not (previous_boundary <= local_frame < transition) or
                    frame_start + transition - event["frame"] > WEAK_LOOKBACK or bool(row.contested)):
                continue
            sc = float(row.side_score) * (1.0 if new_side == C.HOME else -1.0)
            if sc < WEAK_TOUCH_MIN_SCORE:
                continue
            for other in events:
                if (other["kind"] == "control" and other["side"] == new_side and
                        int(other["row"].end) - int(other["row"].start) + 1 >= CONFIRM_CONTROL_FRAMES and
                        event["frame"] < other["frame"] <= frame_start + transition and
                        other["frame"] - event["frame"] <= WEAK_CONFIRM_FRAMES):
                    blockers = [x for x in events if x["kind"] == "touch" and x["side"] == old_side and
                                x["reliable"] and event["frame"] < x["frame"] < other["frame"]]
                    if not blockers:
                        special_candidates.append((local_frame, other["frame"], "weak_touch_control",
                                                   event["side"], row))
                    break

        accepted = candidates + special_candidates
        if accepted:
            candidate, confirmed_at, kind, candidate_side, candidate_row = min(
                accepted, key=lambda item: item[0])
            if candidate < transition and np.all(path[candidate:transition] == old_side):
                path[candidate:transition] = new_side
                corrections.append({
                    "original_switch_frame": frame_start + transition,
                    "corrected_switch_frame": frame_start + candidate,
                    "confirmation_frame": confirmed_at,
                    "from": "H" if old_side == C.HOME else "A",
                    "to": "H" if new_side == C.HOME else "A",
                    "candidate_type": kind,
                    "candidate_touch_side": "H" if candidate_side == C.HOME else "A",
                    "candidate_touch_kind": candidate_row.kind,
                    "candidate_dv": float(candidate_row.dv),
                    "candidate_opp_dist_m": float(candidate_row.opp_dist_m),
                    "candidate_reach_m": float(candidate_row.reach_m),
                    "frames_backdated": transition - candidate,
                })

        # Later switches cannot reach back across this original Viterbi boundary.
        previous_boundary = transition

    return path, corrections


# First-contact backdating, applied after anchoring. A switch lands on the event that confirms it
# (sustained control, a clean trap or kick), but the new side's possession started at its first
# contact. So each switch is moved back to:
#   (a) the new side's earliest touch of any kind in the FOLLOW_WINDOW frames before it, if no
#       reliable touch or control spell of the old side comes in between. A contested touch only
#       counts if another new-side touch follows it before the switch.
#   (b) then, an unexplained ball impulse (stage 01, ball_orphan_impulse) in the FOLLOW_WINDOW
#       frames before that, if nothing at all touches the ball in between: someone the tracker
#       missed played it, and the next contact is the new side's.
# Neither rule creates a switch. MTAH_FIRST_TOUCH=0 and MTAH_ORPHAN=0 turn (a) and (b) off.
FIRST_TOUCH_ON = os.environ.get("MTAH_FIRST_TOUCH", "1") != "0"
ORPHAN_CREDIT_ON = os.environ.get("MTAH_ORPHAN", "1") != "0"


def backdate_to_first_contact(path, frame_start, touches, control, orphan):
    """Move each final switch back to the new side's first contact. Returns (path, rows)."""
    path = path.copy()
    n = len(path)
    rows = []
    if n < 2 or not (FIRST_TOUCH_ON or ORPHAN_CREDIT_ON):
        return path, rows
    tt = touches[(touches["frame"] >= frame_start) & (touches["frame"] < frame_start + n)]
    tf = tt["frame"].to_numpy().astype(np.int64) - frame_start
    tside = np.where(tt["side"].to_numpy() == "H", C.HOME, C.AWAY)
    tcontested = tt["contested"].to_numpy().astype(bool)
    trel = (~tcontested) & (tt["side_score"].abs().to_numpy() >= CONFIDENT_SCORE)
    ctl = control[(control["frames"] >= CONFIRM_CONTROL_MIN_FRAMES) &
                  (control["end"] >= frame_start) & (control["start"] < frame_start + n)]
    cs = ctl["start"].to_numpy().astype(np.int64) - frame_start
    ce = ctl["end"].to_numpy().astype(np.int64) - frame_start
    cside = np.where(ctl["side"].to_numpy() == "H", C.HOME, C.AWAY)
    og = np.flatnonzero(orphan[frame_start:frame_start + n]) if orphan is not None else np.zeros(0, dtype=np.int64)

    def old_side_block(old, lo, hi):
        """First frame in [lo, hi] not preceded, within [lo, hi), by old-side evidence."""
        block = lo
        m = (tside == old) & trel & (tf >= lo) & (tf < hi)
        if m.any():
            block = max(block, int(tf[m].max()) + 1)
        m = (cside == old) & (ce >= lo) & (cs < hi)
        if m.any():
            block = max(block, min(int(ce[m].max()) + 1, hi))
        return block

    previous = 0
    for transition in (np.flatnonzero(np.diff(path) != 0) + 1).tolist():
        old, new = int(path[transition - 1]), int(path[transition])
        cand, kind, row = transition, None, None
        if FIRST_TOUCH_ON:
            lo = max(previous, transition - FOLLOW_WINDOW)
            block = old_side_block(old, lo, transition)
            m = (tside == new) & (tf >= block) & (tf < transition)
            hits = np.flatnonzero(m)
            # A contested touch on its own does not say who played the ball: it counts as the
            # first contact only when another new-side touch follows before the switch.
            if len(hits) and not (tcontested[hits[0]] and len(hits) == 1):
                k = int(hits[0])
                cand, kind, row = int(tf[k]), "first_touch", tt.iloc[k]
        if ORPHAN_CREDIT_ON and len(og):
            lo = max(previous, cand - FOLLOW_WINDOW)
            block = old_side_block(old, lo, cand)
            earlier = tf[tf < cand]
            if len(earlier):
                block = max(block, int(earlier.max()) + 1)     # no touch at all in between
            m = (og >= block) & (og < cand)
            if m.any():
                cand, kind, row = int(og[m].min()), "orphan_impulse", None
        if kind and cand < transition and np.all(path[cand:transition] == old):
            path[cand:transition] = new
            rows.append({
                "original_switch_frame": frame_start + transition,
                "corrected_switch_frame": frame_start + cand,
                "confirmation_frame": frame_start + transition,
                "from": "H" if old == C.HOME else "A",
                "to": "H" if new == C.HOME else "A",
                "candidate_type": kind,
                "candidate_touch_side": ("H" if new == C.HOME else "A"),
                "candidate_touch_kind": (str(row.kind) if row is not None else ""),
                "candidate_dv": (float(row.dv) if row is not None else np.nan),
                "candidate_opp_dist_m": (float(row.opp_dist_m) if row is not None else np.nan),
                "candidate_reach_m": (float(row.reach_m) if row is not None else np.nan),
                "frames_backdated": transition - cand,
            })
        previous = transition
    return path, rows


def confirmed_anchor(row, ev_frames, ev_sides):
    """A clean touch anchors only if the next confident event does not contradict it.

    A kick (a pass or clearance) is contradicted only by the other side's confident event within
    KICK_CONTRADICT_FRAMES: a ball intercepted that quickly was probably never controlled. Later
    than that the kicker's side did have the ball. A trap (reception) keeps the longer window and,
    with no later event at all, needs a strong score.
    """
    k = int(np.searchsorted(ev_frames, int(row.frame), side="right"))
    gap = int(ev_frames[k]) - int(row.frame) if k < len(ev_frames) else None
    if str(row.kind) == "kick":
        return not (gap is not None and gap <= KICK_CONTRADICT_FRAMES and ev_sides[k] != row.side)
    if gap is not None and gap <= CONFIRM_LOOKAHEAD:
        return ev_sides[k] == row.side
    return abs(float(row.side_score)) >= ANCHOR_ALONE_MIN_SCORE


def anchor_possession_boundaries(path, frame_start, touches, control, corrections, restart=None,
                                 resets=()):
    """Use reliable touches and sustained control as the boundaries of possession changes.

    Keep the Viterbi timeline wherever it agrees with the event anchors. When it conflicts, hold
    the anchored side until the next opposing clean trap/kick, sustained control, goalkeeper hold,
    or accepted retrospective correction. This prevents redirects or accumulated proximity alone
    from moving a boundary ahead of the next possession-establishing event.
    """
    path = path.copy()
    if not len(path):
        return path
    frame_end = frame_start + len(path)
    anchors = {}
    backdate_locks = []

    def add_anchor(frame, side, priority):
        frame = int(frame)
        if not frame_start <= frame < frame_end:
            return
        current = anchors.get(frame)
        if current is None or priority > current[1]:
            anchors[frame] = (int(side), priority)

    # A backdate is an explicit accepted switch boundary, including the ambiguous-impulse case
    # where stage 02 attributed the candidate contact to the previous side.
    for correction in corrections:
        side = C.HOME if correction["to"] == "H" else C.AWAY
        add_anchor(correction["corrected_switch_frame"], side, 3)
        if correction["candidate_type"] == "ambiguous_impulse":
            old_side = C.HOME if correction["from"] == "H" else C.AWAY
            backdate_locks.append((correction["corrected_switch_frame"],
                                   correction["confirmation_frame"], old_side))

    # A clean trap or kick is a team touch that can establish possession. Redirects are excluded:
    # they are often just deflections.
    ev_frames, ev_sides = confirmation_events(touches, control)
    for row in touches.itertuples(index=False):
        if (str(row.kind) in ("trap", "kick") and not bool(row.contested) and
                abs(float(row.side_score)) >= CONFIDENT_SCORE):
            if str(row.kind) == "trap" and float(row.dist_m) > RECEPTION_ANCHOR_MAX_M:
                continue                      # not a reception, so it cannot anchor
            if not confirmed_anchor(row, ev_frames, ev_sides):
                continue                      # not confirmed by the next confident event
            side = C.HOME if row.side == "H" else C.AWAY
            add_anchor(row.frame, side, 1)

    # A weak first touch (airborne ball, near tie) of one side, followed within WEAK_CONFIRM_FRAMES by
    # that side's sustained control with no reliable opposing touch in between, starts the possession
    ctl_big = control[control["frames"] >= CONFIRM_CONTROL_FRAMES]
    rel_opp = touches[(touches["side_score"].abs() >= CONFIDENT_SCORE) & (~touches["contested"].astype(bool))]
    for row in touches.itertuples(index=False):
        sc = float(row.side_score) * (1.0 if row.side == "H" else -1.0)
        if bool(row.contested) or abs(float(row.side_score)) >= CONFIDENT_SCORE or sc < WEAK_TOUCH_MIN_SCORE:
            continue
        cc = ctl_big[(ctl_big["side"] == row.side) & (ctl_big["start"] > row.frame) &
                     (ctl_big["start"] - row.frame <= WEAK_CONFIRM_FRAMES)]
        if not len(cc):
            continue
        stop = int(cc["start"].iloc[0])
        if ((rel_opp["side"] != row.side) & (rel_opp["frame"] > row.frame) & (rel_opp["frame"] < stop)).any():
            continue
        add_anchor(row.frame, C.HOME if row.side == "H" else C.AWAY, 1)

    # Sustained control (including goalkeeper holds) can establish or restore possession.
    for row in control.itertuples(index=False):
        is_hold = str(getattr(row, "kind", "close_control")) == "hold"
        if is_hold or int(row.frames) >= CONFIRM_CONTROL_FRAMES:
            side = C.HOME if row.side == "H" else C.AWAY
            if not is_hold:
                k = int(np.searchsorted(ev_frames, int(row.end), side="right"))
                if (k < len(ev_frames) and ev_frames[k] - int(row.end) <= ANCHOR_CONTROL_LOOKAHEAD
                        and ev_sides[k] != row.side):
                    continue                  # the other side's next event contradicts it
            if any(start <= int(row.start) < end and side == old_side
                   for start, end, old_side in backdate_locks):
                continue
            add_anchor(row.start, side, 2)

    # Stale possession: the first clean trap or kick after a long unseen gap anchors its side.
    for r0 in resets:
        first = touches[(touches["frame"] >= r0) & (touches["frame"] <= r0 + FOLLOW_WINDOW) &
                        touches["kind"].isin(["trap", "kick"]) & (~touches["contested"].astype(bool)) &
                        (touches["side_score"].abs() >= CONFIDENT_SCORE)]
        first = first[(first["kind"] != "trap") | (first["dist_m"] <= RECEPTION_ANCHOR_MAX_M)]
        if len(first):
            row = first.sort_values("frame").iloc[0]
            add_anchor(row.frame, C.HOME if row.side == "H" else C.AWAY, 1)

    if restart is not None:
        add_anchor(restart[0], restart[1], 1)   # the confident restarting side holds until an opposing anchor

    if not anchors:
        return path

    result = path.copy()
    ordered = sorted((frame - frame_start, side)
                     for frame, (side, _) in anchors.items())
    for i, (local_frame, side) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(result)
        if end > local_frame and np.any(result[local_frame:end] != side):
            result[local_frame:end] = side
    return result


def segment_prior(r, a):
    """Start log-odds for a live stretch beginning at frame a, from its restart."""
    side = int(r.segment_prior_side[a])
    if side < 0:
        return 0.0, "", 0.0
    row = r.restarts[(r.restarts["restart"] <= a) & (r.restarts["restart"] >= a - F) &
                     (r.restarts["type"] != "end_of_half")]
    conf = float(row["confidence"].iloc[-1]) if len(row) else 0.0
    strength = min(PRIOR_BASE + conf, PRIOR_CAP)
    return (strength if side == C.HOME else -strength), ("H" if side == C.HOME else "A"), conf


# --------------------------------------------------------------------------------------------
# main

def build(work_dir=None):
    t0 = time.time()
    checks = Checks()
    e = load_evidence(work_dir)
    c = load_contacts(work_dir)
    d = load_deadball(work_dir)
    r = load_restarts(work_dir)
    n = e.n_frames
    g = e.graded
    live = g & ~d.dead
    print("loaded s01-s04: {} live frames in {} live stretches".format(int(live.sum()), len(runs(live))))

    # stale first positions after a ball gap: ease the reception band (in memory only)
    pg = post_gap_mask(e)
    tt0 = c.touches.copy()
    ease = pg[tt0["frame"].to_numpy()] & (tt0["kind"] == "trap").to_numpy()
    tt0.loc[ease, "dist_m"] = np.maximum(0.0, tt0.loc[ease, "dist_m"] - POST_GAP_EASE_M)
    c.touches = tt0
    evidence = build_evidence(e, c, live)
    # a virtual restart-taker touch where the strike was not seen
    for a0, b0 in runs(live):
        pr, ps, cf = segment_prior(r, a0)
        if ps and cf >= RESTART_ANCHOR_MIN_CONF / 1.5:
            sd = C.HOME if ps == "H" else C.AWAY
            lo, hi = max(0, a0 - F), min(n, a0 + F)
            if not (c.touch_side[lo:hi] == sd).any():
                w = VIRTUAL_TOUCH_W * max(0.25, min(1.0, cf / 2.0))
                evidence[a0] += w if sd == C.HOME else -w
    poss = np.full(n, -1, dtype=np.int8)
    seg_rows = []
    backdating_rows = []
    n_stale_resets = 0
    # hidden goalkeeper holds (ball unseen); see backdate_confirmed_switches
    hidden_holds = set()
    holds_tbl = getattr(c, "holds", None)
    if holds_tbl is not None and len(holds_tbl) and "virtual" in holds_tbl.columns:
        hidden_holds = set(holds_tbl.loc[holds_tbl["virtual"].astype(bool), "start"].astype(int))
    for k, (a, b) in enumerate(runs(live)):
        prior, prior_side, conf = segment_prior(r, a)
        resets = stale_resets(e, a, b)
        n_stale_resets += len(resets)
        path = viterbi_side(evidence[a:b], prior)
        path = defer_switches(path, evidence[a:b])
        path, corrections = backdate_confirmed_switches(path, a, c.touches, c.control, hidden_holds)
        ranchor = None
        if prior_side and conf >= RESTART_ANCHOR_MIN_CONF:
            ranchor = (a, C.HOME if prior_side == "H" else C.AWAY)
        path = anchor_possession_boundaries(path, a, c.touches, c.control, corrections, ranchor, resets)
        path, first_contact = backdate_to_first_contact(
            path, a, c.touches, c.control, getattr(e, "ball_orphan_impulse", None))
        corrections = corrections + first_contact
        poss[a:b] = path
        seg_rows.append((k, a, b, prior_side, conf))
        for correction in corrections:
            correction["live_stretch"] = k
            backdating_rows.append(correction)

    print("long unseen gaps inside live play (stale possession): {}".format(n_stale_resets))

    # spells and turnovers
    spells, turnovers = [], []
    spell_id = np.full(n, -1, dtype=np.int32)
    turnover = np.zeros(n, dtype=bool)
    tt = c.touches
    for k, a, b, prior_side, conf in seg_rows:
        seg = poss[a:b]
        change = np.flatnonzero(np.diff(seg) != 0) + 1
        bounds = [0] + change.tolist() + [b - a]
        for j in range(len(bounds) - 1):
            s0, s1 = a + bounds[j], a + bounds[j + 1]
            side = "H" if poss[s0] == C.HOME else "A"
            in_spell = tt[(tt["frame"] >= s0) & (tt["frame"] < s1)]
            spell_id[s0:s1] = len(spells)
            spells.append({
                "start": s0, "end": s1 - 1, "frames": s1 - s0, "duration_s": round((s1 - s0) / F, 2),
                "half": int(e.half[s0]), "side": side,
                "starts_with": "restart" if j == 0 else "turnover",
                "restart_side": prior_side if j == 0 else "",
                "restart_overturned": bool(j == 0 and prior_side and prior_side != side),
                "touches_own": int((in_spell["side"] == side).sum()),
                "touches_opp": int((in_spell["side"] != side).sum()),
                "passes_own": int(((in_spell["side"] == side) & in_spell["completes_pass"]).sum()),
                "control_frames": int((c.control_side[s0:s1] == poss[s0]).sum()),
                "ball_tracked_pct": round(100 * float(e.ball_ok[s0:s1].mean()), 1),
                "evidence_sum": round(float(evidence[s0:s1].sum() * (1 if side == "H" else -1)), 2),
            })
            if j > 0:
                turnover[s0] = True
                turnovers.append({"frame": s0, "half": int(e.half[s0]),
                                  "from": "A" if side == "H" else "H", "to": side,
                                  "ball_x_m": round(float(e.ball_m[s0, 0]), 2) if e.ball_ok[s0] else np.nan,
                                  "ball_y_m": round(float(e.ball_m[s0, 1]), 2) if e.ball_ok[s0] else np.nan,
                                  "evidence_next_2s": round(float(evidence[s0:s0 + 2 * F].sum()), 2)})
    spells = pd.DataFrame(spells)
    turnovers = pd.DataFrame(turnovers, columns=["frame", "half", "from", "to", "ball_x_m", "ball_y_m",
                                                 "evidence_next_2s"])
    backdating = pd.DataFrame(backdating_rows, columns=[
        "live_stretch", "original_switch_frame", "corrected_switch_frame", "confirmation_frame",
        "from", "to", "candidate_type", "candidate_touch_side", "candidate_touch_kind",
        "candidate_dv", "candidate_opp_dist_m", "candidate_reach_m", "frames_backdated",
    ])

    labels = np.where(~g, "X", np.where(d.dead, "D", np.where(poss == C.HOME, "H",
                                                               np.where(poss == C.AWAY, "A", "X"))))

    # -- checks ---------------------------------------------------------------------------------
    checks.add("every live frame has a side", bool(np.all(poss[live] >= 0)))
    checks.add("no side on dead or ungraded frames", bool(np.all(poss[~live] == -1)))
    checks.add("labels are X outside the halves", bool(np.all(labels[~g] == "X")))
    checks.add("labels have no X inside the halves", bool(np.all(labels[g] != "X")),
               "{} X frames inside".format(int((labels[g] == "X").sum())))
    graded_min = g.sum() / F / 60.0
    n_spells = len(spells)
    checks.add("possession spells per graded minute", True,
               "{} spells ({:.2f}/min), median {:.1f} s".format(
                   n_spells, n_spells / graded_min, float(spells["duration_s"].median()) if n_spells else 0),
               severity="warning")
    over = int(spells["restart_overturned"].sum()) if n_spells else 0
    checks.add("restarts overturned by evidence (mishit / intercepted)", True,
               "{} of {} live stretches".format(over, len(seg_rows)), severity="warning")
    print("\nchecks")
    checks.report()
    if checks.failed("error"):
        raise SystemExit("\ncritical check failed; nothing written.")

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    np.savez(os.path.join(out, "possession.npz"), evidence=evidence.astype(np.float32),
             poss_side=poss, spell_id=spell_id, turnover=turnover)
    spells.to_csv(os.path.join(out, "spells.csv"), index=False)
    turnovers.to_csv(os.path.join(out, "turnovers.csv"), index=False)
    backdating.to_csv(os.path.join(out, "backdating.csv"), index=False)
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(os.path.join(out, "s05_labels.csv"), index=False)
    home_live = float((poss[live] == C.HOME).mean()) if live.any() else 0.0
    stats = {
        "live_stretches": len(seg_rows),
        "spells": n_spells,
        "spells_per_graded_minute": round(n_spells / graded_min, 2),
        "spell_duration_median_s": round(float(spells["duration_s"].median()), 2) if n_spells else 0,
        "spell_duration_p10_p90_s": [round(float(spells["duration_s"].quantile(q)), 2) for q in (0.1, 0.9)] if n_spells else [],
        "turnovers_in_live_play": int(len(turnovers)),
        "backdated_switches": int(len(backdating)),
        "frames_backdated": int(backdating["frames_backdated"].sum()) if len(backdating) else 0,
        "restarts_overturned": over,
        "home_share_of_live_play": round(home_live, 3),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels[g], return_counts=True))},
    }
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE,
        "parameters": {k: v for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, tuple, str, dict))},
        "stats": stats, "arrays": POSSESSION_ARRAYS, "checks": checks.items,
    })
    print("\nstats")
    for k, v in stats.items():
        print("  {:<28} {}".format(k, v))
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


class Possession(object):
    """Read-only view of work/s05: p.poss_side, p.spells, p.turnovers ..."""

    def __init__(self, work_dir=None):
        folder = os.path.join(work_dir or C.WORK_DIR, STAGE)
        path = os.path.join(folder, "possession.npz")
        if not os.path.isfile(path):
            raise IOError("{} not found -- run: python -m pipeline4.s05_possession".format(path))
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.spells = pd.read_csv(os.path.join(folder, "spells.csv"), keep_default_na=False)
        self.turnovers = pd.read_csv(os.path.join(folder, "turnovers.csv"))
        self.meta = C.read_json(os.path.join(folder, "meta.json"))


def load_possession(work_dir=None):
    return Possession(work_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 05: possession inside live play.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    args = parser.parse_args(argv)
    build(args.work_dir)


if __name__ == "__main__":
    main()
