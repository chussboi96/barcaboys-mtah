"""Stage 02 -- touches, close control and a first-pass possession estimate (rule 14).

Two kinds of contact evidence are extracted from the stage-01 evidence, inside the graded halves:

  touch          an impulse on the ball -- its velocity changes by at least TOUCH_MIN_DV between
                 the IMPULSE_SPAN frames before and the IMPULSE_SPAN after -- with a player within
                 TOUCH_RADIUS_M. One event per impulse (non-maximum suppression). A touch with
                 players of both sides within reach is marked contested.
                 The side that made it is a scored attribution, not just the nearest player:
                 proximity over a window, the direction the ball is struck or received in, close
                 control just before a kick or just after a reception, and the pass chain it sits
                 in (see the attribution parameters). The score is kept (side_score) so later
                 stages can weigh uncertain touches less.
  control spell  a run of frames where one side's player is within CONTROL_RADIUS_M of the ball,
                 moving with it (relative speed below CONTROL_MAX_REL_SPEED) and clearly closer
                 than any opponent. That is a reception, a dribble or a shielded ball.

From them, per frame:

  last contact   side, frame and contested flag of the most recent contact (touch or control) in
                 the same half. Stage 03 uses this for the last-touch rule at the touchline and
                 goal line.
  pass 1 (p1)    a rough possessing side. It switches only when the other side's contact is
                 confirmed -- a control spell of CONFIRM_CONTROL_FRAMES or more, or a contact
                 followed by another contact of the same side within FOLLOW_WINDOW frames with
                 no opponent contact in between. It knows nothing about dead balls yet. It is
                 input for stages 03-05, not a submission.

Thresholds are physical (documented below), not fitted to any ground truth.

    python -m pipeline4.s02_touches

Outputs, in work/s02/:

    contacts.npz        per-frame arrays (see CONTACT_ARRAYS)
    touches.csv         one row per touch
    control_spells.csv  one row per control spell (kind: close_control or hold)
    holds.csv           one row per goalkeeper hold in the own penalty area
    p1_labels.csv       pass 1 in submission format (X outside the halves) -- a diagnostic only
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

STAGE = "s02"

# --------------------------------------------------------------------------------------------
# parameters -- physical, not fitted

# Impulse window: ball velocity over the 8 frames (0.32 s) before vs the 8 after. The delivered
# ball track is heavily smoothed (median second difference ~1 cm per frame^2), so a kick's
# change of velocity is spread over several frames; a short window sees only a fraction of it.
IMPULSE_SPAN = 8
# A pass or a trap changes the ball's velocity by 5-20 m/s; 3 m/s is above what the smoothed
# track produces in free flight and below the softest deliberate touch.
TOUCH_MIN_DV = 3.0
# Tracked positions are body centres; a leg reaches ~1 m, and the smoothed ball lags the real one
# by a few tens of cm at an impulse, so a touching player is within 2 m of the tracked ball.
TOUCH_RADIUS_M = 2.0
# Distances are taken as the minimum over +/- 2 frames, because the impulse peak and the closest
# approach of the player need not fall on the same frame.
TOUCH_NEAR_WINDOW = 2
# Two impulses closer than this (0.32 s) are the same touch.
TOUCH_NMS_FRAMES = 8

# Touch attribution (which side made the touch). The nearest player at the impulse alone is not
# enough: the delivered ball is smoothed and lags the real one, and a pressing opponent is often
# closer to it than the kicker. Each cue adds a signed score (+ home, - away):
#   proximity      closest player of each side over +/- ATTR_WINDOW frames      up to 1.5
#   kick direction a ball struck >= KICK_DIR_MIN_SPEED towards one side's       1.0 (1.5 if
#                  attacking goal was most likely struck by that side            >= 15 m/s)
#   receive dir.   a moving ball stopped by a side usually came from a           0.5
#                  teammate, i.e. was travelling in that side's attacking direction
#   continuity     a close-control spell of one side right before a kick (the     1.0
#                  kicker had it) or right after a trap (the receiver keeps it),
#                  within ATTR_CONTINUITY frames and CONTINUITY_RADIUS_M
#   pass chain     an uncertain touch between two confident touches of one side   1.0
#                  within PASS_CHAIN_GAP frames each way (rule 18)
ATTR_WINDOW = 4
W_PROXIMITY, PROXIMITY_SCALE_M = 1.5, 3.0
KICK_DIR_MIN_SPEED, KICK_DIR_MIN_ADV = 8.0, 5.0
W_KICK_DIR, W_KICK_DIR_LONG, LONG_KICK_SPEED = 1.0, 1.5, 15.0
W_RECEIVE_DIR, RECEIVE_MIN_SPEED = 0.5, 5.0
W_CONTINUITY, ATTR_CONTINUITY, CONTINUITY_RADIUS_M = 1.0, 25, 4.0
W_PASS_CHAIN, PASS_CHAIN_GAP, CONFIDENT_SCORE = 1.0, 75, 0.75
# A touch is contested when both sides were within reach and the attribution stays uncertain.
CONTESTED_MAX_SCORE = 1.0

# Reach grows with ball speed. The tracked ball lags the real one by roughly 0.15 s when
# it moves fast, so a player striking or receiving a 12 m/s ball can be ~3.8 m from the tracked
# position. Reach = TOUCH_RADIUS_M + TOUCH_LAG_S x ball speed, capped at TOUCH_RADIUS_MAX_M.
TOUCH_LAG_S = 0.15
TOUCH_RADIUS_MAX_M = 4.0

# Interceptions.
#   The "received from a teammate" cue only applies when the pass came from a touch at most
#   RECEIVE_MAX_ORIGIN_M away within RECEIVE_ORIGIN_WINDOW frames; long balls are exactly the
#   passes that get intercepted, and a ball of unknown origin says nothing.
#   What happens after a reception is evidence of who received it: the next confident contact
#   (touch or control spell) within NEXT_CONTACT_WINDOW frames, beyond the continuity window,
#   and the direction the ball is carried over the next 1.5 s.
#   A pass counts as completed only when both touches are confidently attributed.
RECEIVE_MAX_ORIGIN_M, RECEIVE_ORIGIN_WINDOW = 25.0, 150
W_NEXT_CONTACT, NEXT_CONTACT_WINDOW = 0.75, 75
W_NEXT_DIR, NEXT_DIR_FROM, NEXT_DIR_TO, NEXT_DIR_MIN_M = 0.5, 3, 40, 3.0

# Goalkeeper hold. The ball nearly still (1-s mean speed <= HOLD_MAX_SPEED) inside a
# team's own penalty area with one of that team's players within HOLD_RADIUS_M for at least
# HOLD_MIN_FRAMES is held by that team -- typically the goalkeeper. Under the Laws of the Game a
# goalkeeper in control of the ball with the hands cannot be challenged, so during the hold
# opponents' proximity and close control count for nothing. The hold lasts until the release:
# the first frame within RELEASE_SEARCH where the ball moves at RELEASE_MIN_SPEED or more
# (throw, kick, roll out). A touch inside the hold or within RELEASE_GRACE frames of the release
# is credited to the holding side with W_HOLD.
HOLD_MAX_SPEED, HOLD_RADIUS_M, HOLD_MIN_FRAMES, HOLD_MAX_HOLE = 1.5, 1.5, 25, 5
RELEASE_SEARCH, RELEASE_MIN_SPEED, RELEASE_GRACE = 75, 5.0, 12
W_HOLD = 3.0

# Close control: within 1 m of the ball, moving with it, and the nearest opponent at least 1 m
# further away.
CONTROL_RADIUS_M = 1.0
CONTROL_MAX_REL_SPEED = 3.0
CONTROL_MIN_MARGIN_M = 1.0
# A spell must last 0.24 s; holes of up to 2 frames inside one are closed.
CONTROL_MIN_FRAMES = 6
CONTROL_MAX_HOLE = 2

# Pass-1 confirmation of a change of side.
# Slow carry: a player carrying the ball slowly (dribble, slow build-up) is never an impulse and rarely within
# the strict close-control radius. A looser "carry" spell supplies live-ball and possession evidence
# only; it never anchors a possession boundary.
CARRY_RADIUS_M, CARRY_MIN_MARGIN_M, CARRY_MAX_REL_SPEED = 1.5, 0.7, 2.0
CARRY_MIN_FRAMES, CARRY_MAX_HOLE = 25, 5
CARRY_MIN_BALL_SPEED = 0.7    # a ball that is not moving is placed or lying, not carried
# A touch is also contested when both sides are this close with almost the same distance, whatever
# the direction/next-contact cues say
CONTESTED_BOTH_M, CONTESTED_DIFF_M = 1.5, 0.5
# Hidden goalkeeper hold: the ball vanishes inside a penalty area right after a fast approach.
VHOLD_MIN_GAP, VHOLD_MAX_GAP = 2 * C.FPS, 20 * C.FPS
VHOLD_MIN_APPROACH, VHOLD_DEFENDER_M = 5.0, 8.0
CONFIRM_CONTROL_FRAMES = 10   # 0.4 s of close control establishes possession on its own
FOLLOW_WINDOW = 75            # or a second same-side contact within 3 s, no opponent between

SIDE_NAMES = np.array(["H", "A"])

CONTACT_ARRAYS = {
    "ball_dv": "(N,) float32       |v_after - v_before| over IMPULSE_SPAN frames each side, m/s",
    "ball_speed_before": "(N,) float32", "ball_speed_after": "(N,) float32",
    "touch": "(N,) bool          a touch event is centred on this frame",
    "touch_side": "(N,) int8     side of that touch, -1 none",
    "control_side": "(N,) int8   side in a control spell or a hold on this frame, -1 none",
    "carry_side": "(N,) int8     side slowly carrying the ball, outside control spells, -1 none",
    "hold_side": "(N,) int8      side holding the ball in its own penalty area (goalkeeper), -1 none",
    "contact_side": "(N,) int8   touch or control side on this frame, -1 none",
    "last_contact_side": "(N,) int8 side of the latest contact at or before this frame, same half",
    "last_contact_frame": "(N,) int32 frame of that contact, -1 none",
    "last_contact_contested": "(N,) bool that contact was a contested touch",
    "p1_side": "(N,) int8        pass-1 possessing side, -1 undecided or outside the halves",
    "p1_change": "(N,) bool      pass 1 switches side on this frame",
}


# --------------------------------------------------------------------------------------------
# helpers

def shifted_min(values, half_window):
    """Min over +/- half_window frames (NaN treated as +inf)."""
    v = np.where(np.isfinite(values), values, np.inf)
    out = v.copy()
    n = len(v)
    for k in range(1, half_window + 1):
        out[k:] = np.minimum(out[k:], v[:n - k])
        out[:n - k] = np.minimum(out[:n - k], v[k:])
    return out


def is_local_max(values, half_window):
    """True where values is the first maximum of its +/- half_window neighbourhood."""
    n = len(values)
    v = np.where(np.isfinite(values), values, -np.inf)
    keep = np.isfinite(values).copy()
    for k in range(1, half_window + 1):
        keep[k:] &= v[k:] > v[:n - k]        # strictly greater than earlier neighbours
        keep[:n - k] &= v[:n - k] >= v[k:]   # at least later neighbours
    return keep


def impulse(ball_m, ball_ok, run_id, span):
    """|v_after - v_before|, |v_before|, |v_after| and the two velocity vectors (m/s)."""
    n = len(ball_m)
    dv = np.full(n, np.nan, dtype=np.float32)
    sb = np.full(n, np.nan, dtype=np.float32)
    sa = np.full(n, np.nan, dtype=np.float32)
    vb_all = np.full((n, 2), np.nan, dtype=np.float32)
    va_all = np.full((n, 2), np.nan, dtype=np.float32)
    i = np.arange(span, n - span)
    a, b = i - span, i + span
    good = ball_ok[a] & ball_ok[i] & ball_ok[b] & (run_id[a] == run_id[i]) & (run_id[i] == run_id[b])
    i, a, b = i[good], a[good], b[good]
    vb = (ball_m[i] - ball_m[a]) * (C.FPS / float(span))
    va = (ball_m[b] - ball_m[i]) * (C.FPS / float(span))
    dv[i] = np.linalg.norm(va - vb, axis=1)
    sb[i] = np.linalg.norm(vb, axis=1)
    sa[i] = np.linalg.norm(va, axis=1)
    vb_all[i], va_all[i] = vb, va
    return dv, sb, sa, vb_all, va_all


# --------------------------------------------------------------------------------------------
# touches

def side_distance_window(e, frames, half_window):
    """(len(frames), 2): closest eligible player of each side to the ball over +/- half_window."""
    out = np.full((len(frames), 2), np.inf)
    for k in range(-half_window, half_window + 1):
        f = np.clip(frames + k, 0, e.n_frames - 1)
        d = np.where(np.isfinite(e.near_dist[f]), e.near_dist[f], np.inf)
        out = np.minimum(out, d)
    return out


def attack_sign(e, home_attacks_positive_x):
    """(N,) +1 where home attacks +x, -1 where it attacks -x, 0 outside the halves."""
    sign = np.zeros(e.n_frames, dtype=np.float64)
    for h in (1, 2):
        sign[e.half == h] = 1.0 if home_attacks_positive_x[str(h)] else -1.0
    return sign


def hold_lookup(holds, frames, e):
    """(len(frames),) side (+1 home, -1 away, 0 none) of a hold covering each touch frame."""
    out = np.zeros(len(frames))
    for row in holds.itertuples(index=False):
        sgn = 1.0 if row.side == "H" else -1.0
        inside = (frames >= row.start) & (frames <= row.release + RELEASE_GRACE)
        out[inside] = sgn
    return out


def detect_touches(e, dv, sb, sa, vb, va, control, holds, home_attacks_positive_x):
    near = np.stack([shifted_min(e.near_dist[:, s], TOUCH_NEAR_WINDOW) for s in (0, 1)], axis=1)
    speed = np.fmax(np.nan_to_num(sb, nan=0.0), np.nan_to_num(sa, nan=0.0))
    radius = np.minimum(TOUCH_RADIUS_MAX_M, TOUCH_RADIUS_M + TOUCH_LAG_S * speed)
    reach = near <= radius[:, None]
    candidate = e.graded & e.ball_ok & (dv >= TOUCH_MIN_DV) & reach.any(axis=1)
    peak = candidate & is_local_max(np.where(candidate, dv, np.nan), TOUCH_NMS_FRAMES)
    frames = np.flatnonzero(peak)
    n_t = len(frames)
    half = e.half[frames]
    bx, by = e.ball_m[frames, 0], e.ball_m[frames, 1]

    gain = sa[frames] - sb[frames]
    kind = np.where(gain >= TOUCH_MIN_DV, "kick", np.where(gain <= -TOUCH_MIN_DV, "trap", "redirect"))
    adv_sign_all = attack_sign(e, home_attacks_positive_x)
    adv_sign = adv_sign_all[frames]

    # goalkeeper hold: touches inside a hold or at its release belong to the holder
    hold = W_HOLD * hold_lookup(holds, frames, e)

    # 1. proximity over a window -- ignored during a hold, when opponents cannot challenge
    dwin = side_distance_window(e, frames, ATTR_WINDOW)
    diff = np.where(np.isfinite(dwin).all(axis=1), dwin[:, 1] - dwin[:, 0],
                    np.where(np.isfinite(dwin[:, 0]), PROXIMITY_SCALE_M, -PROXIMITY_SCALE_M))
    prox = W_PROXIMITY * np.clip(diff, -PROXIMITY_SCALE_M, PROXIMITY_SCALE_M) / PROXIMITY_SCALE_M
    prox = np.where(hold != 0, 0.0, prox)

    # 2. direction of play
    adv_out = va[frames, 0] * adv_sign
    adv_in = vb[frames, 0] * adv_sign
    struck = (sa[frames] >= KICK_DIR_MIN_SPEED) & (kind != "trap") & (np.abs(adv_out) >= KICK_DIR_MIN_ADV)
    w_kick = np.where(sa[frames] >= LONG_KICK_SPEED, W_KICK_DIR_LONG, W_KICK_DIR)
    direction = np.where(struck, w_kick * np.sign(adv_out), 0.0)
    # received from a teammate: only for short passes of known origin
    short_origin = np.zeros(n_t, dtype=bool)
    for k in range(1, n_t):
        if half[k] == half[k - 1] and frames[k] - frames[k - 1] <= RECEIVE_ORIGIN_WINDOW:
            short_origin[k] = np.hypot(bx[k] - bx[k - 1], by[k] - by[k - 1]) <= RECEIVE_MAX_ORIGIN_M
    received = ((kind == "trap") & (sb[frames] >= RECEIVE_MIN_SPEED) &
                (np.abs(adv_in) >= KICK_DIR_MIN_ADV) & short_origin)
    direction += np.where(received, W_RECEIVE_DIR * np.sign(adv_in), 0.0)
    direction = np.nan_to_num(direction)

    # 3. continuity with close control just before a kick / just after a trap
    cont = np.zeros(n_t)
    cs = control["start"].to_numpy() if len(control) else np.array([], dtype=int)
    ce = control["end"].to_numpy() if len(control) else np.array([], dtype=int)
    cside = np.where(control["side"].to_numpy() == "H", 1.0, -1.0) if len(control) else np.array([])
    if len(control):
        cx, cy = control["ball_x_m"].to_numpy(), control["ball_y_m"].to_numpy()
        ex = np.array([float(np.nanmean(e.ball_m[a:b + 1, 0])) for a, b in zip(cs, ce)])
        ey = np.array([float(np.nanmean(e.ball_m[a:b + 1, 1])) for a, b in zip(cs, ce)])
        for k, f in enumerate(frames):
            if kind[k] != "trap":
                sel = (ce <= f) & (ce >= f - ATTR_CONTINUITY) & (np.hypot(ex - bx[k], ey - by[k]) <= CONTINUITY_RADIUS_M)
                if sel.any():
                    cont[k] += W_CONTINUITY * cside[np.flatnonzero(sel)[-1]]
            if kind[k] != "kick":
                sel = (cs >= f) & (cs <= f + ATTR_CONTINUITY) & (np.hypot(cx - bx[k], cy - by[k]) <= CONTINUITY_RADIUS_M)
                if sel.any():
                    cont[k] += W_CONTINUITY * cside[np.flatnonzero(sel)[0]]

    pre = prox + direction + cont + hold

    # 4. what happens after a reception
    nxt = np.zeros(n_t)
    for k, f in enumerate(frames):
        if kind[k] == "kick":
            continue
        later = [j for j in range(k + 1, n_t) if half[j] == half[k] and frames[j] - f <= NEXT_CONTACT_WINDOW
                 and frames[j] - f > ATTR_CONTINUITY and abs(pre[j]) >= CONFIDENT_SCORE]
        cand = [(int(frames[later[0]]), float(np.sign(pre[later[0]])))] if later else []
        if len(control):
            sel = np.flatnonzero((cs > f + ATTR_CONTINUITY) & (cs <= f + NEXT_CONTACT_WINDOW))
            if len(sel):
                cand.append((int(cs[sel[0]]), float(cside[sel[0]])))
        if cand:
            nxt[k] += W_NEXT_CONTACT * sorted(cand)[0][1]
        a, b = f + NEXT_DIR_FROM, f + NEXT_DIR_TO
        if b < e.n_frames and e.ball_ok[a] and e.ball_ok[b] and e.half[a] == e.half[b] == half[k]:
            carried = (e.ball_m[b, 0] - e.ball_m[a, 0]) * adv_sign[k]
            if abs(carried) >= NEXT_DIR_MIN_M:
                nxt[k] += W_NEXT_DIR * np.sign(carried)
    score = pre + nxt

    # 5. pass chain (rule 18): an uncertain touch between two confident touches of one side
    chain = np.zeros(n_t)
    for k in range(1, n_t - 1):
        if abs(score[k]) >= CONFIDENT_SCORE:
            continue
        p, q = k - 1, k + 1
        if half[p] != half[k] or half[q] != half[k]:
            continue
        if frames[k] - frames[p] > PASS_CHAIN_GAP or frames[q] - frames[k] > PASS_CHAIN_GAP:
            continue
        if abs(score[p]) >= CONFIDENT_SCORE and abs(score[q]) >= CONFIDENT_SCORE and np.sign(score[p]) == np.sign(score[q]):
            chain[k] = W_PASS_CHAIN * np.sign(score[p])
    score = score + chain

    nearest = np.argmin(near[frames], axis=1).astype(np.int8)
    side = np.where(score > 0, C.HOME, np.where(score < 0, C.AWAY, nearest)).astype(np.int8)
    both = reach[frames].all(axis=1)
    raw = e.near_dist[frames].astype(np.float64)
    close_tie = (np.nan_to_num(raw[:, 0], nan=99) <= CONTESTED_BOTH_M) & (np.nan_to_num(raw[:, 1], nan=99) <= CONTESTED_BOTH_M) & \
        (np.abs(np.nan_to_num(raw[:, 0], nan=99) - np.nan_to_num(raw[:, 1], nan=99)) <= CONTESTED_DIFF_M)
    contested = (both & (np.abs(score) < CONTESTED_MAX_SCORE) & (hold == 0)) | (close_tie & (hold == 0))
    slot = e.near_slot[frames, side]
    table = pd.DataFrame({
        "frame": frames,
        "half": half,
        "side": SIDE_NAMES[side],
        "side_score": np.round(score, 3),
        "side_nearest": SIDE_NAMES[nearest],
        "score_proximity": np.round(prox, 3),
        "score_direction": np.round(direction, 3),
        "score_continuity": np.round(cont, 3),
        "score_hold": np.round(hold, 3),
        "score_next": np.round(nxt, 3),
        "score_pass_chain": np.round(chain, 3),
        "slot": slot,
        "reach_m": np.round(radius[frames], 2),
        "dist_m": near[frames, side].round(3),
        "opp_dist_m": np.where(np.isfinite(near[frames, 1 - side]), near[frames, 1 - side], np.nan).round(3),
        "both_in_reach": both,
        "contested": contested,
        "dv": dv[frames].round(2),
        "speed_before": sb[frames].round(2),
        "speed_after": sa[frames].round(2),
        "turn_deg": e.ball_turn_deg[frames].round(1),
        "ball_x_m": bx.round(2),
        "ball_y_m": by.round(2),
        "ball_adv_home": e.ball_adv_home[frames].round(2),
        "kind": kind,
    })

    # Pass-like: same side as the previous touch in the half, both confidently attributed,
    # by someone else (touch points more than 5 m apart), within 6 s.
    prev_same = np.zeros(n_t, dtype=bool)
    for k in range(1, n_t):
        if half[k] != half[k - 1]:
            continue
        gap = frames[k] - frames[k - 1]
        moved = np.hypot(bx[k] - bx[k - 1], by[k] - by[k - 1])
        confident = abs(score[k]) >= CONFIDENT_SCORE and abs(score[k - 1]) >= CONFIDENT_SCORE
        prev_same[k] = (side[k] == side[k - 1]) and confident and gap <= 150 and moved > 5.0
    table["completes_pass"] = prev_same
    return table


# --------------------------------------------------------------------------------------------
# control spells

def detect_control(e):
    n = e.n_frames
    side = e.nearest_side
    cond = (e.graded & e.ball_ok & (side >= 0) & (e.nearest_dist <= CONTROL_RADIUS_M) &
            (e.nearest_rel_speed <= CONTROL_MAX_REL_SPEED) &
            (np.nan_to_num(e.nearest_margin, nan=np.inf) >= CONTROL_MIN_MARGIN_M))
    control_side = np.full(n, -1, dtype=np.int8)
    rows = []
    for s in (C.HOME, C.AWAY):
        mask = cond & (side == s)
        # close short holes
        for start, stop in runs(~mask):
            if 0 < start and stop < n and stop - start <= CONTROL_MAX_HOLE and mask[start - 1] and mask[stop] \
                    and e.half[start - 1] == e.half[stop]:
                mask[start:stop] = True
        for start, stop in runs(mask):
            if stop - start < CONTROL_MIN_FRAMES:
                continue
            control_side[start:stop] = s
            travel = np.nansum(np.linalg.norm(np.diff(e.ball_m[start:stop], axis=0), axis=1))
            rows.append({"start": start, "end": stop - 1, "frames": stop - start,
                         "half": int(e.half[start]), "side": SIDE_NAMES[s],
                         "mean_dist_m": round(float(np.nanmean(e.nearest_dist[start:stop])), 3),
                         "ball_travel_m": round(float(travel), 2),
                         "ball_x_m": round(float(e.ball_m[start, 0]), 2),
                         "ball_y_m": round(float(e.ball_m[start, 1]), 2)})
    table = pd.DataFrame(rows, columns=["start", "end", "frames", "half", "side", "mean_dist_m",
                                        "ball_travel_m", "ball_x_m", "ball_y_m"])
    return control_side, table.sort_values("start").reset_index(drop=True)


def goalkeeper_slots(e, home_attacks_positive_x):
    """Goalkeeper slot of each side in each half: {(half, side): slot}.

    The goalkeeper is the slot whose tracked positions sit closest to its own goal on median over
    the half. This only needs the tracking and team labels, no tags and no labels.
    """
    out = {}
    for h in (1, 2):
        home_pos = bool(home_attacks_positive_x[str(h)])
        d0 = C.HOME if home_pos else C.AWAY       # side defending the goal at x = 0
        sel = (e.half == h) & e.graded
        for side in (C.HOME, C.AWAY):
            own_goal_x = 0.0 if side == d0 else C.PITCH_LENGTH
            best, best_d = -1, np.inf
            for slot in np.where(e.slot_team == side)[0]:
                ok = sel & e.players_present[:, slot]
                if ok.sum() < 25 * 60:
                    continue
                d = float(np.median(np.abs(e.players_m[ok, slot, 0] - own_goal_x)))
                if d < best_d:
                    best, best_d = int(slot), d
            out[(h, side)] = best
    return out


# A keeper who catches a fast ball often hides it from the tracker. If the ball was seen
# arriving at CATCH_MIN_SPEED or more well inside the pitch (no ball-out in the 10 s before) and is
# next seen held still in the same penalty area, the hold began when it vanished, not when it
# reappeared.
CATCH_MIN_SPEED, CATCH_MAX_GAP, CATCH_MIN_INSIDE_M, CATCH_OUT_WINDOW = 5.0, 20 * C.FPS, 2.0, 10 * C.FPS


def detect_carry(e, control_side):
    """Looser slow-carry spells (side per frame, -1 none), excluding frames already in control."""
    n = e.n_frames
    side = e.nearest_side
    cond = (e.graded & e.ball_ok & (side >= 0) & (e.nearest_dist <= CARRY_RADIUS_M) &
            (e.nearest_rel_speed <= CARRY_MAX_REL_SPEED) &
            (np.nan_to_num(e.nearest_margin, nan=np.inf) >= CARRY_MIN_MARGIN_M) &
            (np.nan_to_num(e.ball_speed_1s, nan=0.0) >= CARRY_MIN_BALL_SPEED))
    carry = np.full(n, -1, dtype=np.int8)
    for sd in (C.HOME, C.AWAY):
        mask = cond & (side == sd)
        for start, stop in runs(~mask):
            if 0 < start and stop < n and stop - start <= CARRY_MAX_HOLE and mask[start - 1] and mask[stop] \
                    and e.half[start - 1] == e.half[stop]:
                mask[start:stop] = True
        for start, stop in runs(mask):
            if stop - start >= CARRY_MIN_FRAMES:
                carry[start:stop] = sd
    carry[control_side >= 0] = -1
    return carry


def caught_start(e, start):
    lo = max(0, start - CATCH_MAX_GAP)
    seen = np.flatnonzero(e.ball_ok[lo:start]) + lo
    if not len(seen) or seen[-1] >= start - 1:
        return start
    last = int(seen[-1])
    if e.half[last] != e.half[start]:
        return start
    recent = slice(max(0, last - 3 * C.FPS), last + 1)
    approach = np.nanmax(np.where(e.ball_ok[recent], e.ball_speed[recent], np.nan))
    inside = float(e.ball_boundary_dist[last]) >= CATCH_MIN_INSIDE_M
    out = bool(e.ball_out[max(0, last - CATCH_OUT_WINDOW):start].any())
    if np.isfinite(approach) and approach >= CATCH_MIN_SPEED and inside and not out:
        return last + 1
    return start


# Confirmed hidden hold. A keeper who gathers the ball usually hides it from the tracker, and with
# everyone walking, stage 03 sees a missing ball and calls it dead. The hidden-hold rule above
# gives such a gap to the defending side, but it can't tell a hold from a goal, a ball gone out or
# a foul, so stage 03 ignores it. How the ball comes back decides. The gap is a live hold when:
#   - the ball vanished inside the penalty area (at least CHOLD_MIN_INSIDE_M inside the pitch) with
#     a defender within VHOLD_DEFENDER_M and no ball-out in the CATCH_OUT_WINDOW before, either
#     after a fast approach or straight out of a visible hold (within CHOLD_CHAIN frames);
#   - it reappears within CHOLD_MAX_GAP, on the pitch, within CHOLD_RETURN_MAX_M of that goal line
#     and not in the goal area (+ CHOLD_GOAL_AREA_MARGIN_M) -- a ball there is a goal kick;
#   - no set-piece tag starts and no GOALS tag is active during the gap;
#   - it comes back with the defending side: launched away from goal at RELEASE_MIN_SPEED or more,
#     or with a defender within CHOLD_NEAR_M and clearly closer than any attacker.
# Stage 03 then treats the hold as live play and stage 05 as the defending side's possession.
# MTAH_CONFIRMED_HOLD=0 turns this off.
CHOLD_ON = os.environ.get("MTAH_CONFIRMED_HOLD", "1") != "0"
CHOLD_MAX_GAP = 30 * C.FPS
CHOLD_MIN_INSIDE_M = 0.5
CHOLD_CHAIN = C.FPS
CHOLD_RETURN_MAX_M = 40.0
CHOLD_GOAL_AREA_MARGIN_M = 1.0
CHOLD_GOAL_AREA_DEPTH_M, CHOLD_GOAL_AREA_HALF_WIDTH_M = 5.5, 9.16
CHOLD_NEAR_M, CHOLD_NEAR_MARGIN_M = 2.5, 1.0
CHOLD_TAG_MARGIN = C.FPS


def hold_confirmed_by_return(e, start, stop, side, area_id):
    """The ball unseen over [start, stop) comes back as a goalkeeper's release (see the comment above)."""
    n = e.n_frames
    if not e.ball_ok[stop] or float(e.ball_boundary_dist[stop]) < 0:
        return False
    goal_x = 0.0 if area_id == 0 else C.PITCH_LENGTH
    x, y = float(e.ball_m[stop, 0]), float(e.ball_m[stop, 1])
    depth = abs(x - goal_x)
    if depth > CHOLD_RETURN_MAX_M:
        return False
    if (depth <= CHOLD_GOAL_AREA_DEPTH_M + CHOLD_GOAL_AREA_MARGIN_M and
            abs(y - C.PITCH_WIDTH / 2.0) <= CHOLD_GOAL_AREA_HALF_WIDTH_M + CHOLD_GOAL_AREA_MARGIN_M):
        return False
    f = np.arange(max(0, start - CHOLD_TAG_MARGIN), min(n, stop + CHOLD_TAG_MARGIN + 1))
    tagged = e.sp_active[f] >= 0
    tag_start = f - e.sp_since_start[f]
    if (tagged & (tag_start >= start - CHOLD_TAG_MARGIN)).any() or (e.goal_active[start:stop] >= 0).any():
        return False
    away = 1.0 if area_id == 0 else -1.0
    vx = e.ball_v[stop:min(n, stop + RELEASE_GRACE), 0].astype(np.float64) * away
    vx = vx[np.isfinite(vx)]
    launched = bool(len(vx) and vx.max() >= RELEASE_MIN_SPEED)
    own, opp = float(e.near_dist[stop, side]), float(e.near_dist[stop, 1 - side])
    own = own if np.isfinite(own) else 99.0
    opp = opp if np.isfinite(opp) else 99.0
    in_hand = own <= CHOLD_NEAR_M and opp - own >= CHOLD_NEAR_MARGIN_M
    return launched or in_hand


def detect_holds(e, home_attacks_positive_x):
    """Goalkeeper holds: (hold_side per frame, table of holds).

    Identity swaps: the tracker sometimes hands a goalkeeper's track to the opposing team.
    A goalkeeper cannot be in the *opposition's* penalty area holding a still ball, so when the
    only player within reach of a still ball in a penalty area is the opposing side's goalkeeper
    slot, that player is the defending goalkeeper with a swapped team label and the hold goes to
    the defending side. Nothing else about the track is changed.
    """
    n = e.n_frames
    area = e.ball_in_penalty_area
    defender = np.full(n, -1, dtype=np.int8)
    for h in (1, 2):
        home_pos = bool(home_attacks_positive_x[str(h)])
        in_half = e.half == h
        d0 = C.HOME if home_pos else C.AWAY      # side defending the goal at x = 0
        defender[in_half & (area == 0)] = d0
        defender[in_half & (area == 1)] = 1 - d0
    slow = np.nan_to_num(e.ball_speed_1s, nan=99.0) <= HOLD_MAX_SPEED
    dist_def = np.where(defender == C.HOME, e.near_dist[:, 0],
                        np.where(defender == C.AWAY, e.near_dist[:, 1], np.nan))
    cond_own = e.graded & e.ball_ok & (defender >= 0) & slow & (np.nan_to_num(dist_def, nan=99.0) <= HOLD_RADIUS_M)

    # swapped goalkeeper: the opposing side's goalkeeper slot is the one at the ball
    gk = goalkeeper_slots(e, home_attacks_positive_x)
    swapped = np.zeros(n, dtype=bool)
    cand = e.graded & e.ball_ok & (defender >= 0) & slow & ~cond_own
    for h in (1, 2):
        for side in (C.HOME, C.AWAY):
            slot = gk[(h, side)]
            if slot < 0:
                continue
            # frames where `side` is the attacker, i.e. the defender is the other side
            m = cand & (e.half == h) & (defender == 1 - side)
            idx = np.where(m)[0]
            if not len(idx):
                continue
            ok = e.players_eligible[idx, slot]
            d = np.linalg.norm(e.players_m[idx, slot] - e.ball_m[idx], axis=1)
            swapped[idx[ok & (d <= HOLD_RADIUS_M)]] = True
    cond = cond_own | swapped

    hold_side = np.full(n, -1, dtype=np.int8)
    rows = []
    for s in (C.HOME, C.AWAY):
        mask = cond & (defender == s)
        for start, stop in runs(~mask):
            if 0 < start and stop < n and stop - start <= HOLD_MAX_HOLE and mask[start - 1] and mask[stop] \
                    and e.half[start - 1] == e.half[stop]:
                mask[start:stop] = True
        for start, stop in runs(mask):
            if stop - start < HOLD_MIN_FRAMES:
                continue
            f = np.arange(stop, min(stop + RELEASE_SEARCH, n))
            fast = f[(np.nan_to_num(e.ball_speed[f], nan=0.0) >= RELEASE_MIN_SPEED) & (e.half[f] == e.half[start])]
            release = int(fast[0]) if len(fast) else int(stop)
            seen_at = int(start)
            start = caught_start(e, int(start))
            hold_side[start:release] = s
            rows.append({"start": int(start), "end": int(stop) - 1, "release": release,
                         "frames": release - int(start), "half": int(e.half[start]), "side": SIDE_NAMES[s],
                         "area": "x0" if area[seen_at] == 0 else "xL",
                         "ball_x_m": round(float(e.ball_m[seen_at, 0]), 2),
                         "ball_y_m": round(float(e.ball_m[seen_at, 1]), 2),
                         "swapped_gk": bool(swapped[seen_at:stop].any())})
    for r in rows:
        r["virtual"] = False
    # Hidden hold: the ball is lost inside a penalty area right after a fast approach (a shot or
    # cross collected by the goalkeeper) and a defending player stood close by. The visible-hold rule
    # cannot see it; credit the defending side from the vanish to the reappearance.
    taken = hold_side >= 0
    visible = hold_side.copy()
    for start, stop in runs(~e.ball_ok & e.graded):
        gap = stop - start
        if start == 0 or stop >= n or gap < VHOLD_MIN_GAP or taken[start:stop].any():
            continue
        last = start - 1
        if not e.ball_ok[last] or e.half[last] != e.half[min(stop, n - 1)] or defender[last] < 0 or not e.graded[stop]:
            continue
        side = int(defender[last])
        dd = float(e.near_dist[last, side]) if np.isfinite(e.near_dist[last, side]) else 99.0
        if dd > VHOLD_DEFENDER_M or e.ball_out[max(0, last - CATCH_OUT_WINDOW):stop].any():
            continue
        recent = slice(max(0, last - 3 * C.FPS), last + 1)
        approach = np.nanmax(np.where(e.ball_ok[recent], e.ball_speed[recent], np.nan)) if e.ball_ok[recent].any() else np.nan
        fast_in = bool(np.isfinite(approach) and approach >= VHOLD_MIN_APPROACH)
        inside = float(e.ball_boundary_dist[last])
        plain = gap <= VHOLD_MAX_GAP and fast_in and inside >= CATCH_MIN_INSIDE_M
        # Confirmed hidden hold: the way the ball comes back shows the keeper had it all along.
        was_held = bool((visible[max(0, last - CHOLD_CHAIN):last + 1] == side).any())
        confirmed = (CHOLD_ON and gap <= CHOLD_MAX_GAP and inside >= CHOLD_MIN_INSIDE_M and
                     (fast_in or was_held) and hold_confirmed_by_return(e, start, stop, side, int(area[last])))
        if not (plain or confirmed):
            continue
        hold_side[start:stop] = side
        rows.append({"start": int(start), "end": int(stop) - 1, "release": int(stop), "frames": int(gap),
                     "half": int(e.half[last]), "side": SIDE_NAMES[side], "area": "x0" if area[last] == 0 else "xL",
                     "ball_x_m": round(float(e.ball_m[last, 0]), 2), "ball_y_m": round(float(e.ball_m[last, 1]), 2),
                     "swapped_gk": False, "virtual": True, "confirmed": bool(confirmed)})
    for r in rows:
        r.setdefault("confirmed", False)
    table = pd.DataFrame(rows, columns=["start", "end", "release", "frames", "half", "side", "area",
                                        "ball_x_m", "ball_y_m", "swapped_gk", "virtual", "confirmed"])
    return hold_side, table.sort_values("start").reset_index(drop=True)


def apply_holds(e, control_side, control, hold_side, holds):
    """Holds override control: the holder controls, opponents' spells inside a hold are dropped."""
    control_side = np.where(hold_side >= 0, hold_side, control_side).astype(np.int8)
    control = control.copy()
    control["kind"] = "close_control"
    keep = np.ones(len(control), dtype=bool)
    for row in holds.itertuples(index=False):
        overlap = (control["end"] >= row.start) & (control["start"] < row.release) & (control["side"] != row.side)
        keep &= ~overlap.to_numpy()
    control = control[keep]
    extra = pd.DataFrame([{"start": r.start, "end": r.release - 1, "frames": r.release - r.start,
                           "half": r.half, "side": r.side,
                           "mean_dist_m": round(float(np.nanmean(e.near_dist[r.start:r.release, 0 if r.side == "H" else 1])), 3),
                           "ball_travel_m": round(float(np.nansum(np.linalg.norm(np.diff(e.ball_m[r.start:r.release], axis=0), axis=1))), 2),
                           "ball_x_m": r.ball_x_m, "ball_y_m": r.ball_y_m, "kind": "hold"}
                          for r in holds.itertuples(index=False)], columns=list(control.columns))
    control = pd.concat([control, extra], ignore_index=True).sort_values("start").reset_index(drop=True)
    return control_side, control


# --------------------------------------------------------------------------------------------
# last contact and pass 1

def last_contact(e, touches, control_side):
    n = e.n_frames
    contact_side = control_side.copy()
    contested_at = np.zeros(n, dtype=bool)
    tf = touches["frame"].to_numpy()
    ts = np.where(touches["side"].to_numpy() == "H", C.HOME, C.AWAY).astype(np.int8)
    contact_side[tf] = ts                     # a touch overrides a control frame it falls on
    contested_at[tf] = touches["contested"].to_numpy()

    idx = np.arange(n)
    has = contact_side >= 0
    last = np.maximum.accumulate(np.where(has, idx, -1))
    same_half = (last >= 0) & (e.half[np.clip(last, 0, n - 1)] == e.half) & (e.half > 0)
    last_frame = np.where(same_half, last, -1).astype(np.int32)
    last_side = np.where(same_half, contact_side[np.clip(last, 0, n - 1)], -1).astype(np.int8)
    last_contested = same_half & contested_at[np.clip(last, 0, n - 1)]
    return contact_side, last_side, last_frame, last_contested


def pass_one(e, touches, control):
    """Rough possession: switch side only on confirmed contact. Returns (p1_side, p1_change)."""
    events = []  # (frame, side, confirms_alone)
    for row in touches.itertuples(index=False):
        events.append((int(row.frame), C.HOME if row.side == "H" else C.AWAY, False))
    for row in control.itertuples(index=False):
        events.append((int(row.start), C.HOME if row.side == "H" else C.AWAY,
                       row.frames >= CONFIRM_CONTROL_FRAMES))
    events.sort()

    n = e.n_frames
    p1 = np.full(n, -1, dtype=np.int8)
    change = np.zeros(n, dtype=bool)
    blocks = [(h, a, b) for h, a, b in C.games.half_video_blocks(C.GAME)]
    for h, first, last in blocks:
        ev = [x for x in events if first <= x[0] <= last]
        state, since = -1, first
        switches = []
        for k, (frame, side, alone) in enumerate(ev):
            if side == state:
                continue
            confirmed = alone
            if not confirmed and k + 1 < len(ev):
                nf, ns, _ = ev[k + 1]
                confirmed = ns == side and nf - frame <= FOLLOW_WINDOW
            if state == -1:
                confirmed = True  # first evidence of the half sets the initial side
            if confirmed:
                switches.append((frame, side))
                state = side
        for k, (frame, side) in enumerate(switches):
            stop = switches[k + 1][0] if k + 1 < len(switches) else last + 1
            p1[frame:stop] = side
            if k > 0:
                change[frame] = True
        del since
    return p1, change


# --------------------------------------------------------------------------------------------
# main

def build(work_dir=None):
    t0 = time.time()
    checks = Checks()
    e = load_evidence(work_dir)
    n = e.n_frames
    g = e.graded
    print("loaded s01 evidence: {} frames".format(n))

    dv, sb, sa, vb, va = impulse(e.ball_m.astype(np.float64), e.ball_ok, e.ball_run_id, IMPULSE_SPAN)
    s00_meta = C.read_json(os.path.join(work_dir or C.WORK_DIR, "s00", "meta.json"))
    hp = s00_meta["home_attacks_positive_x"]
    control_side, control = detect_control(e)
    hold_side, holds = detect_holds(e, hp)
    control_side, control = apply_holds(e, control_side, control, hold_side, holds)
    touches = detect_touches(e, dv, sb, sa, vb, va, control, holds, hp)
    carry_side = detect_carry(e, control_side)
    contact_side, last_side, last_frame, last_contested = last_contact(e, touches, control_side)
    p1, p1_change = pass_one(e, touches, control)

    touch = np.zeros(n, dtype=bool)
    touch[touches["frame"].to_numpy()] = True
    touch_side = np.full(n, -1, dtype=np.int8)
    touch_side[touches["frame"].to_numpy()] = np.where(touches["side"] == "H", C.HOME, C.AWAY)

    # -- checks and statistics ------------------------------------------------------------------
    checks.add("touches only on graded frames with a usable ball",
               bool(np.all(touch <= (g & e.ball_ok))))
    checks.add("pass 1 is undecided outside the halves", bool(np.all(p1[~g] == -1)))
    checks.add("last contact never reaches into another half",
               bool(np.all((last_frame < 0) | (e.half[np.clip(last_frame, 0, n - 1)] == e.half))))
    graded_minutes = g.sum() / C.FPS / 60.0
    p1_changes = int(p1_change.sum())
    spells = [(a, b) for s in (0, 1) for a, b in runs(p1 == s)]
    durations = np.array([b - a for a, b in spells]) / float(C.FPS) if spells else np.array([0.0])
    stats = {
        "touches": int(len(touches)),
        "touches_per_minute": round(len(touches) / graded_minutes, 2),
        "touches_H": int((touches["side"] == "H").sum()),
        "touches_A": int((touches["side"] == "A").sum()),
        "touches_contested_pct": round(100 * float(touches["contested"].mean()), 1) if len(touches) else 0,
        "touch_kinds": touches["kind"].value_counts().to_dict(),
        "touches_side_changed_vs_nearest_pct": round(100 * float((touches["side"] != touches["side_nearest"]).mean()), 1) if len(touches) else 0,
        "touches_confident_pct": round(100 * float((touches["side_score"].abs() >= CONFIDENT_SCORE).mean()), 1) if len(touches) else 0,
        "pass_like_touches": int(touches["completes_pass"].sum()),
        "control_spells": int((control["kind"] == "close_control").sum()),
        "goalkeeper_holds": int(len(holds)),
        "touches_credited_by_hold": int((touches["score_hold"] != 0).sum()),
        "touches_with_next_contact_cue": int((touches["score_next"] != 0).sum()),
        "control_frames_pct_graded": round(100 * float((control_side[g] >= 0).mean()), 2),
        "p1_changes": p1_changes,
        "p1_spells": len(spells),
        "p1_spell_median_s": round(float(np.median(durations)), 2),
        "p1_coverage_pct_graded": round(100 * float((p1[g] >= 0).mean()), 2),
        "p1_home_share": round(float((p1[g] == 0).sum() / max(1, (p1[g] >= 0).sum())), 3),
    }
    checks.add("touch rate is plausible for a football match", 5 <= stats["touches_per_minute"] <= 30,
               "{} per graded minute ({} touches)".format(stats["touches_per_minute"], stats["touches"]),
               severity="warning")
    checks.add("pass-1 spells", True, "{} spells, median {} s, home share {}".format(
        stats["p1_spells"], stats["p1_spell_median_s"], stats["p1_home_share"]), severity="warning")

    print("\nchecks")
    checks.report()
    if checks.failed("error"):
        raise SystemExit("\ncritical check failed; nothing written.")

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    np.savez(os.path.join(out, "contacts.npz"),
             ball_dv=dv, ball_speed_before=sb, ball_speed_after=sa,
             touch=touch, touch_side=touch_side, control_side=control_side, hold_side=hold_side, carry_side=carry_side,
             contact_side=contact_side, last_contact_side=last_side,
             last_contact_frame=last_frame, last_contact_contested=last_contested,
             p1_side=p1, p1_change=p1_change)
    touches.to_csv(os.path.join(out, "touches.csv"), index=False)
    control.to_csv(os.path.join(out, "control_spells.csv"), index=False)
    holds.to_csv(os.path.join(out, "holds.csv"), index=False)
    labels = np.where(p1 == 0, "H", np.where(p1 == 1, "A", "X"))
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(
        os.path.join(out, "p1_labels.csv"), index=False)
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE,
        "parameters": {k: v for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, tuple, str))},
        "stats_graded": stats,
        "arrays": CONTACT_ARRAYS,
        "checks": checks.items,
        "note": "p1_labels.csv has no dead-ball state; it is a diagnostic, not a submission.",
    })
    print("\nstats")
    for k, v in stats.items():
        print("  {:<28} {}".format(k, v))
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


class Contacts(object):
    """Read-only view of work/s02: c.p1_side, c.last_contact_side, c.touches ..."""

    def __init__(self, work_dir=None):
        folder = os.path.join(work_dir or C.WORK_DIR, STAGE)
        path = os.path.join(folder, "contacts.npz")
        if not os.path.isfile(path):
            raise IOError("{} not found -- run: python -m pipeline4.s02_touches".format(path))
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.touches = pd.read_csv(os.path.join(folder, "touches.csv"))
        self.control = pd.read_csv(os.path.join(folder, "control_spells.csv"))
        self.holds = pd.read_csv(os.path.join(folder, "holds.csv"))
        self.meta = C.read_json(os.path.join(folder, "meta.json"))


def load_contacts(work_dir=None):
    return Contacts(work_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 02: touches and pass-1 possession.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    args = parser.parse_args(argv)
    build(args.work_dir)


if __name__ == "__main__":
    main()
