"""Stage 01 -- frame-level evidence (rule 3, plus the tracking clean-up it needs).

For every video frame, from the stage-00 timeline:

  ball      cleaned position (spikes removed, sub-0.2 s gaps bridged), velocity, speed,
            acceleration, turn angle, stillness, gap bookkeeping (since/until seen, gap length,
            last/next seen position), jumps
  pitch     signed distance to goal lines and touchlines, out-of-bounds flags, nearest boundary,
            distance to centre spot and nearest corner, penalty/goal-area membership, position
            along each team's direction of attack
  players   eligibility (on or near the pitch, tracker duplicates merged), velocities, tracklets
            (id, age, length, births/deaths, tracker block boundaries), per-team count, centroid,
            mean speed, share of slow players
  contact   nearest eligible player of each team to the ball, distances, margin, slot, speed,
            velocity relative to the ball, players within 2 m and 5 m
  tagging   Smart Tagging hints restricted to graded instances: possession-phase side, zone side,
            active set piece / goal instance and frames since it started
  camera    homography validity, error, reliability flag

Nothing here assigns a label. Thresholds are physical constants chosen from football and from
the sensor's behaviour (documented next to each one), never fitted to any ground truth.

    python -m pipeline4.s01_evidence

Outputs, in work/s01/:

    evidence.npz          every array (see EVIDENCE_ARRAYS)
    evidence_frames.csv   the per-frame scalars, for a spreadsheet
    tracklets.csv         one row per tracklet: slot, side, first/last frame, length
    meta.json             parameters, statistics and checks
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

STAGE = "s01"

# --------------------------------------------------------------------------------------------
# parameters -- physical, not fitted

# Ball. The hardest shots in football are ~35 m/s; 40 m/s (1.6 m per frame at 25 fps) is beyond
# anything real, so a single-frame excursion further than that from both neighbours is a
# detector error, and a step that large between two genuine observations is a track jump.
BALL_MAX_STEP_M = 1.6
BALL_SPIKE_PASSES = 2
# Gaps of at most this many frames (0.2 s) are bridged linearly. The tracker almost never leaves
# gaps this short, so this only repairs the few single-frame dropouts and spikes removed above.
BALL_MAX_INTERP_FRAMES = 5
# Velocity: centred difference over +/- 2 frames (80 ms span), falling back to +/- 1 and then
# one-sided at the end of a run.
VEL_HALF_SPAN = 2
# Turn angle: compare the ball's displacement over the 3 frames before and the 3 after. Below
# 0.15 m of movement in either (~1.25 m/s) the direction is noise and the angle is undefined.
TURN_SPAN = 3
TURN_MIN_DISP_M = 0.15
# A ball slower than this is "still" (placed for a restart, or at a player's feet).
BALL_STILL_SPEED = 1.0
STILL_HALF_WINDOW = 12  # rolling share of still frames over +/- 0.5 s
# Net displacement over 1 s (+/- 12 frames). Robust to frame-level jitter, which at 25 fps turns
# ~0.1 m of noise into ~1-2 m/s of apparent speed. During stoppages the ball is often carried or
# rolled back, so it is rarely truly still; low net displacement is the steadier cue.
DISP_HALF_WINDOW = 12

# Pitch geometry (Laws of the Game), metres.
PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH = 16.5, 40.32
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.32
# A ball has to be wholly over the line; tracking is accurate to a few tens of cm. Two flags:
# "out" beyond the line at all, "clearly out" beyond 1 m.
OUT_CLEAR_MARGIN_M = 1.0

# Players. A track more than 3 m outside the pitch is a substitute, coach or steward, not a
# player on the field (a throw-in taker stands within a metre or so of the line).
PLAYER_FIELD_MARGIN_M = 3.0
# Two same-team tracks closer than this are one person tracked twice. Same-team pair distances
# show no excess near zero, so this is deliberately tight: it removes clear duplicates only.
DUPLICATE_DIST_M = 0.3
# Walking pace. The share of players below it is a stoppage cue for later stages.
SLOW_SPEED = 1.5
CONTACT_RADII_M = (2.0, 5.0)
# The tracker works in blocks of this many frames; tracklet starts cluster on its multiples.
TRACKER_BLOCK = 125

# Homography error above this is treated as unreliable (median during play is ~0.24).
HOM_RELIABLE_MAX_ERROR = 2.0

# Smart Tagging families that say who has the ball. Zone tags are kept apart: they say where the
# ball is, and only weakly who has it.

# Codes for side-valued int8 arrays.
NONE, SIDE_H, SIDE_A, SIDE_BOTH = -1, 0, 1, 2
# Nearest boundary codes.
BOUNDARY_NAMES = ("goal_line_x0", "goal_line_xL", "touchline_y0", "touchline_yW")

# Ball trust. A slow ball can't speed up by itself. If the tracked ball gains ORPHAN_MIN_GAIN m/s
# (mean speed over ORPHAN_SPAN frames after vs before) from ORPHAN_MAX_SPEED_BEFORE m/s or less,
# with nobody within ORPHAN_CLEAR_M (the widest reach stage 02 allows for a touch), the track is
# not following the real ball -- usually a stale or interpolated position. The stretch around that
# impulse is flagged: back to the last frame a player was in reach (plus ORPHAN_TOUCH_GUARD frames,
# so that touch keeps its velocity window) and on to the next one, at most ORPHAN_MAX_EXTENT each
# way. Impulses off the pitch are ignored, since ball boys and throw-in takers aren't tracked.
# MTAH_BALL_TRUST=0 turns this off.
BALL_TRUST_ON = os.environ.get("MTAH_BALL_TRUST", "1") != "0"
# The stretch is only flagged; stage 05 uses the impulse to time a change of possession. Dropping
# the positions as well (MTAH_BALL_TRUST_DROP=1) also hits real play where a player is simply not
# tracked, e.g. a keeper collecting a through ball, and stage 03 then calls it a dead ball.
BALL_TRUST_DROP = os.environ.get("MTAH_BALL_TRUST_DROP", "0") != "0"
ORPHAN_SPAN = 8
ORPHAN_MIN_GAIN = 3.0
ORPHAN_MAX_SPEED_BEFORE = 5.0
ORPHAN_CLEAR_M = 4.0
ORPHAN_REACH_BASE_M, ORPHAN_REACH_PER_MS, ORPHAN_REACH_CAP_M = 2.0, 0.15, 4.0   # stage 02 touch reach
ORPHAN_TOUCH_GUARD = 8
ORPHAN_MAX_EXTENT = 5 * C.FPS


EVIDENCE_ARRAYS = {
    # copied from s00
    "period": "(N,) int8", "half": "(N,) int8", "graded": "(N,) bool",
    # ball
    "ball_observed": "(N,) bool    tracked and not a spike",
    "ball_spike": "(N,) bool       tracked but removed as a spike",
    "ball_interp": "(N,) bool      bridged by interpolation",
    "ball_ok": "(N,) bool          observed or bridged: a usable ball position",
    "ball_m": "(N,2) float32       cleaned position in metres, NaN when not ok",
    "ball_v": "(N,2) float32       velocity m/s",
    "ball_speed": "(N,) float32    m/s",
    "ball_accel": "(N,) float32    m/s^2, magnitude of the change in velocity",
    "ball_turn_deg": "(N,) float32 angle between movement before and after, degrees",
    "ball_jump": "(N,) bool        step from the previous frame larger than BALL_MAX_STEP_M",
    "ball_still": "(N,) bool       speed < BALL_STILL_SPEED",
    "ball_still_share": "(N,) float32 share of still frames over +/- 0.5 s",
    "ball_disp_1s": "(N,) float32  net displacement between -0.5 s and +0.5 s, m",
    "ball_speed_1s": "(N,) float32 rolling mean speed over +/- 0.5 s",
    "ball_run_id": "(N,) int32     id of the continuous ok run, -1 in a gap",
    "ball_since_seen": "(N,) int32 frames since the ball was last ok, 0 when ok, -1 unknown",
    "ball_until_seen": "(N,) int32 frames until the ball is next ok, 0 when ok, -1 unknown",
    "ball_gap_len": "(N,) int32    length of the gap a missing frame belongs to, 0 when ok, -1 open",
    "ball_last_seen_m": "(N,2) float32 last ok position in the same period",
    "ball_next_seen_m": "(N,2) float32 next ok position in the same period",
    "ball_untrusted": "(N,) bool    position not explained by any player (flag; discarded only with MTAH_BALL_TRUST_DROP=1)",
    "ball_orphan_impulse": "(N,) bool frame of the unexplained acceleration inside an untrusted stretch",
    # pitch
    "ball_goalline_dist": "(N,) float32  signed distance to the nearer goal line, < 0 outside",
    "ball_touchline_dist": "(N,) float32 signed distance to the nearer touchline, < 0 outside",
    "ball_boundary_dist": "(N,) float32  min of the two",
    "ball_nearest_boundary": "(N,) int8  index into BOUNDARY_NAMES",
    "ball_out": "(N,) bool               beyond any line",
    "ball_out_clear": "(N,) bool         beyond any line by > OUT_CLEAR_MARGIN_M",
    "ball_centre_dist": "(N,) float32",
    "ball_corner_dist": "(N,) float32",
    "ball_in_penalty_area": "(N,) int8   -1 none, 0 area at x=0, 1 area at x=L",
    "ball_in_goal_area": "(N,) int8      -1 none, 0 at x=0, 1 at x=L",
    "ball_adv_home": "(N,) float32  ball metres from home's own goal line towards the goal home attacks",
    # players
    "players_m": "(N,S,2) float32", "players_v": "(N,S,2) float32 m/s from positions",
    "players_speed": "(N,S) float32 delivered speed (m/s)",
    "players_present": "(N,S) bool", "players_eligible": "(N,S) bool",
    "players_off_pitch": "(N,S) bool", "players_duplicate": "(N,S) bool",
    "slot_team": "(S,) int8",
    "players_team": "(N,S) int8   team per frame: slot_team except swapped goalkeepers",
    "tracklet_id": "(N,S) int32 -1 absent", "tracklet_age": "(N,S) int32 frames since start",
    "tracklet_len": "(N,S) int32 total length of the tracklet",
    "track_births": "(N,) int16", "track_deaths": "(N,) int16",
    "block_boundary": "(N,) bool frame % TRACKER_BLOCK == 0",
    "n_eligible": "(N,2) int8 per side", "team_overcount": "(N,2) bool more than 11 eligible",
    "team_centroid_m": "(N,2,2) float32 [side, xy]",
    "team_mean_speed": "(N,2) float32", "mean_speed_all": "(N,) float32",
    "slow_share": "(N,) float32 share of eligible players slower than SLOW_SPEED",
    "mean_speed_all_2s": "(N,) float32 rolling +/- 1 s",
    # contact
    "near_dist": "(N,2) float32 nearest eligible player of each side to the ball, m",
    "near_slot": "(N,2) int16 slot of that player, -1 none",
    "nearest_side": "(N,) int8 side of the nearest player overall, -1 none",
    "nearest_dist": "(N,) float32", "nearest_margin": "(N,) float32 other side's distance minus nearest's",
    "nearest_speed": "(N,) float32 delivered speed of the nearest player",
    "nearest_rel_speed": "(N,) float32 |v_ball - v_player| for the nearest player",
    "n_within": "(N,2,R) int8 players of each side within CONTACT_RADII_M",
    # tagging
    "tag_poss_hint": "(N,) int8 side from possession-phase tags: -1 none, 0 H, 1 A, 2 both",
    "tag_zone_side": "(N,) int8 side from zone tags, same coding",
    "sp_active": "(N,) int32 row in smart_tagging_events.csv of the latest-started active set piece, -1",
    "sp_since_start": "(N,) int32 frames since that set piece started, -1",
    "goal_active": "(N,) int32 row of the latest-started active GOALS instance, -1",
    # camera
    "hom_valid": "(N,) bool", "hom_error": "(N,) float32", "hom_reliable": "(N,) bool",
}


# --------------------------------------------------------------------------------------------
# small helpers

def run_ids(mask):
    """Id of each True run (0, 1, ...), -1 on False frames."""
    mask = np.asarray(mask, dtype=bool)
    starts = mask & ~np.concatenate(([False], mask[:-1]))
    ids = np.cumsum(starts) - 1
    return np.where(mask, ids, -1).astype(np.int32)


def runs(mask):
    """[(start, stop)] half-open spans where mask is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def rolling_mean(values, half_window):
    """NaN-aware centred mean over +/- half_window frames; NaN where the window has no data."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    n = len(values)
    idx = np.arange(n)
    lo = np.clip(idx - half_window, 0, n)
    hi = np.clip(idx + half_window + 1, 0, n)
    count = c[hi] - c[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (s[hi] - s[lo]) / count
    out[count == 0] = np.nan
    return out.astype(np.float32)


def centred_velocity(pos, ok, seg, half_span):
    """Velocity (m/s) of a (N, ..., 2) track by centred differences inside a segment.

    Tries +/- half_span, then +/- 1, then one-sided forward/backward. A difference is only taken
    between two frames that are both ok and belong to the same segment (ball run or tracklet), so
    a velocity never spans a gap or an identity switch.
    """
    n = pos.shape[0]
    vel = np.full(pos.shape, np.nan, dtype=np.float32)
    have = np.zeros(ok.shape, dtype=bool)

    def attempt(back, fwd):
        i = np.arange(back, n - fwd)
        a, b = i - back, i + fwd
        good = ok[a] & ok[b] & (seg[a] == seg[b]) & ~have[i]
        if not good.any():
            return
        dt = (back + fwd) / float(C.FPS)
        diff = (pos[b] - pos[a]) / dt
        if good.ndim == 1:          # ball: (n',)
            vel[i[good]] = diff[good]
            have[i[good]] = True
        else:                       # players: (n', S)
            fi, si = np.nonzero(good)
            vel[i[fi], si] = diff[fi, si]
            have[i[fi], si] = True

    for back, fwd in ((half_span, half_span), (1, 1), (0, 1), (1, 0)):
        attempt(back, fwd)
    return vel


# --------------------------------------------------------------------------------------------
# ball

def clean_ball(ball_xy_norm, period, drop=None):
    """Spike removal, short-gap bridging, then kinematics and gap bookkeeping.

    drop: optional (N,) bool, tracked positions to discard before cleaning (ball-trust fix).
    """
    n = len(ball_xy_norm)
    pos = C.to_metres(ball_xy_norm).astype(np.float64)
    if drop is not None:
        pos[drop] = np.nan
    tracked = np.isfinite(pos[:, 0])
    observed = tracked.copy()

    # 1. spikes: a lone observation far from both neighbours that are close to each other.
    spike = np.zeros(n, dtype=bool)
    for _ in range(BALL_SPIKE_PASSES):
        idx = np.flatnonzero(observed)
        if len(idx) < 3:
            break
        prev_i, cur_i, next_i = idx[:-2], idx[1:-1], idx[2:]
        d_prev = np.linalg.norm(pos[cur_i] - pos[prev_i], axis=1) / (cur_i - prev_i)
        d_next = np.linalg.norm(pos[next_i] - pos[cur_i], axis=1) / (next_i - cur_i)
        d_skip = np.linalg.norm(pos[next_i] - pos[prev_i], axis=1) / (next_i - prev_i)
        is_spike = (d_prev > BALL_MAX_STEP_M) & (d_next > BALL_MAX_STEP_M) & (d_skip <= BALL_MAX_STEP_M)
        if not is_spike.any():
            break
        spike[cur_i[is_spike]] = True
        observed[cur_i[is_spike]] = False

    # 2. bridge short gaps inside a period, between two observations.
    clean = np.where(observed[:, None], pos, np.nan)
    interp = np.zeros(n, dtype=bool)
    for start, stop in runs(~observed):
        length = stop - start
        if length > BALL_MAX_INTERP_FRAMES or start == 0 or stop >= n:
            continue
        a, b = start - 1, stop
        if period[a] != period[b] or not (observed[a] and observed[b]):
            continue
        if np.linalg.norm(pos[b] - pos[a]) > BALL_MAX_STEP_M * (length + 1):
            continue  # the two ends are not the same ball trajectory
        w = (np.arange(start, stop) - a) / float(b - a)
        clean[start:stop] = pos[a] + w[:, None] * (pos[b] - pos[a])
        interp[start:stop] = True
    ok = observed | interp

    # 3. kinematics, never across a gap or a period change
    seg = run_ids(ok)
    seg = np.where(ok, seg * 8 + period, -1)  # period folded in so a run cannot span one
    vel = centred_velocity(clean, ok, seg, VEL_HALF_SPAN)
    speed = np.linalg.norm(vel, axis=1)
    accel_vec = centred_velocity(vel, ok & np.isfinite(speed), seg, VEL_HALF_SPAN)
    accel = np.linalg.norm(accel_vec, axis=1)

    turn = np.full(n, np.nan)
    i = np.arange(TURN_SPAN, n - TURN_SPAN)
    a, b = i - TURN_SPAN, i + TURN_SPAN
    good = ok[a] & ok[i] & ok[b] & (seg[a] == seg[i]) & (seg[i] == seg[b])
    back = clean[i] - clean[a]
    fwd = clean[b] - clean[i]
    nb, nf = np.linalg.norm(back, axis=1), np.linalg.norm(fwd, axis=1)
    good &= (nb > TURN_MIN_DISP_M) & (nf > TURN_MIN_DISP_M)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip((back * fwd).sum(axis=1) / (nb * nf), -1, 1)
    turn[i[good]] = np.degrees(np.arccos(cos[good]))

    step = np.full(n, np.nan)
    step[1:] = np.linalg.norm(clean[1:] - clean[:-1], axis=1)
    jump = observed & np.concatenate(([False], observed[:-1])) & (step > BALL_MAX_STEP_M)

    still = ok & (speed < BALL_STILL_SPEED)
    disp = np.full(n, np.nan)
    k = DISP_HALF_WINDOW
    i = np.arange(k, n - k)
    good = ok[i - k] & ok[i + k] & (seg[i - k] == seg[i + k])
    disp[i[good]] = np.linalg.norm(clean[i[good] + k] - clean[i[good] - k], axis=1)
    still_share = rolling_mean(np.where(ok, still.astype(float), np.nan), STILL_HALF_WINDOW)

    # 4. gap bookkeeping within a period
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(ok, idx, -1))
    nxt = np.minimum.accumulate(np.where(ok, idx, n)[::-1])[::-1]
    last_ok = (last >= 0) & (period[np.clip(last, 0, n - 1)] == period)
    next_ok = (nxt < n) & (period[np.clip(nxt, 0, n - 1)] == period)
    since = np.where(ok, 0, np.where(last_ok, idx - last, -1)).astype(np.int32)
    until = np.where(ok, 0, np.where(next_ok, nxt - idx, -1)).astype(np.int32)
    gap_len = np.where(ok, 0, np.where(last_ok & next_ok, nxt - last - 1, -1)).astype(np.int32)
    last_seen = np.where(last_ok[:, None], clean[np.clip(last, 0, n - 1)], np.nan)
    next_seen = np.where(next_ok[:, None], clean[np.clip(nxt, 0, n - 1)], np.nan)

    return {
        "ball_observed": observed, "ball_spike": spike, "ball_interp": interp, "ball_ok": ok,
        "ball_m": clean.astype(np.float32), "ball_v": vel.astype(np.float32),
        "ball_speed": speed.astype(np.float32), "ball_accel": accel.astype(np.float32),
        "ball_turn_deg": turn.astype(np.float32), "ball_jump": jump,
        "ball_still": still, "ball_still_share": still_share,
        "ball_disp_1s": disp.astype(np.float32),
        "ball_speed_1s": rolling_mean(speed, DISP_HALF_WINDOW),
        "ball_run_id": run_ids(ok), "ball_since_seen": since, "ball_until_seen": until,
        "ball_gap_len": gap_len,
        "ball_last_seen_m": last_seen.astype(np.float32),
        "ball_next_seen_m": next_seen.astype(np.float32),
    }


def untrusted_ball(ball, players_m, eligible, period):
    """Stretches where the tracked ball moves in a way no player can have caused.

    Returns (untrusted, impulse): frames to discard, and the frame of each unexplained impulse.
    """
    n = len(period)
    ok = ball["ball_ok"]
    speed = ball["ball_speed"].astype(np.float64)
    run = ball["ball_run_id"]
    diff = players_m.astype(np.float64) - ball["ball_m"][:, None, :].astype(np.float64)
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist = np.where(eligible & np.isfinite(dist), dist, np.inf)
    nearest = dist.min(axis=1)
    del diff, dist
    reach = np.minimum(ORPHAN_REACH_BASE_M + ORPHAN_REACH_PER_MS * np.nan_to_num(speed), ORPHAN_REACH_CAP_M)
    in_reach = ok & (nearest <= reach)

    k = ORPHAN_SPAN
    csum = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(speed), speed, 0.0))))
    i = np.arange(k, n - k)
    whole = ok[i - k] & ok[i + k] & (run[i - k] == run[i + k]) & np.isfinite(speed[i - k]) & np.isfinite(speed[i + k])
    before = (csum[i] - csum[i - k]) / k
    after = (csum[i + k + 1] - csum[i + 1]) / k
    clear = pd.Series(nearest).rolling(2 * k + 1, center=True, min_periods=2 * k + 1).min().to_numpy()[i]
    gain = np.zeros(n)
    # Only on the pitch: outside it the ball is handled by people who are not tracked (ball
    # boys, a thrower stepping back), so an unexplained impulse there is expected and real.
    bx, by = ball["ball_m"][i, 0].astype(np.float64), ball["ball_m"][i, 1].astype(np.float64)
    on_pitch = (bx >= 0) & (bx <= C.PITCH_LENGTH) & (by >= 0) & (by <= C.PITCH_WIDTH)
    hit = (whole & on_pitch & (after - before >= ORPHAN_MIN_GAIN) & (before <= ORPHAN_MAX_SPEED_BEFORE) &
           (clear > ORPHAN_CLEAR_M))
    gain[i[hit]] = (after - before)[hit]

    untrusted = np.zeros(n, dtype=bool)
    impulse = np.zeros(n, dtype=bool)
    for a, b in runs(gain > 0):
        g = a + int(np.argmax(gain[a:b]))
        impulse[g] = True
        j, last_touch = g, None
        while j - 1 >= max(0, g - ORPHAN_MAX_EXTENT) and ok[j - 1] and period[j - 1] == period[g]:
            if in_reach[j - 1]:
                last_touch = j - 1
                break
            j -= 1
        start = j if last_touch is None else min(g, last_touch + 1 + ORPHAN_TOUCH_GUARD)
        e = g
        while e + 1 < min(n, g + ORPHAN_MAX_EXTENT) and ok[e + 1] and period[e + 1] == period[g] \
                and not in_reach[e + 1]:
            e += 1
        untrusted[start:e + 1] = True
    return untrusted, impulse


def pitch_geometry(ball_m, half, home_attacks_positive_x):
    L, W = C.PITCH_LENGTH, C.PITCH_WIDTH
    x, y = ball_m[:, 0].astype(np.float64), ball_m[:, 1].astype(np.float64)
    gl = np.minimum(x, L - x)          # signed: negative once past a goal line
    tl = np.minimum(y, W - y)
    boundary = np.minimum(gl, tl)
    cand = np.stack([x, L - x, y, W - y], axis=1)
    nearest = np.where(np.isfinite(x), np.nanargmin(np.where(np.isfinite(cand), cand, np.inf), axis=1), -1)
    out = boundary < 0
    out_clear = boundary < -OUT_CLEAR_MARGIN_M
    centre = np.hypot(x - L / 2, y - W / 2)
    corner = np.min(np.stack([np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)], axis=1), axis=1)

    def area(depth, width):
        in_y = np.abs(y - W / 2) <= width / 2
        res = np.full(len(x), -1, dtype=np.int8)
        res[in_y & (x >= 0) & (x <= depth)] = 0
        res[in_y & (x <= L) & (x >= L - depth)] = 1
        return res

    attacks_pos = np.zeros(len(x), dtype=bool)
    for h in (1, 2):
        attacks_pos[half == h] = bool(home_attacks_positive_x[str(h)])
    adv_home = np.where(attacks_pos, x, L - x)
    adv_home[half == 0] = np.nan

    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    return {
        "ball_goalline_dist": f32(gl), "ball_touchline_dist": f32(tl),
        "ball_boundary_dist": f32(boundary), "ball_nearest_boundary": nearest.astype(np.int8),
        "ball_out": out & np.isfinite(x), "ball_out_clear": out_clear & np.isfinite(x),
        "ball_centre_dist": f32(centre), "ball_corner_dist": f32(corner),
        "ball_in_penalty_area": area(PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH),
        "ball_in_goal_area": area(GOAL_AREA_DEPTH, GOAL_AREA_WIDTH),
        "ball_adv_home": f32(adv_home),
    }


# --------------------------------------------------------------------------------------------
# players

def tracklets(present):
    """Per (frame, slot): tracklet id, age, total length; plus a tracklet table."""
    n, s = present.shape
    tid = np.full((n, s), -1, dtype=np.int32)
    age = np.full((n, s), -1, dtype=np.int32)
    tlen = np.full((n, s), -1, dtype=np.int32)
    rows = []
    next_id = 0
    for slot in range(s):
        for start, stop in runs(present[:, slot]):
            tid[start:stop, slot] = next_id
            age[start:stop, slot] = np.arange(stop - start)
            tlen[start:stop, slot] = stop - start
            rows.append((next_id, slot, start, stop - 1, stop - start))
            next_id += 1
    table = pd.DataFrame(rows, columns=["tracklet", "slot", "first_frame", "last_frame", "length"])
    births = np.zeros(n, dtype=np.int16)
    deaths = np.zeros(n, dtype=np.int16)
    np.add.at(births, table["first_frame"].to_numpy(), 1)
    last = table["last_frame"].to_numpy()
    np.add.at(deaths, last[last < n - 1] + 1, 1)  # a death is the first frame without the track
    return tid, age, tlen, table, births, deaths


def eligibility(players_m, present, slot_team, tlen):
    """On (or just off) the pitch, and not a duplicate of a longer-lived teammate track."""
    L, W = C.PITCH_LENGTH, C.PITCH_WIDTH
    x, y = players_m[..., 0], players_m[..., 1]
    with np.errstate(invalid="ignore"):
        off = present & ((x < -PLAYER_FIELD_MARGIN_M) | (x > L + PLAYER_FIELD_MARGIN_M) |
                         (y < -PLAYER_FIELD_MARGIN_M) | (y > W + PLAYER_FIELD_MARGIN_M))
    candidate = present & ~off
    duplicate = np.zeros_like(present)
    n = present.shape[0]
    for side in (C.HOME, C.AWAY):
        cols = np.flatnonzero(slot_team == side)
        for start in range(0, n, 20000):
            sl = slice(start, min(n, start + 20000))
            p = players_m[sl][:, cols]
            ok = candidate[sl][:, cols]
            prio = np.where(ok, tlen[sl][:, cols], -1).astype(np.int64)
            d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=3)
            close = (d < DUPLICATE_DIST_M) & ok[:, :, None] & ok[:, None, :]
            k = len(cols)
            close[:, np.arange(k), np.arange(k)] = False
            # j is dropped if a close teammate i outranks it (longer tracklet, then lower slot)
            pi, pj = prio[:, :, None], prio[:, None, :]
            lower_i = (np.arange(k)[:, None] < np.arange(k)[None, :])[None]
            outranks = (pi > pj) | ((pi == pj) & lower_i)
            dropped = (close & outranks).any(axis=1)
            duplicate[sl, cols] = dropped
    eligible = candidate & ~duplicate
    return eligible, off, duplicate


# Swapped goalkeepers. The tracker sometimes hands a goalkeeper's track to the other
# team's slot (a new tracklet appears in the wrong team's goalkeeper slot), so the keeper looks
# like an opposition player. A team's goalkeeper cannot be inside the *opposing* penalty area
# while the ball is in that same area, so on those frames the slot is relabelled to the side
# that defends the area. Only the per-frame team used for nearest-player evidence changes; the
# fixed slot_team array and the tracking itself are untouched.
SWAP_MIN_FRAMES = 25 * 60


def goalkeeper_slots(players_m, present, slot_team, half, graded, home_attacks_positive_x):
    """{(half, side): slot} -- the slot whose median x sits closest to its own goal."""
    out = {}
    for h in (1, 2):
        d0 = C.HOME if bool(home_attacks_positive_x[str(h)]) else C.AWAY   # defends x = 0
        sel = (half == h) & graded
        for side in (C.HOME, C.AWAY):
            goal_x = 0.0 if side == d0 else C.PITCH_LENGTH
            best, best_d = -1, np.inf
            for slot in np.flatnonzero(slot_team == side):
                ok = sel & present[:, slot]
                if ok.sum() < SWAP_MIN_FRAMES:
                    continue
                d = float(np.median(np.abs(players_m[ok, slot, 0] - goal_x)))
                if d < best_d:
                    best, best_d = int(slot), d
            out[(h, side)] = best
    return out


def player_team_per_frame(players_m, present, eligible, slot_team, ball_area, half, graded,
                          home_attacks_positive_x):
    """(N, S) team of every slot on every frame: slot_team, except swapped goalkeepers."""
    L, W = C.PITCH_LENGTH, C.PITCH_WIDTH
    n = players_m.shape[0]
    team = np.broadcast_to(slot_team[None, :], (n, len(slot_team))).copy()
    gk = goalkeeper_slots(players_m, present, slot_team, half, graded, home_attacks_positive_x)
    swapped = 0
    for h in (1, 2):
        d0 = C.HOME if bool(home_attacks_positive_x[str(h)]) else C.AWAY
        for side in (C.HOME, C.AWAY):
            slot = gk[(h, side)]
            if slot < 0:
                continue
            x, y = players_m[:, slot, 0], players_m[:, slot, 1]
            in_y = np.abs(y - W / 2) <= PENALTY_AREA_WIDTH / 2
            for a, goal_defender in ((0, d0), (1, 1 - d0)):
                if goal_defender == side:
                    continue                      # its own area
                in_area = in_y & ((x >= 0) & (x <= PENALTY_AREA_DEPTH) if a == 0
                                  else (x <= L) & (x >= L - PENALTY_AREA_DEPTH))
                m = (half == h) & eligible[:, slot] & in_area & (ball_area == a)
                team[m, slot] = goal_defender
                swapped += int(m.sum())
    return team.astype(np.int8), swapped


def team_stats(players_m, players_speed, eligible, slot_team):
    n = players_m.shape[0]
    n_elig = np.zeros((n, 2), dtype=np.int8)
    centroid = np.full((n, 2, 2), np.nan, dtype=np.float32)
    mean_speed = np.full((n, 2), np.nan, dtype=np.float32)
    for side in (C.HOME, C.AWAY):
        cols = slot_team == side
        e = eligible[:, cols]
        cnt = e.sum(axis=1)
        n_elig[:, side] = cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in (0, 1):
                v = np.where(e, players_m[:, cols, k], 0.0).sum(axis=1) / cnt
                centroid[:, side, k] = np.where(cnt > 0, v, np.nan)
            sp = players_speed[:, cols]
            se = e & np.isfinite(sp)
            v = np.where(se, sp, 0.0).sum(axis=1) / se.sum(axis=1)
            mean_speed[:, side] = np.where(se.sum(axis=1) > 0, v, np.nan)
    se = eligible & np.isfinite(players_speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        all_mean = np.where(se, players_speed, 0.0).sum(axis=1) / se.sum(axis=1)
        slow = (se & (players_speed < SLOW_SPEED)).sum(axis=1) / se.sum(axis=1)
    all_mean = np.where(se.sum(axis=1) > 0, all_mean, np.nan).astype(np.float32)
    slow = np.where(se.sum(axis=1) > 0, slow, np.nan).astype(np.float32)
    return n_elig, n_elig > 11, centroid, mean_speed, all_mean, slow


# --------------------------------------------------------------------------------------------
# contact

def contact(ball_m, ball_v, ball_ok, players_m, players_v, players_speed, eligible, slot_team,
            team_frame=None):
    n = len(ball_m)
    diff = players_m - ball_m[:, None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    valid = eligible & ball_ok[:, None] & np.isfinite(dist)
    dist = np.where(valid, dist, np.inf)

    near_dist = np.full((n, 2), np.nan, dtype=np.float32)
    near_slot = np.full((n, 2), -1, dtype=np.int16)
    n_within = np.zeros((n, 2, len(CONTACT_RADII_M)), dtype=np.int8)
    if team_frame is None:
        team_frame = np.broadcast_to(slot_team[None, :], dist.shape)
    for side in (C.HOME, C.AWAY):
        d = np.where(team_frame == side, dist, np.inf)
        j = np.argmin(d, axis=1)
        best = d[np.arange(n), j]
        has = np.isfinite(best)
        near_dist[has, side] = best[has]
        near_slot[has, side] = j[has]
        for r, radius in enumerate(CONTACT_RADII_M):
            n_within[:, side, r] = (d < radius).sum(axis=1)

    has_any = np.isfinite(near_dist).any(axis=1)
    nd = np.where(np.isfinite(near_dist), near_dist, np.inf)
    nearest_side = np.where(has_any, np.argmin(nd, axis=1), -1).astype(np.int8)
    nearest_dist = np.where(has_any, nd.min(axis=1), np.nan).astype(np.float32)
    other = np.where(has_any, nd.max(axis=1), np.nan)
    margin = np.where(np.isfinite(other), other - nearest_dist, np.nan).astype(np.float32)

    rows = np.flatnonzero(has_any)
    slot = near_slot[rows, nearest_side[rows]]
    nearest_speed = np.full(n, np.nan, dtype=np.float32)
    nearest_speed[rows] = players_speed[rows, slot]
    rel = np.full(n, np.nan, dtype=np.float32)
    rel[rows] = np.linalg.norm(ball_v[rows] - players_v[rows, slot], axis=1)
    return {"near_dist": near_dist, "near_slot": near_slot, "nearest_side": nearest_side,
            "nearest_dist": nearest_dist, "nearest_margin": margin,
            "nearest_speed": nearest_speed, "nearest_rel_speed": rel, "n_within": n_within}


# --------------------------------------------------------------------------------------------
# Smart Tagging hints

def tagging(events, n):
    ev = events[events["graded_frames"] > 0]

    def side_paint(rows, flip):
        h = np.zeros(n, dtype=bool)
        a = np.zeros(n, dtype=bool)
        for side, start, end in rows[["side", "start_frame", "end_frame"]].itertuples(index=False):
            if side not in ("H", "A"):
                continue
            s = ({"H": "A", "A": "H"}[side]) if flip else side
            (h if s == "H" else a)[int(start):int(end)] = True
        return h, a

    h1, a1 = side_paint(ev[ev["family"] == "attack_phase"], flip=False)
    h2, a2 = side_paint(ev[ev["family"] == "defend_phase"], flip=True)  # Team = defending side
    h, a = h1 | h2, a1 | a2
    poss = np.full(n, NONE, dtype=np.int8)
    poss[h & ~a], poss[a & ~h], poss[h & a] = SIDE_H, SIDE_A, SIDE_BOTH

    hz, az = side_paint(ev[ev["family"] == "zone"], flip=False)
    zone = np.full(n, NONE, dtype=np.int8)
    zone[hz & ~az], zone[az & ~hz], zone[hz & az] = SIDE_H, SIDE_A, SIDE_BOTH

    def latest_active(rows):
        active = np.full(n, -1, dtype=np.int32)
        since = np.full(n, -1, dtype=np.int32)
        # painting in start order makes the latest-started instance win where they overlap
        for idx, start, end in sorted(zip(rows.index, rows["start_frame"], rows["end_frame"]),
                                      key=lambda r: (r[1], r[0])):
            active[int(start):int(end)] = idx
            since[int(start):int(end)] = np.arange(int(end) - int(start))
        return active, since

    sp, sp_since = latest_active(ev[ev["family"] == "set_piece"])
    goal, _ = latest_active(ev[ev["family"] == "goal"])
    return {"tag_poss_hint": poss, "tag_zone_side": zone, "sp_active": sp,
            "sp_since_start": sp_since, "goal_active": goal}


# --------------------------------------------------------------------------------------------
# main

def build(work_dir=None):
    t0 = time.time()
    checks = Checks()
    t = load_timeline(work_dir)
    n, slot_team = t.n_frames, t.slot_team.astype(np.int8)
    print("loaded s00 timeline: {} frames, {} slots".format(n, len(slot_team)))

    ev = {"period": t.period, "half": t.half, "graded": t.graded}

    # ball
    ball = clean_ball(t.ball_xy, t.period)
    ev.update(ball)
    ev.update(pitch_geometry(ball["ball_m"], t.half, t.meta["home_attacks_positive_x"]))

    # players
    players_m = C.to_metres(t.players_xy)
    present = t.players_present
    tid, age, tlen, table, births, deaths = tracklets(present)
    players_v = centred_velocity(players_m.astype(np.float64), present, tid, VEL_HALF_SPAN)
    eligible, off, duplicate = eligibility(players_m, present, slot_team, tlen)
    n_elig, overcount, centroid, team_speed, all_speed, slow = team_stats(
        players_m, t.players_speed, eligible, slot_team)
    # ball trust: discard stretches no player can explain, then clean the ball again
    untrusted = np.zeros(n, dtype=bool)
    orphan = np.zeros(n, dtype=bool)
    if BALL_TRUST_ON:
        untrusted, orphan = untrusted_ball(ball, players_m, eligible, t.period)
        if BALL_TRUST_DROP and untrusted.any():
            ball = clean_ball(t.ball_xy, t.period, drop=untrusted)
            ev.update(ball)
            ev.update(pitch_geometry(ball["ball_m"], t.half, t.meta["home_attacks_positive_x"]))
            untrusted &= ~ball["ball_ok"]      # a stretch short enough to be bridged stays usable
    ev["ball_untrusted"] = untrusted
    ev["ball_orphan_impulse"] = orphan
    print("ball trust: {} unexplained impulses, {} frames flagged (graded: {}, {})".format(
        int(orphan.sum()), int(untrusted.sum()), int((orphan & t.graded).sum()),
        int((untrusted & t.graded).sum())))

    table["side"] = np.where(slot_team[table["slot"].to_numpy()] == C.HOME, "H", "A")
    table["starts_on_block_boundary"] = (table["first_frame"] % TRACKER_BLOCK == 0)
    ev.update({
        "players_m": players_m, "players_v": players_v, "players_speed": t.players_speed,
        "players_present": present, "players_eligible": eligible,
        "players_off_pitch": off, "players_duplicate": duplicate, "slot_team": slot_team,
        "tracklet_id": tid, "tracklet_age": age, "tracklet_len": tlen,
        "track_births": births, "track_deaths": deaths,
        "block_boundary": (np.arange(n) % TRACKER_BLOCK) == 0,
        "n_eligible": n_elig, "team_overcount": overcount, "team_centroid_m": centroid,
        "team_mean_speed": team_speed, "mean_speed_all": all_speed, "slow_share": slow,
        "mean_speed_all_2s": rolling_mean(all_speed, C.FPS),
    })

    # contact; swapped goalkeepers are relabelled per frame first
    players_team, n_swapped = player_team_per_frame(
        players_m, present, eligible, slot_team, ev["ball_in_penalty_area"], t.half, t.graded,
        t.meta["home_attacks_positive_x"])
    print("swapped-goalkeeper frames relabelled: {}".format(n_swapped))
    ev["players_team"] = players_team
    ev.update(contact(ball["ball_m"], ball["ball_v"], ball["ball_ok"], players_m, players_v,
                      t.players_speed, eligible, slot_team, players_team))

    # tagging and camera
    ev.update(tagging(t.events, n))
    ev.update({"hom_valid": t.hom_valid, "hom_error": t.hom_error,
               "hom_reliable": t.hom_valid & (t.hom_error <= HOM_RELIABLE_MAX_ERROR)})

    # -- checks ---------------------------------------------------------------------------------
    g = t.graded
    for name, arr in ev.items():
        if name == "slot_team":
            continue
        if arr.shape[0] != n:
            checks.add("array {} has N rows".format(name), False, str(arr.shape))
    checks.add("all arrays have one row per frame", not checks.failed())
    checks.add("cleaned ball only where tracked or bridged",
               bool(np.all(np.isfinite(ev["ball_m"][:, 0]) == ev["ball_ok"])))
    checks.add("nearest player only where the ball is usable",
               bool(np.all((ev["nearest_side"] >= 0) <= ev["ball_ok"])))
    checks.add("eligible implies present", bool(np.all(eligible <= present)))
    jumps_kept = int((ev["ball_jump"] & g).sum())
    checks.add("ball spikes removed", True, "{} spikes, {} frames bridged, {} jumps kept (graded)".format(
        int((ball["ball_spike"] & g).sum()), int((ball["ball_interp"] & g).sum()), jumps_kept),
        severity="warning")
    checks.add("graded frames with a usable ball", True, "{:.1f}%".format(
        100 * ev["ball_ok"][g].mean()), severity="warning")
    checks.add("graded frames with a team over 11 eligible players", True,
               "home {:.1f}%, away {:.1f}% -- extra tracks (referee, staff) not removable here".format(
                   100 * overcount[g, 0].mean(), 100 * overcount[g, 1].mean()), severity="warning")
    births_on_block = int(table["starts_on_block_boundary"].sum())
    checks.add("tracklet starts on tracker block boundaries", True,
               "{} of {} tracklets start on a multiple of {} frames".format(
                   births_on_block, len(table), TRACKER_BLOCK), severity="warning")

    print("\nchecks")
    checks.report()
    failed = checks.failed("error")
    if failed:
        raise SystemExit("\n{} critical check(s) failed; nothing written.".format(len(failed)))

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    np.savez(os.path.join(out, "evidence.npz"), **ev)
    table.to_csv(os.path.join(out, "tracklets.csv"), index=False)

    r2 = list(CONTACT_RADII_M)
    frames = pd.DataFrame({
        "frame": np.arange(n), "period": np.array(C.PERIOD_NAMES)[t.period], "graded": g.astype(np.int8),
        "ball_ok": ev["ball_ok"].astype(np.int8), "ball_interp": ev["ball_interp"].astype(np.int8),
        "ball_x_m": ev["ball_m"][:, 0], "ball_y_m": ev["ball_m"][:, 1],
        "ball_speed": ev["ball_speed"], "ball_accel": ev["ball_accel"], "ball_turn_deg": ev["ball_turn_deg"],
        "ball_still_share": ev["ball_still_share"], "ball_disp_1s": ev["ball_disp_1s"],
        "ball_since_seen": ev["ball_since_seen"],
        "ball_until_seen": ev["ball_until_seen"], "ball_gap_len": ev["ball_gap_len"],
        "ball_boundary_dist": ev["ball_boundary_dist"],
        "ball_nearest_boundary": np.where(ev["ball_nearest_boundary"] >= 0,
                                          np.array(BOUNDARY_NAMES)[ev["ball_nearest_boundary"].clip(0)], ""),
        "ball_out": ev["ball_out"].astype(np.int8), "ball_adv_home": ev["ball_adv_home"],
        "nearest_side": np.where(ev["nearest_side"] >= 0, np.array(["H", "A"])[ev["nearest_side"].clip(0)], ""),
        "nearest_dist": ev["nearest_dist"], "nearest_margin": ev["nearest_margin"],
        "near_dist_H": ev["near_dist"][:, 0], "near_dist_A": ev["near_dist"][:, 1],
        "nearest_rel_speed": ev["nearest_rel_speed"],
        "H_within_{:g}m".format(r2[0]): ev["n_within"][:, 0, 0], "A_within_{:g}m".format(r2[0]): ev["n_within"][:, 1, 0],
        "n_elig_H": n_elig[:, 0], "n_elig_A": n_elig[:, 1],
        "mean_speed_all": all_speed, "slow_share": slow,
        "track_births": births, "block_boundary": ev["block_boundary"].astype(np.int8),
        "tag_poss_hint": ev["tag_poss_hint"], "tag_zone_side": ev["tag_zone_side"],
        "sp_active": ev["sp_active"], "goal_active": ev["goal_active"],
        "hom_reliable": ev["hom_reliable"].astype(np.int8),
    })
    frames.round(3).to_csv(os.path.join(out, "evidence_frames.csv"), index=False)

    gm = lambda a: float(np.nanmedian(a[g])) if np.isfinite(a[g]).any() else None  # noqa: E731
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE,
        "parameters": {k: v for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, tuple, str))},
        "stats_graded": {
            "ball_ok_pct": 100 * float(ev["ball_ok"][g].mean()),
            "ball_spikes": int((ball["ball_spike"] & g).sum()),
            "ball_bridged_frames": int((ball["ball_interp"] & g).sum()),
            "ball_jumps": jumps_kept,
            "ball_orphan_impulses": int((ev["ball_orphan_impulse"] & g).sum()),
            "ball_untrusted_frames": int((ev["ball_untrusted"] & g).sum()),
            "ball_out_frames": int((ev["ball_out"] & g).sum()),
            "ball_speed_median": gm(ev["ball_speed"]),
            "nearest_dist_median": gm(ev["nearest_dist"]),
            "home_overcount_pct": 100 * float(overcount[g, 0].mean()),
            "away_overcount_pct": 100 * float(overcount[g, 1].mean()),
            "duplicates_removed": int((duplicate & g[:, None]).sum()),
            "off_pitch_tracks": int((off & g[:, None]).sum()),
            "n_tracklets": int(len(table)),
            "tracklets_on_block_boundary": births_on_block,
            "tag_poss_hint_coverage_pct": 100 * float((ev["tag_poss_hint"][g] >= 0).mean()),
            "tag_poss_hint_conflict_pct": 100 * float((ev["tag_poss_hint"][g] == SIDE_BOTH).mean()),
        },
        "boundary_names": list(BOUNDARY_NAMES),
        "arrays": EVIDENCE_ARRAYS,
        "checks": checks.items,
    })
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


class Evidence(object):
    """Read-only view of work/s01 for later stages: e.ball_m, e.near_dist, e.meta ..."""

    def __init__(self, work_dir=None):
        folder = os.path.join(work_dir or C.WORK_DIR, STAGE)
        path = os.path.join(folder, "evidence.npz")
        if not os.path.isfile(path):
            raise IOError("{} not found -- run: python -m pipeline4.s01_evidence".format(path))
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.meta = C.read_json(os.path.join(folder, "meta.json"))
        self.tracklets = pd.read_csv(os.path.join(folder, "tracklets.csv"))
        self.n_frames = int(len(self.period))


def load_evidence(work_dir=None):
    return Evidence(work_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 01: frame-level evidence.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    args = parser.parse_args(argv)
    build(args.work_dir)


if __name__ == "__main__":
    main()
