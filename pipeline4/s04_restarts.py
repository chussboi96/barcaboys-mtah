"""Stage 04 -- restart type and restarting team (rules 8, 10, 11, 12, and the prior of rule 19).

For every stoppage from stage 03, plus the opening kickoff of any half graded from its kickoff:

  type   kickoff, throw_in, corner, goal_kick, free_kick, or end_of_half. Decided, in order, by:
         the stoppage itself (start-of-half / confirmed goal -> kickoff, disallowed goal -> by
         location), a matching Smart Tagging set piece, the line the ball crossed (touchline ->
         throw-in; goal line -> corner if the defending side touched it last, else goal kick),
         then where the ball is at the restart (goal area, corner, touchline, centre spot).

  team   a vote. Each piece of evidence adds a signed weight (+ home, - away); the side is the
         sign of the sum and |sum| is the confidence. Nearest-player distance is one vote among
         several and never decides alone (rule 12):
           kickoff taker        team with a player on the centre spot in the line-up     3.0
           conceding side       the side that did not score kicks off after a goal       2.0
           second-half kickoff  the side that did not kick off the first half            2.0
           Smart Tagging side   the set-piece tag's Team, when given                     2.0
           goal-line geometry   corner -> attackers of that goal, goal kick -> defenders 2.5
           last-touch rule      throw-in to the side that did not touch it last          1.5
                                (0.5 if that touch was contested)
           disallowed goal      the free kick goes to the side that did not "score"      2.0
           taker                side of the player at the ball as it is struck           1.0
                                (0.3 if an opponent is within a metre of that player)
           first contact        side of the first touch or control after the restart     1.0
           possession before    side in possession before a free kick (fouled side)      0.5

It also re-reads the goal analysis of stage 03 and works out who scored from the tracking alone:
the side that kicks off conceded. The inferred score is reported; with --expected-score it is
compared against a result you pass in. The comparison is a report only -- it never changes a
label, because a known result is human-produced information about the match (RULES.md).

    python -m pipeline4.s04_restarts [--expected-score 2-1]

Outputs, in work/s04/:

    restarts.csv     one row per restart: type, team, confidence, every vote, location
    goals.csv        stage-03 goals with the kicking side, scorer and the inferred score
    restarts.npz     per-frame arrays (see RESTART_ARRAYS)
    p2d_labels.csv   diagnostic: D from stage 03; each live stretch starts with the restarting
                     team and then follows pass-1 changes of side (stage 02)
    meta.json        parameters, statistics and checks
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
from pipeline4.s01_evidence import load_evidence, runs  # noqa: E402
from pipeline4.s02_touches import load_contacts  # noqa: E402
from pipeline4.s03_stoppages import load_deadball, set_piece_tags  # noqa: E402

STAGE = "s04"
F = C.FPS
L, W = C.PITCH_LENGTH, C.PITCH_WIDTH

# --------------------------------------------------------------------------------------------
# parameters

# vote weights (see the module docstring)
W_KICKOFF_TAKER = 3.0
W_CONCEDING = 2.0
W_SECOND_HALF = 2.0
W_TAG = 2.0
W_GOAL_LINE = 2.5
W_LAST_TOUCH, W_LAST_TOUCH_CONTESTED = 1.5, 0.5
W_DISALLOWED = 2.0
W_TAKER, W_TAKER_CROWDED = 1.0, 0.3
W_FIRST_CONTACT = 1.0
W_POSSESSION_BEFORE = 0.5

TAG_MATCH_BEFORE = 5 * F        # a set-piece tag starting from 5 s before the onset ...
TAG_MATCH_AFTER = F             # ... up to 1 s after the restart belongs to the stoppage
LAST_TOUCH_MAX_AGE = 5 * F      # the last contact must be within 5 s of the onset
TAKER_WINDOW = F // 2           # the taker is at the ball in the 0.5 s up to the restart
TAKER_CROWDED_M = 1.0
FIRST_CONTACT_SKIP = 3          # frames after the restart that still belong to the strike
FIRST_CONTACT_WINDOW = 4 * F
LOCATION_SEARCH = 2 * F         # ball unseen at the restart: first sighting within 2 s after
CORNER_RADIUS_M = 3.0           # corner arc is 1 m; tracking and smoothing add the rest
TOUCHLINE_NEAR_M = 2.0

TAG_TYPES = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick",
             "free kick": "free_kick", "kick off (start)": "kickoff", "kick off (after goal)": "kickoff"}

RESTART_ARRAYS = {
    "restart_frame": "(N,) bool   a restart happens on this frame",
    "restart_side": "(N,) int8    restarting side on restart frames, -1 elsewhere/unknown",
    "live_segment": "(N,) int32   id of the live stretch (restart to next stoppage), -1 dead/outside",
    "segment_prior_side": "(N,) int8 restarting side of the live stretch the frame is in, -1 unknown",
    "segment_restart_type": "(N,) int8 index into RESTART_TYPES for that stretch, -1",
}
RESTART_TYPES = ("kickoff", "throw_in", "corner", "goal_kick", "free_kick", "end_of_half")


# --------------------------------------------------------------------------------------------
# geometry helpers

def defender_of_line(x_line_is_zero, half, home_attacks_positive_x):
    """Side that defends the goal at x = 0 (True) or x = L (False) in a half."""
    home_pos = bool(home_attacks_positive_x[str(half)])
    home_defends_zero = home_pos           # attacking +x means defending the goal at x = 0
    if x_line_is_zero:
        return C.HOME if home_defends_zero else C.AWAY
    return C.AWAY if home_defends_zero else C.HOME


def restart_location(e, restart, onset):
    """Ball position at the restart, else first sighting within 2 s, else last sighting before."""
    n = e.n_frames
    if 0 <= restart < n and e.ball_ok[restart]:
        return e.ball_m[restart].astype(float), "at_restart"
    f = np.arange(restart, min(restart + LOCATION_SEARCH, n))
    ok = f[e.ball_ok[f]]
    if len(ok):
        return e.ball_m[ok[0]].astype(float), "first_seen_after"
    f = np.arange(max(onset, 0), min(restart, n))
    ok = f[e.ball_ok[f]]
    if len(ok):
        return e.ball_m[ok[-1]].astype(float), "last_seen_before"
    return np.array([np.nan, np.nan]), "unknown"


def classify_location(xy):
    """Restart kind implied by where the ball is: goal_kick / corner / throw_in / free_kick / None.

    A ball at the centre spot is not called a kickoff here: a kickoff only follows a goal or opens
    a half, and those come from stage 03, not from where the ball happens to be.
    """
    x, y = xy
    if not np.isfinite(x):
        return None, None
    in_goal_area = abs(y - W / 2) <= 18.32 / 2 + 1.0 and (x <= 5.5 + 1.0 or x >= L - 5.5 - 1.0)
    corner = min(np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W))
    touch = min(abs(y), abs(W - y))
    if corner <= CORNER_RADIUS_M:
        return "corner", x < L / 2
    if in_goal_area:
        return "goal_kick", x < L / 2
    if touch <= TOUCHLINE_NEAR_M or y < 0 or y > W:
        return "throw_in", None
    return "free_kick", None


def centre_taker(e, lo, hi):
    """Side with the eligible player closest to the centre spot over [lo, hi), and that distance."""
    lo, hi = max(lo, 0), min(hi, e.n_frames)
    if hi <= lo:
        return None, np.nan
    centre = np.array([L / 2, W / 2], dtype=np.float32)
    d = np.linalg.norm(e.players_m[lo:hi] - centre, axis=2)
    d = np.where(e.players_eligible[lo:hi], d, np.inf)
    best = {}
    team = e.players_team[lo:hi]
    for side in (C.HOME, C.AWAY):
        best[side] = float(np.median(np.where(team == side, d, np.inf).min(axis=1)))
    side = min(best, key=best.get)
    return (side, best[side]) if np.isfinite(best[side]) else (None, np.nan)


def ball_taker(e, xy, lo, hi):
    """Side of the eligible player nearest to the restart spot over [lo, hi); crowded flag."""
    lo, hi = max(lo, 0), min(hi, e.n_frames)
    if hi <= lo or not np.isfinite(xy[0]):
        return None, False
    d = np.linalg.norm(e.players_m[lo:hi] - xy.astype(np.float32), axis=2)
    d = np.where(e.players_eligible[lo:hi], d, np.inf)
    team = e.players_team[lo:hi]
    med = {s: float(np.median(np.where(team == s, d, np.inf).min(axis=1))) for s in (C.HOME, C.AWAY)}
    side = min(med, key=med.get)
    if not np.isfinite(med[side]):
        return None, False
    return side, bool(med[1 - side] - med[side] < TAKER_CROWDED_M)


# --------------------------------------------------------------------------------------------
# one restart

def analyse(row, t, e, c, sp, goals_by_restart, first_half_kicker):
    """Type, side, confidence and votes for one stoppage row (a dict)."""
    hp = t.meta["home_attacks_positive_x"]
    half, onset, restart = int(row["half"]), int(row["onset"]), int(row["restart"])
    sources = set(str(row["source"]).split("+"))
    votes = []

    def vote(name, side, weight):
        if side is None or side < 0:
            return
        votes.append((name, int(side), float(weight)))

    blocks = {h: (a, b) for h, a, b in C.games.half_video_blocks(C.GAME)}
    if restart > blocks[half][1]:
        return {"type": "end_of_half", "type_source": "stoppage_runs_to_half_end",
                "side": -1, "confidence": 0.0, "votes": [], "loc": (np.nan, np.nan), "loc_how": "",
                "tag": ""}

    xy, loc_how = restart_location(e, restart, onset)
    loc_type, loc_zero_line = classify_location(xy)

    # matching Smart Tagging set piece: latest start before the restart
    tags = sp[(sp["start_frame"] >= onset - TAG_MATCH_BEFORE) & (sp["start_frame"] < restart + TAG_MATCH_AFTER)]
    tag = tags.iloc[-1] if len(tags) else None
    tag_type = TAG_TYPES.get(str(tag["type"]).lower()) if tag is not None else None
    tag_desc = "{}:{}:{}".format(int(tags.index[-1]), tag["type"], tag["side"] or "N/A") if tag is not None else ""

    goal = goals_by_restart.get(restart)

    # -- type -----------------------------------------------------------------------------------
    if "kickoff_start_of_half" in sources or (goal is not None and goal["status"] == "goal"):
        rtype, tsrc = "kickoff", "kickoff_stoppage"
    elif goal is not None and goal["status"] == "disallowed_goal":
        rtype = loc_type if loc_type in ("goal_kick", "free_kick", "corner") else "free_kick"
        tsrc = "disallowed_goal+location"
    elif tag_type is not None and tag_type != "kickoff":
        rtype, tsrc = tag_type, "smart_tagging"
    elif row["onset_how"] in ("ball_out", "ball_out_inferred") and row["boundary_at_onset"]:
        b = row["boundary_at_onset"]
        if b.startswith("touchline"):
            rtype, tsrc = "throw_in", "line_crossed"
        else:
            zero = b == "goal_line_x0"
            defender = defender_of_line(zero, half, hp)
            lc = row["last_contact_side"]
            if lc in ("H", "A"):
                rtype = "corner" if (C.HOME if lc == "H" else C.AWAY) == defender else "goal_kick"
                tsrc = "line_crossed+last_touch"
            elif loc_type in ("corner", "goal_kick"):
                rtype, tsrc = loc_type, "line_crossed+location"
            else:
                rtype, tsrc = "goal_kick", "line_crossed"
    elif loc_type is not None:
        rtype, tsrc = loc_type, "location"
    else:
        rtype, tsrc = "free_kick", "default"

    # -- votes ----------------------------------------------------------------------------------
    if rtype == "kickoff":
        side, dist = centre_taker(e, restart - F, restart + 1)
        vote("kickoff_taker", side, W_KICKOFF_TAKER)
        if goal is not None and goal.get("goals_tag_side") in ("H", "A"):
            vote("conceding_side", C.AWAY if goal["goals_tag_side"] == "H" else C.HOME, W_CONCEDING)
        if "kickoff_start_of_half" in sources and half == 2 and first_half_kicker is not None:
            vote("second_half_kickoff", 1 - first_half_kicker, W_SECOND_HALF)
    else:
        if tag is not None and tag["side"] in ("H", "A"):
            vote("smart_tagging_side", C.HOME if tag["side"] == "H" else C.AWAY, W_TAG)
        if rtype in ("corner", "goal_kick"):
            zero = (row["boundary_at_onset"] == "goal_line_x0") if str(row["boundary_at_onset"]).startswith("goal_line") \
                else (loc_zero_line if loc_zero_line is not None else (xy[0] < L / 2 if np.isfinite(xy[0]) else None))
            if zero is not None:
                defender = defender_of_line(bool(zero), half, hp)
                vote("goal_line_geometry", defender if rtype == "goal_kick" else 1 - defender, W_GOAL_LINE)
        if rtype == "throw_in":
            lc, lcf = row["last_contact_side"], int(row["last_contact_frame"])
            if lc in ("H", "A") and lcf >= 0 and onset - lcf <= LAST_TOUCH_MAX_AGE:
                w = W_LAST_TOUCH_CONTESTED if str(row["last_contact_contested"]) == "True" else W_LAST_TOUCH
                vote("last_touch_rule", C.AWAY if lc == "H" else C.HOME, w)
        if goal is not None and goal["status"] == "disallowed_goal" and goal.get("goals_tag_side") in ("H", "A"):
            vote("disallowed_goal", C.AWAY if goal["goals_tag_side"] == "H" else C.HOME, W_DISALLOWED)
        side, crowded = ball_taker(e, xy, restart - TAKER_WINDOW, restart + 1)
        vote("taker_at_ball", side, W_TAKER_CROWDED if crowded else W_TAKER)
        if rtype == "free_kick":
            before = c.p1_side[max(onset - 1, 0)]
            vote("possession_before", before, W_POSSESSION_BEFORE)

    # first contact after the restart (all types)
    f = np.arange(restart + FIRST_CONTACT_SKIP, min(restart + FIRST_CONTACT_WINDOW, e.n_frames))
    contact = c.contact_side[f]
    hit = np.flatnonzero(contact >= 0)
    if len(hit):
        vote("first_contact_after", int(contact[hit[0]]), W_FIRST_CONTACT)

    score = sum(w if s == C.HOME else -w for _, s, w in votes)
    side = C.HOME if score > 0 else C.AWAY if score < 0 else -1
    return {"type": rtype, "type_source": tsrc, "side": side, "confidence": round(abs(score), 2),
            "votes": votes, "loc": xy, "loc_how": loc_how, "tag": tag_desc,
            "loc_type": loc_type or ""}


# --------------------------------------------------------------------------------------------
# main

def build(work_dir=None, expected_score=None):
    t0 = time.time()
    checks = Checks()
    t = load_timeline(work_dir)
    e = load_evidence(work_dir)
    c = load_contacts(work_dir)
    d = load_deadball(work_dir)
    n = e.n_frames
    g = e.graded
    sp = set_piece_tags(t)
    blocks = C.games.half_video_blocks(C.GAME)
    print("loaded s00-s03: {} stoppages".format(len(d.stoppages)))

    goals = d.goals.copy()
    goals_by_restart = {}
    for _, r in goals.iterrows():
        if pd.notna(r.get("restart")):
            goals_by_restart[int(r["restart"])] = r.to_dict()

    rows = []
    # opening kickoff of a half graded from its kickoff (no stoppage precedes it)
    ko_tags = t.events[(t.events["family"] == "set_piece") & (t.events["type"].str.lower() == "kick off (start)")]
    first_half_kicker = None
    for h, first, last in blocks:
        if ((ko_tags["start_frame"] <= first) & (ko_tags["end_frame"] > first)).any():
            side, dist = centre_taker(e, first, first + F // 2)
            votes = [("kickoff_taker", side, W_KICKOFF_TAKER)] if side is not None else []
            f = np.arange(first + FIRST_CONTACT_SKIP, first + FIRST_CONTACT_WINDOW)
            hit = np.flatnonzero(c.contact_side[f] >= 0)
            if len(hit):
                votes.append(("first_contact_after", int(c.contact_side[f][hit[0]]), W_FIRST_CONTACT))
            sc = sum(w if s == C.HOME else -w for _, s, w in votes)
            s_ = C.HOME if sc > 0 else C.AWAY if sc < 0 else -1
            if h == 1:
                first_half_kicker = s_ if s_ >= 0 else None
            rows.append({"stoppage": -1, "half": h, "onset": first, "restart": first, "type": "kickoff",
                         "type_source": "half_graded_from_kickoff", "side": s_, "confidence": abs(sc),
                         "votes": votes, "loc": (np.nan, np.nan), "loc_how": "", "tag": "", "loc_type": ""})

    st = d.stoppages
    # first-half kicker must be known before the second-half kickoff is analysed
    for i, r in st.iterrows():
        res = analyse(r, t, e, c, sp, goals_by_restart, first_half_kicker)
        res.update({"stoppage": int(r["stoppage"]), "half": int(r["half"]), "onset": int(r["onset"]),
                    "restart": int(r["restart"])})
        rows.append(res)
        if res["type"] == "kickoff" and "kickoff_start_of_half" in str(r["source"]) and int(r["half"]) == 1:
            first_half_kicker = res["side"] if res["side"] >= 0 else first_half_kicker

    rows.sort(key=lambda x: (x["restart"], x["stoppage"]))
    table = pd.DataFrame([{
        "stoppage": x["stoppage"], "half": x["half"], "onset": x["onset"], "restart": x["restart"],
        "type": x["type"], "type_source": x["type_source"],
        "side": "H" if x["side"] == C.HOME else "A" if x["side"] == C.AWAY else "",
        "confidence": x["confidence"],
        "votes": " ".join("{}:{}{:.1f}".format(nm, "+" if s == C.HOME else "-", w) for nm, s, w in x["votes"]),
        "restart_x_m": round(float(x["loc"][0]), 2) if np.isfinite(x["loc"][0]) else np.nan,
        "restart_y_m": round(float(x["loc"][1]), 2) if np.isfinite(x["loc"][1]) else np.nan,
        "location_how": x["loc_how"], "location_says": x.get("loc_type", ""), "smart_tagging": x["tag"],
    } for x in rows])

    # -- goals: who scored, from the kicking side --------------------------------------------
    restart_side = {int(r["restart"]): r["side"] for _, r in table.iterrows()}
    goals["kicking_side_tracking"] = [restart_side.get(int(r["restart"]), "") if pd.notna(r.get("restart")) else ""
                                      for _, r in goals.iterrows()]
    goals["scorer_inferred"] = np.where(goals["status"] == "goal",
                                        goals["kicking_side_tracking"].map({"H": "A", "A": "H"}).fillna(""), "")
    goals["scorer_agrees_with_goals_tag"] = np.where(goals["status"] == "goal",
                                                     goals["scorer_inferred"] == goals["goals_tag_side"], "")
    score_h = int((goals["scorer_inferred"] == "H").sum())
    score_a = int((goals["scorer_inferred"] == "A").sum())
    inferred = "{}-{}".format(score_h, score_a)

    # -- per-frame arrays -----------------------------------------------------------------------
    restart_frame = np.zeros(n, dtype=bool)
    rside = np.full(n, -1, dtype=np.int8)
    seg = np.full(n, -1, dtype=np.int32)
    prior = np.full(n, -1, dtype=np.int8)
    rtype_arr = np.full(n, -1, dtype=np.int8)
    live = g & ~d.dead
    for k, x in enumerate(rows):
        r = x["restart"]
        if not (0 <= r < n) or x["type"] == "end_of_half":
            continue
        restart_frame[r] = True
        rside[r] = x["side"]
    for k, (a, b) in enumerate(runs(live)):
        seg[a:b] = k
        # the restart that opens this live stretch: the latest restart at or before its start
        cand = [x for x in rows if x["restart"] <= a and x["type"] != "end_of_half" and a - x["restart"] <= F]
        if cand:
            x = cand[-1]
            prior[a:b] = x["side"]
            rtype_arr[a:b] = RESTART_TYPES.index(x["type"])

    # diagnostic labels: restart side, then pass-1 changes of side within the stretch
    labels = np.where(~g, "X", np.where(d.dead, "D", "?")).astype("<U1")
    for a, b in runs(live):
        state = prior[a]
        if state < 0:
            state = c.p1_side[a]
        for f in range(a, b):
            if c.p1_change[f] and f > a:
                state = c.p1_side[f]
            labels[f] = "H" if state == C.HOME else "A" if state == C.AWAY else "X"

    # -- checks ---------------------------------------------------------------------------------
    known = table[table["type"] != "end_of_half"]
    unknown_side = int((known["side"] == "").sum())
    checks.add("every restart has a type", bool(table["type"].notna().all()))
    checks.add("restarts with a team", unknown_side == 0, "{} of {} without a side".format(unknown_side, len(known)),
               severity="warning")
    checks.add("every live stretch has a restart prior", bool((prior[live] >= 0).mean() > 0.95),
               "{:.1f}% of live frames".format(100 * float((prior[live] >= 0).mean())), severity="warning")
    checks.add("goals inferred from kickoffs", True, "score {} (H-A) from {} standing goals; {} disallowed".format(
        inferred, int((goals["status"] == "goal").sum()), int((goals["status"] == "disallowed_goal").sum())),
        severity="warning")
    if expected_score:
        checks.add("inferred score matches --expected-score (report only)", inferred == expected_score,
                   "inferred {}, expected {}".format(inferred, expected_score), severity="warning")
    agree = goals.loc[goals["status"] == "goal", "scorer_agrees_with_goals_tag"]
    checks.add("kicking side agrees with the GOALS tag (scorer = other side)", bool(agree.astype(bool).all()) if len(agree) else True,
               "{} of {}".format(int(agree.astype(bool).sum()), len(agree)), severity="warning")
    print("\nchecks")
    checks.report()
    if checks.failed("error"):
        raise SystemExit("\ncritical check failed; nothing written.")

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    table.to_csv(os.path.join(out, "restarts.csv"), index=False)
    goals.to_csv(os.path.join(out, "goals.csv"), index=False)
    np.savez(os.path.join(out, "restarts.npz"), restart_frame=restart_frame, restart_side=rside,
             live_segment=seg, segment_prior_side=prior, segment_restart_type=rtype_arr)
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(os.path.join(out, "p2d_labels.csv"), index=False)
    stats = {
        "restarts": int(len(known)),
        "types": known["type"].value_counts().to_dict(),
        "type_sources": known["type_source"].value_counts().to_dict(),
        "sides": known["side"].replace("", "unknown").value_counts().to_dict(),
        "confidence_median": float(known["confidence"].median()) if len(known) else 0.0,
        "low_confidence_restarts": int((known["confidence"] < 1.5).sum()),
        "inferred_score_H_A": inferred,
        "goals_standing": int((goals["status"] == "goal").sum()),
        "goals_disallowed": int((goals["status"] == "disallowed_goal").sum()),
    }
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE,
        "parameters": {k: v for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, tuple, str))},
        "restart_types": list(RESTART_TYPES), "stats": stats, "arrays": RESTART_ARRAYS,
        "checks": checks.items, "expected_score_reported_only": expected_score,
        "note": "p2d_labels.csv is a diagnostic: stage-03 D, restart side, then pass-1 changes.",
    })
    print("\nstats")
    for k, v in stats.items():
        print("  {:<26} {}".format(k, v))
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


class Restarts(object):
    """Read-only view of work/s04: r.restarts, r.goals, r.segment_prior_side ..."""

    def __init__(self, work_dir=None):
        folder = os.path.join(work_dir or C.WORK_DIR, STAGE)
        path = os.path.join(folder, "restarts.npz")
        if not os.path.isfile(path):
            raise IOError("{} not found -- run: python -m pipeline4.s04_restarts".format(path))
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.restarts = pd.read_csv(os.path.join(folder, "restarts.csv"), keep_default_na=False)
        self.goals = pd.read_csv(os.path.join(folder, "goals.csv"))
        self.meta = C.read_json(os.path.join(folder, "meta.json"))


def load_restarts(work_dir=None):
    return Restarts(work_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 04: restart type and restarting team.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    parser.add_argument("--expected-score", default=None, metavar="H-A",
                        help="known final score, e.g. 2-1, compared against the inferred one (report only)")
    args = parser.parse_args(argv)
    build(args.work_dir, args.expected_score)


if __name__ == "__main__":
    main()
