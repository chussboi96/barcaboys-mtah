"""Stage 00 -- the frame timeline (rules 1 and 2).

One record per video frame of the tactical recording, indexed 0..199318 exactly like a submission
row, with every provided asset aligned to it:

    ATD (tracking)     frame_id 1..199319        video_frame = frame_id - 1
    ball, players      read from the ATD line of that frame
    homography         frame 0..199318           already a video frame
    Smart Tagging      seconds from video start  video_frame = round(seconds * 25), end exclusive

and every frame classified by period (pre-match, 1st half, half-time, 2nd half, post-match) from
the published half boundaries in games.py. Frames outside the two halves are never scored and get
the label X here; every graded frame is left UNDECIDED ("?") for later stages. That includes the
graded frames before the 2nd-half kickoff, which are dead ball -- not X -- and are resolved by the
stoppage stage.

Nothing is cleaned, smoothed or interpreted here. Raw values go in, raw values come out, aligned.
The only judgement calls are checks: every assumption the alignment rests on is tested against
the files and reported, and a failed critical check stops the run instead of writing a timeline
that is quietly wrong.

    python -m pipeline4.s00_timeline --data-dir "D:\\More Than A Hack 2027 - Initial Challenge"

Outputs, in work/s00/:

    timeline.npz               the arrays (see TIMELINE_ARRAYS below)
    smart_tagging_events.csv   one row per Smart Tagging instance, in video frames
    periods.csv                the five periods with first/last frame and summary statistics
    frames.csv                 a light per-frame table for eyeballing in a spreadsheet
    meta.json                  teams, slot map, attack directions, file paths, every check

work/ is gitignored: frame-level tracking derived from the gated data must not be published.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline4 import common as C  # noqa: E402
from pipeline4.common import games  # noqa: E402

STAGE = "s00"

# Lines of raw ATD parsed per batch. Bounds peak memory to a few hundred MB on any machine.
ATD_CHUNK_LINES = 20000

HOMOGRAPHY_MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]

# Smart Tagging code -> family. Matched case-insensitively. The defending codes describe the same
# intervals as their attacking mirror, with the Team tag naming the defending side; the zone codes
# say where the ball is, not who has it. Families are labels for later stages, not decisions.
TAG_FAMILIES = {
    "set pieces": "set_piece",
    "goals": "goal",
    "build up": "attack_phase",
    "progression": "attack_phase",
    "creating chances": "attack_phase",
    "attacking transition": "attack_phase",
    "long ball": "attack_phase",
    "defending in attacking third": "defend_phase",
    "defending in middle third": "defend_phase",
    "defending in defensive third": "defend_phase",
    "defensive transition": "defend_phase",
    "ball in final third": "zone",
    "ball in the box": "zone",
    "players in the box": "zone",
}
TAG_NO_TEAM = {"", "n/a", "na", "none"}

# Every array in timeline.npz, with its shape. N = 199319 frames, S = track slots, C = tag codes.
TIMELINE_ARRAYS = {
    "frame": "(N,) int32      video frame = submission row",
    "atd_frame_id": "(N,) int32      ATD frame_id of that video frame (frame + 1)",
    "period": "(N,) int8       0 pre_match, 1 half_1, 2 half_time, 3 half_2, 4 post_match",
    "half": "(N,) int8       1 or 2 inside a graded half, else 0",
    "graded": "(N,) bool       frame is scored",
    "label_init": "(N,) <U1       'X' outside the halves, '?' (undecided) inside",
    "ball_xy": "(N,2) float32   normalised 0-1, NaN when the ball is not tracked",
    "ball_visible": "(N,) bool",
    "ball_speed_raw": "(N,) float32    ATD ball speed column as delivered (mostly NaN, unreliable)",
    "players_xy": "(N,S,2) float32 normalised 0-1, NaN when the slot is empty",
    "players_speed": "(N,S) float32   ATD player speed column (m/s in practice, see meta.json)",
    "players_present": "(N,S) bool",
    "slot_team": "(S,) int8       0 home, 1 away",
    "slot_player_id": "(S,) int16      XML Player id of each slot, in channel order",
    "n_home": "(N,) int8       tracked home slots on the frame",
    "n_away": "(N,) int8       tracked away slots on the frame",
    "hom_matrix": "(N,3,3) float64 normalised pitch -> pixels, NaN when invalid",
    "hom_error": "(N,) float32    reprojection error, lower is better",
    "hom_valid": "(N,) bool",
    "tag_codes": "(C,) str        Smart Tagging codes, column order of tag_active",
    "tag_active": "(N,C) bool      frame lies inside an instance of that code",
}


# --------------------------------------------------------------------------------------------
# checks

class Checks(object):
    """Collects named pass/fail checks. 'error' failures abort the run before anything is saved."""

    def __init__(self):
        self.items = []

    def add(self, name, ok, detail="", severity="error"):
        self.items.append({"name": name, "ok": bool(ok), "severity": severity, "detail": detail})
        return bool(ok)

    def failed(self, severity="error"):
        return [c for c in self.items if not c["ok"] and c["severity"] == severity]

    def report(self):
        for c in self.items:
            mark = "ok  " if c["ok"] else ("FAIL" if c["severity"] == "error" else "warn")
            print("  [{}] {}{}".format(mark, c["name"], " -- " + c["detail"] if c["detail"] else ""))


# --------------------------------------------------------------------------------------------
# ATD metadata

def parse_atd_meta(path):
    """Teams, slot -> team map and the exact channel order of the raw file, from the EPTS XML.

    The slot order is read from DataFormatSpecification rather than assumed from Player ids,
    because that block is what defines the column order of every raw line.
    """
    root = ET.parse(path).getroot()

    teams = []
    for team in root.iter("Team"):
        params = {(p.findtext("Name") or "").strip(): (p.findtext("Value") or "").strip()
                  for p in team.iter("ProviderParameter")}
        teams.append({"id": team.get("id"), "name": (team.findtext("Name") or "").strip(),
                      "params": params})
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(path, len(teams)))
    # FIFA/EPTS convention, also followed by kloppy: the first team listed is the home side.
    home, away = teams
    team_side = {home["id"]: C.HOME, away["id"]: C.AWAY}

    player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    unknown_team = sorted(pid for pid, tid in player_team.items() if tid not in team_side)
    if unknown_team:
        raise ValueError("{}: players with an unknown teamId: {}".format(path, unknown_team))

    # PlayerChannel id -> (player id, channel) ; e.g. player12_x -> ("12", "x")
    channel_def = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                   for pc in root.iter("PlayerChannel")}

    spec = next(root.iter("DataFormatSpecification"), None)
    if spec is None:
        raise ValueError("{}: no DataFormatSpecification".format(path))
    refs = [channel_def.get(ref.get("playerChannelId")) for ref in spec.iter("PlayerChannelRef")]
    if any(r is None for r in refs) or len(refs) % 3:
        raise ValueError("{}: unresolvable PlayerChannelRef entries".format(path))

    slot_player_ids = []
    for i in range(0, len(refs), 3):
        triple = refs[i:i + 3]
        pids = {pid for pid, _ in triple}
        channels = [ch for _, ch in triple]
        if len(pids) != 1 or channels != ["x", "y", "s"]:
            raise ValueError("{}: slot {} channels are {} for players {}; expected x,y,s of one "
                             "player".format(path, i // 3 + 1, channels, sorted(pids)))
        slot_player_ids.append(triple[0][0])

    ball_channels = [ref.get("channelId") for ref in spec.iter("BallChannelRef")]
    if ball_channels != ["x", "y", "s"]:
        raise ValueError("{}: ball channels {} != x,y,s".format(path, ball_channels))

    slot_team = np.array([team_side[player_team[pid]] for pid in slot_player_ids], dtype=np.int8)

    speed_unit = None
    for sensor in root.iter("Sensor"):
        if sensor.get("id") == "speed":
            speed_unit = (sensor.findtext("Name") or "").strip()

    home_dir = home["params"].get("attack_direction_first_half", "")
    if home_dir not in ("left_to_right", "right_to_left"):
        raise ValueError("{}: home attack_direction_first_half is {!r}".format(path, home_dir))
    home_attacks_positive_x_h1 = home_dir == "left_to_right"

    field = next(root.iter("FieldSize"), None)
    return {
        "home": {"id": home["id"], "name": home["name"], "params": home["params"]},
        "away": {"id": away["id"], "name": away["name"], "params": away["params"]},
        "slot_player_ids": slot_player_ids,
        "slot_team": slot_team,
        "n_slots": len(slot_player_ids),
        "spec_start_frame": int(spec.get("startFrame")),
        "spec_end_frame": int(spec.get("endFrame")),
        "frame_rate": float(root.findtext(".//FrameRate") or "nan"),
        "field_size_m": (float(field.findtext("Width")), float(field.findtext("Height")))
        if field is not None else None,
        "declared_speed_unit": speed_unit,
        # Direction of attack in normalised x for each half. Second half swaps.
        "home_attacks_positive_x": {1: home_attacks_positive_x_h1, 2: not home_attacks_positive_x_h1},
    }


# --------------------------------------------------------------------------------------------
# ATD raw

def _floats(text):
    """Comma-separated numbers (NaN allowed) -> float32 array, without a Python object per value."""
    try:
        return np.fromstring(text, dtype=np.float32, sep=",")
    except Exception:  # pragma: no cover - fallback if a future numpy drops text-mode fromstring
        return np.array(text.split(","), dtype=np.float32)


def parse_atd_raw(path, n_slots):
    """(frame_ids, values) with values (n_lines, (n_slots + 1) * 3): slot x,y,s ... then ball x,y,s.

    Every line is checked for shape ("id:players:ball", n_slots groups, 3 values per group); a
    malformed line raises with its line number instead of shifting every later column.
    """
    width = (n_slots + 1) * 3
    ids, blocks = [], []
    batch_ids, batch_text = [], []

    def flush():
        if not batch_ids:
            return
        values = _floats(",".join(batch_text))
        if values.size != len(batch_ids) * width:
            raise ValueError("{}: batch ending at frame_id {} has {} values, expected {}".format(
                path, batch_ids[-1], values.size, len(batch_ids) * width))
        blocks.append(values.reshape(len(batch_ids), width))
        ids.extend(batch_ids)
        del batch_ids[:], batch_text[:]

    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3:
                raise ValueError("{}:{}: expected 'id:players:ball'".format(path, line_no))
            frame_id, players, ball = parts
            if players.count(";") != n_slots - 1 or players.count(",") != 2 * n_slots:
                raise ValueError("{}:{}: expected {} player groups of 3 values".format(
                    path, line_no, n_slots))
            if ball.count(",") != 2:
                raise ValueError("{}:{}: ball group is not x,y,s".format(path, line_no))
            batch_ids.append(int(frame_id))
            batch_text.append(players.replace(";", ","))
            batch_text.append(ball)
            if len(batch_ids) >= ATD_CHUNK_LINES:
                flush()
    flush()
    if not blocks:
        raise ValueError("{}: no tracking records".format(path))
    return np.asarray(ids, dtype=np.int64), np.concatenate(blocks, axis=0)


def align_atd(frame_ids, values, n_slots, n_frames, checks):
    """Place ATD records on the video timeline: record with frame_id k goes to video frame k - 1."""
    checks.add("atd record count equals video frames",
               len(frame_ids) == n_frames, "{} records, {} frames".format(len(frame_ids), n_frames))
    contiguous = len(frame_ids) > 0 and np.array_equal(frame_ids, np.arange(1, len(frame_ids) + 1))
    checks.add("atd frame_id is exactly 1..N in order", contiguous,
               "range {}..{}".format(frame_ids.min(), frame_ids.max()))

    video_frames = frame_ids - 1
    inside = (video_frames >= 0) & (video_frames < n_frames)
    aligned = np.full((n_frames, values.shape[1]), np.nan, dtype=np.float32)
    aligned[video_frames[inside]] = values[inside]
    covered = np.zeros(n_frames, dtype=bool)
    covered[video_frames[inside]] = True
    checks.add("every video frame has one atd record", covered.all() and inside.all()
               and len(np.unique(video_frames)) == len(video_frames),
               "{} frames without a record".format(int((~covered).sum())))

    players = aligned[:, :n_slots * 3].reshape(n_frames, n_slots, 3)
    ball = aligned[:, n_slots * 3:]
    atd_frame_id = np.arange(1, n_frames + 1, dtype=np.int32)
    return atd_frame_id, players, ball


# --------------------------------------------------------------------------------------------
# homography

def load_homography(path, n_frames, checks):
    table = pd.read_csv(path)
    missing = [c for c in ["frame", "reprojection_error", "valid"] + HOMOGRAPHY_MATRIX_COLUMNS
               if c not in table.columns]
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))

    frames = table["frame"].to_numpy()
    ok = len(frames) == n_frames and np.array_equal(frames, np.arange(n_frames))
    checks.add("homography rows are frames 0..N-1 in order", ok,
               "{} rows".format(len(frames)))
    if not ok:
        raise ValueError("{}: homography is not one row per video frame".format(path))

    matrix = table[HOMOGRAPHY_MATRIX_COLUMNS].to_numpy(dtype=np.float64).reshape(n_frames, 3, 3)
    error = table["reprojection_error"].to_numpy(dtype=np.float32)
    valid = table["valid"].to_numpy() == 1
    nan_rows = np.isnan(matrix).any(axis=(1, 2))
    checks.add("homography matrix is NaN exactly where valid == 0",
               np.array_equal(nan_rows, ~valid),
               "{} valid, {} invalid".format(int(valid.sum()), int((~valid).sum())))
    matrix[~valid] = np.nan
    return matrix, error, valid


# --------------------------------------------------------------------------------------------
# Smart Tagging

def parse_smart_tagging(path, n_frames, side_by_team_name):
    """One row per instance, in video frames (end exclusive), with the tags spread into columns."""
    rows = []
    for inst in ET.parse(path).getroot().iter("instance"):
        tags = {}
        for label in inst.findall("label"):
            group = (label.findtext("group") or "").strip()
            tags.setdefault(group, []).append((label.findtext("text") or "").strip())
        code = (inst.findtext("code") or "").strip()
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        team_raw = "|".join(tags.get("Team", []))
        team_key = team_raw.strip().lower()
        if team_key in TAG_NO_TEAM:
            side = ""
        elif team_key in side_by_team_name:
            side = side_by_team_name[team_key]
        else:
            side = "?"  # a team string the ATD metadata does not know; flagged by a check
        rows.append({
            "id": int(inst.findtext("ID")),
            "code": code,
            "family": TAG_FAMILIES.get(code.lower(), "other"),
            "start_s": start_s,
            "end_s": end_s,
            # DATA.md: video_frame = round(seconds * 25). End is exclusive.
            "start_frame": int(round(start_s * C.FPS)),
            "end_frame": int(round(end_s * C.FPS)),
            "team_raw": team_raw,
            "side": side,
            "type": "|".join(tags.get("Type", [])),
            "half_tag": "|".join(tags.get("Half", [])),
            "pitch_side": "|".join(tags.get("Side", [])),
            "ball_entry": "|".join(tags.get("Direction of ball entry", [])),
            "max_players_in_box": "|".join(tags.get("Max Players in the box", [])),
            "other_tags": ";".join("{}={}".format(g, "|".join(v)) for g, v in sorted(tags.items())
                                   if g not in ("Team", "Type", "Half", "Side",
                                                "Direction of ball entry",
                                                "Max Players in the box")),
        })
    events = pd.DataFrame(rows)
    events["start_frame"] = events["start_frame"].clip(0, n_frames)
    events["end_frame"] = events["end_frame"].clip(0, n_frames)
    return events.sort_values(["start_frame", "end_frame", "id"]).reset_index(drop=True)


def tag_activity(events, n_frames):
    codes = sorted(events["code"].unique().tolist())
    column = {code: i for i, code in enumerate(codes)}
    active = np.zeros((n_frames, len(codes)), dtype=bool)
    for code, start, end in events[["code", "start_frame", "end_frame"]].itertuples(index=False):
        active[start:end, column[code]] = True
    return np.array(codes), active


# --------------------------------------------------------------------------------------------
# periods (rule 2)

def build_periods(n_frames):
    """period / half / graded arrays from the published half boundaries in games.py."""
    blocks = games.half_video_blocks(C.GAME)
    (h1, a1, b1), (h2, a2, b2) = blocks
    if (h1, h2) != (1, 2) or not (0 <= a1 <= b1 < a2 <= b2 < n_frames):
        raise ValueError("unexpected half blocks from games.py: {}".format(blocks))

    period = np.empty(n_frames, dtype=np.int8)
    period[:a1] = C.PRE
    period[a1:b1 + 1] = C.H1
    period[b1 + 1:a2] = C.HT
    period[a2:b2 + 1] = C.H2
    period[b2 + 1:] = C.POST
    half = np.where(period == C.H1, 1, np.where(period == C.H2, 2, 0)).astype(np.int8)
    graded = half > 0
    label_init = np.where(graded, C.LABEL_UNDECIDED, C.LABEL_X).astype("<U1")
    spans = [(C.PRE, 0, a1 - 1), (C.H1, a1, b1), (C.HT, b1 + 1, a2 - 1),
             (C.H2, a2, b2), (C.POST, b2 + 1, n_frames - 1)]
    return period, half, graded, label_init, spans


# --------------------------------------------------------------------------------------------
# data-driven sanity checks (reported, never used to move a boundary)

def kickoff_windows(events):
    """'Kick Off (Start)' set-piece windows per Half tag."""
    is_ko = (events["family"] == "set_piece") & (events["type"].str.lower() == "kick off (start)")
    out = {}
    for row in events[is_ko].itertuples(index=False):
        half = 1 if row.half_tag.lower().startswith("1") else 2 if row.half_tag.lower().startswith("2") else 0
        out.setdefault(half, []).append((int(row.start_frame), int(row.end_frame)))
    return out


def team_mean_x(players_xy, players_present, slot_team, frames):
    """Mean normalised x of each side's tracked players over a set of frames."""
    xs = players_xy[frames, :, 0]
    present = players_present[frames]
    out = {}
    for side in (C.HOME, C.AWAY):
        sel = present[:, slot_team == side]
        vals = xs[:, slot_team == side][sel]
        out[side] = float(np.mean(vals)) if vals.size else float("nan")
    return out


def speed_unit_check(players_xy, players_speed, graded):
    """Median ratio of the delivered speed column to speed from consecutive positions (m/s)."""
    frames = np.flatnonzero(graded[1:] & graded[:-1]) + 1
    frames = frames[::5]  # a fifth of the graded frames is plenty and keeps memory small
    step = (players_xy[frames] - players_xy[frames - 1]) * C.PITCH_SCALE
    derived = np.sqrt((step ** 2).sum(axis=2)) * C.FPS
    given = players_speed[frames]
    ok = np.isfinite(derived) & np.isfinite(given) & (derived > 0.5) & (derived < 12)
    if ok.sum() < 1000:
        return float("nan"), float("nan")
    ratio = float(np.median(given[ok] / derived[ok]))
    corr = float(np.corrcoef(given[ok], derived[ok])[0, 1])
    return ratio, corr


def video_frame_count(path):
    """Frame count of the video, or None if neither OpenCV nor ffprobe is available."""
    if not path or not os.path.isfile(path):
        return None
    try:
        import cv2  # noqa
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=nb_frames", "-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=60).stdout.strip()
            return int(out) if out.isdigit() else None
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------------------------
# main

def build(data_dir=None, work_dir=None, check_video=True):
    t0 = time.time()
    n = C.N_FRAMES
    checks = Checks()
    paths = C.resolve_assets(data_dir, "tactical")
    C.require_files(paths, ["atd_meta", "atd_raw", "homography", "smart_tagging"])
    print("data: {}".format(os.path.dirname(paths["atd_raw"])))

    # -- ATD ------------------------------------------------------------------------------------
    meta = parse_atd_meta(paths["atd_meta"])
    s = meta["n_slots"]
    print("atd meta: H = {}, A = {}, {} slots ({} home, {} away)".format(
        meta["home"]["name"], meta["away"]["name"], s,
        int((meta["slot_team"] == C.HOME).sum()), int((meta["slot_team"] == C.AWAY).sum())))
    checks.add("atd metadata frame span is 1..N",
               (meta["spec_start_frame"], meta["spec_end_frame"]) == (1, n),
               "{}..{}".format(meta["spec_start_frame"], meta["spec_end_frame"]))
    checks.add("atd frame rate matches games.py", meta["frame_rate"] == C.FPS,
               "{} fps".format(meta["frame_rate"]))

    print("reading raw tracking ...")
    frame_ids, values = parse_atd_raw(paths["atd_raw"], s)
    atd_frame_id, players, ball = align_atd(frame_ids, values, s, n, checks)
    del values

    players_xy = np.ascontiguousarray(players[:, :, :2])
    players_speed = np.ascontiguousarray(players[:, :, 2])
    players_present = ~np.isnan(players_xy).any(axis=2)
    del players
    ball_xy = np.ascontiguousarray(ball[:, :2])
    ball_speed_raw = np.ascontiguousarray(ball[:, 2])
    ball_visible = ~np.isnan(ball_xy).any(axis=1)
    checks.add("ball x and y are missing together",
               np.array_equal(np.isnan(ball_xy[:, 0]), np.isnan(ball_xy[:, 1])))
    checks.add("player x and y are missing together",
               np.array_equal(np.isnan(players_xy[:, :, 0]), np.isnan(players_xy[:, :, 1])))
    slot_team = meta["slot_team"]
    n_home = players_present[:, slot_team == C.HOME].sum(axis=1).astype(np.int8)
    n_away = players_present[:, slot_team == C.AWAY].sum(axis=1).astype(np.int8)

    # -- homography -----------------------------------------------------------------------------
    hom_matrix, hom_error, hom_valid = load_homography(paths["homography"], n, checks)

    # -- Smart Tagging --------------------------------------------------------------------------
    side_by_name = {meta["home"]["name"].lower(): "H", meta["away"]["name"].lower(): "A"}
    events = parse_smart_tagging(paths["smart_tagging"], n, side_by_name)
    unknown_teams = sorted(events.loc[events["side"] == "?", "team_raw"].unique().tolist())
    checks.add("every Smart Tagging Team value is a known club or N/A", not unknown_teams,
               "unknown: {}".format(unknown_teams) if unknown_teams else
               "{} = H, {} = A".format(meta["home"]["name"], meta["away"]["name"]))
    checks.add("every Smart Tagging code has a known family",
               not (events["family"] == "other").any(),
               ", ".join(sorted(events.loc[events["family"] == "other", "code"].unique())),
               severity="warning")
    checks.add("Smart Tagging intervals are well formed",
               bool((events["end_frame"] > events["start_frame"]).all()))
    tag_codes, tag_active = tag_activity(events, n)

    # -- periods (rule 2) -----------------------------------------------------------------------
    period, half, graded, label_init, spans = build_periods(n)
    events["period_of_start"] = [C.PERIOD_NAMES[p] for p in period[events["start_frame"].clip(0, n - 1)]]
    graded_cum = np.concatenate(([0], np.cumsum(graded)))
    events["graded_frames"] = graded_cum[events["end_frame"]] - graded_cum[events["start_frame"]]
    events["half_from_frame"] = half[events["start_frame"].clip(0, n - 1)]
    tag_half = events["half_tag"].str[:1].map({"1": 1, "2": 2}).fillna(0).astype(int)
    inconsistent = (events["half_from_frame"] > 0) & (tag_half > 0) & (tag_half != events["half_from_frame"])
    events["half_tag_consistent"] = ~inconsistent
    checks.add("Half tag agrees with the half the instance starts in", not inconsistent.any(),
               "{} disagree".format(int(inconsistent.sum())), severity="warning")
    outside = events["graded_frames"] == 0
    checks.add("Smart Tagging instances entirely outside the graded halves", not outside.any(),
               "{} instances ({}) -- keep them out of later stages".format(
                   int(outside.sum()),
                   ", ".join("{}x {}".format(v, k) for k, v in
                             events.loc[outside, "code"].value_counts().items())),
               severity="warning")

    # kickoffs: the first graded frame of half 1 is the kickoff (DATA.md); the half-2 kickoff
    # tag must sit inside half 2. Both are checks on the time base, not inputs to it.
    blocks = {h: (a, b) for h, a, b in games.half_video_blocks(C.GAME)}
    ko = kickoff_windows(events)
    ko1 = [w for w in ko.get(1, []) if w[0] <= blocks[1][0] < w[1]]
    checks.add("a 1st-half 'Kick Off (Start)' tag contains the first graded frame", bool(ko1),
               "tags {} vs frame {}".format(ko.get(1), blocks[1][0]))
    ko2 = [w for w in ko.get(2, []) if blocks[2][0] <= w[0] <= blocks[2][1]]
    checks.add("a 2nd-half 'Kick Off (Start)' tag lies inside half 2", bool(ko2),
               "tags {} vs half {}".format(ko.get(2), blocks[2]))

    # attack direction from the XML, checked at each kickoff: every side starts in its own half.
    direction = {}
    for h, windows in ((1, ko1), (2, ko2)):
        if not windows:
            continue
        a, b = windows[0]
        means = team_mean_x(players_xy, players_present, slot_team, np.arange(a, b))
        home_pos = meta["home_attacks_positive_x"][h]
        ok = (means[C.HOME] < 0.5 < means[C.AWAY]) if home_pos else (means[C.AWAY] < 0.5 < means[C.HOME])
        direction[h] = {"home_attacks_positive_x": home_pos, "home_mean_x": means[C.HOME],
                        "away_mean_x": means[C.AWAY], "kickoff_window": [a, b]}
        checks.add("half {}: teams line up on the sides the XML attack direction implies".format(h),
                   ok, "home mean x {:.3f}, away mean x {:.3f}, home attacks {}".format(
                       means[C.HOME], means[C.AWAY], "+x" if home_pos else "-x"))

    # speed units: the XML says km/h; measure it.
    ratio, corr = speed_unit_check(players_xy, players_speed, graded)
    inferred = "m/s" if 0.8 <= ratio <= 1.25 else "km/h" if 3.0 <= ratio <= 4.3 else "unknown"
    checks.add("player speed unit inferred from positions", inferred != "unknown",
               "column / derived m/s = {:.3f} (corr {:.3f}) -> {}; XML says '{}'".format(
                   ratio, corr, inferred, meta["declared_speed_unit"]), severity="warning")
    ball_speed_missing = float(np.mean(np.isnan(ball_speed_raw[ball_visible]))) if ball_visible.any() else 1.0

    if check_video:
        nv = video_frame_count(paths.get("video"))
        if nv is None:
            checks.add("tactical video frame count", True,
                       "skipped (video not found or no OpenCV/ffprobe): {}".format(paths.get("video")),
                       severity="warning")
        else:
            checks.add("tactical video frame count equals N", nv == n,
                       "{} frames in {}".format(nv, os.path.basename(paths["video"])))

    # -- summaries ------------------------------------------------------------------------------
    period_rows = []
    for code, first, last in spans:
        sl = slice(first, last + 1)
        period_rows.append({
            "period": C.PERIOD_NAMES[code], "code": code, "first_frame": first, "last_frame": last,
            "n_frames": last - first + 1, "graded": code in (C.H1, C.H2),
            "start_clock": "{:d}:{:05.2f}".format(int(first / C.FPS // 60), first / C.FPS % 60),
            "ball_visible_pct": round(100.0 * float(ball_visible[sl].mean()), 2),
            "mean_home_tracked": round(float(n_home[sl].mean()), 2),
            "mean_away_tracked": round(float(n_away[sl].mean()), 2),
            "hom_valid_pct": round(100.0 * float(hom_valid[sl].mean()), 2),
            "hom_error_median": round(float(np.nanmedian(hom_error[sl])), 3),
        })
    periods = pd.DataFrame(period_rows)

    notes = []
    if ko2:
        pre_ko = ko2[0][0] - blocks[2][0]
        notes.append("Half 2 is graded from frame {} but its kickoff tag starts at {}: at least {} "
                     "graded frames precede the restart and must be resolved as D, not X, by the "
                     "stoppage stage.".format(blocks[2][0], ko2[0][0], pre_ko))
    notes.append("Ball speed column is NaN on {:.1f}% of frames with a ball; derive ball speed "
                 "from positions.".format(100 * ball_speed_missing))

    print("\nchecks")
    checks.report()
    failed = checks.failed("error")
    if failed:
        raise SystemExit("\n{} critical check(s) failed; nothing written.".format(len(failed)))

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    np.savez(
        os.path.join(out, "timeline.npz"),
        frame=np.arange(n, dtype=np.int32), atd_frame_id=atd_frame_id,
        period=period, half=half, graded=graded, label_init=label_init,
        ball_xy=ball_xy, ball_visible=ball_visible, ball_speed_raw=ball_speed_raw,
        players_xy=players_xy, players_speed=players_speed, players_present=players_present,
        slot_team=slot_team, slot_player_id=np.array(meta["slot_player_ids"], dtype=np.int16),
        n_home=n_home, n_away=n_away,
        hom_matrix=hom_matrix, hom_error=hom_error, hom_valid=hom_valid,
        tag_codes=tag_codes, tag_active=tag_active,
    )
    events.to_csv(os.path.join(out, "smart_tagging_events.csv"), index=False)
    periods.to_csv(os.path.join(out, "periods.csv"), index=False)

    ball_m = C.to_metres(ball_xy)
    pd.DataFrame({
        "frame": np.arange(n), "period": np.array(C.PERIOD_NAMES)[period], "half": half,
        "graded": graded.astype(np.int8), "label_init": label_init,
        "ball_visible": ball_visible.astype(np.int8),
        "ball_x_m": np.round(ball_m[:, 0], 2), "ball_y_m": np.round(ball_m[:, 1], 2),
        "n_home": n_home, "n_away": n_away,
        "hom_valid": hom_valid.astype(np.int8), "hom_error": np.round(hom_error, 3),
        "n_tags_active": tag_active.sum(axis=1),
    }).to_csv(os.path.join(out, "frames.csv"), index=False)

    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE, "game": C.GAME, "n_frames": n, "fps": C.FPS,
        "pitch_m": [C.PITCH_LENGTH, C.PITCH_WIDTH],
        "files": {k: v for k, v in paths.items()},
        "teams": {"H": meta["home"], "A": meta["away"]},
        "n_slots": s,
        "slot_team": ["H" if t == C.HOME else "A" for t in slot_team],
        "slot_player_id": meta["slot_player_ids"],
        "home_attacks_positive_x": {str(k): v for k, v in meta["home_attacks_positive_x"].items()},
        "kickoff_direction_check": {str(k): v for k, v in direction.items()},
        "declared_speed_unit": meta["declared_speed_unit"],
        "player_speed_unit_inferred": inferred, "player_speed_ratio_to_derived": ratio,
        "ball_speed_missing_when_ball_visible": ball_speed_missing,
        "half_video_blocks": [list(b) for b in games.half_video_blocks(C.GAME)],
        "periods": period_rows,
        "label_counts": {"X": int((label_init == C.LABEL_X).sum()),
                         "undecided": int((label_init == C.LABEL_UNDECIDED).sum())},
        "smart_tagging": {"n_instances": int(len(events)),
                          "codes": events["code"].value_counts().to_dict()},
        "arrays": TIMELINE_ARRAYS,
        "checks": checks.items,
        "notes": notes,
    })

    print("\nperiods")
    print(periods[["period", "first_frame", "last_frame", "n_frames", "graded", "start_clock",
                   "ball_visible_pct", "mean_home_tracked", "mean_away_tracked"]].to_string(index=False))
    print("\nlabels: {} X (ungraded), {} undecided (graded)".format(
        int((label_init == C.LABEL_X).sum()), int((label_init == C.LABEL_UNDECIDED).sum())))
    for note in notes:
        print("note: " + note)
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


class Timeline(object):
    """Read-only view of work/s00 for later stages: t.ball_xy, t.events, t.meta, t.metres(...)."""

    def __init__(self, work_dir=None):
        folder = os.path.join(work_dir or C.WORK_DIR, STAGE)
        path = os.path.join(folder, "timeline.npz")
        if not os.path.isfile(path):
            raise IOError("{} not found -- run: python -m pipeline4.s00_timeline".format(path))
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                setattr(self, name, data[name])
        self.events = pd.read_csv(os.path.join(folder, "smart_tagging_events.csv"),
                                  keep_default_na=False)
        self.periods = pd.read_csv(os.path.join(folder, "periods.csv"))
        self.meta = C.read_json(os.path.join(folder, "meta.json"))
        self.n_frames = int(len(self.frame))

    @staticmethod
    def metres(xy):
        return C.to_metres(xy)

    def tag(self, code):
        """(N,) bool: frames inside any instance of a Smart Tagging code."""
        idx = np.flatnonzero(self.tag_codes == code)
        if not len(idx):
            raise KeyError("unknown code {!r}; have {}".format(code, self.tag_codes.tolist()))
        return self.tag_active[:, idx[0]]


def load_timeline(work_dir=None):
    return Timeline(work_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 00: build the aligned frame timeline.")
    parser.add_argument("--data-dir", default=None,
                        help="folder holding tactical/ and broadcast/ (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    parser.add_argument("--skip-video", action="store_true", help="do not count video frames")
    args = parser.parse_args(argv)
    build(args.data_dir, args.work_dir, check_video=not args.skip_video)


if __name__ == "__main__":
    main()
