"""Stage 07 -- temporal consistency of possession spells (rule 22).

Stage 06 gives a label per frame that already respects the state rules. What is left are short
possession spells the evidence does not really back: a restart credited to one side and taken by
the other in the very next frames, a one-frame flicker inside another side's possession, a
blip of "live play" inside what is really one long stoppage. Each of them is an extra possession
on the ranking metric (spellF1 counts possessions, not frames), so each costs precision.

The rule is about support, not length. A possession spell shorter than MAX_SHORT_S is kept if
its side has any of:

    a confident touch      a touch credited to that side with |side_score| >= CONFIDENT_SCORE
                           and not contested (stage 02)
    accepted backdate      a switch explicitly corrected to an earlier impulse in stage 05
    close control          a control spell of that side overlapping it (stage 02)
    evidence               possession-model log-odds for that side >= SUPPORT_EVIDENCE over it
                           (stage 05) -- the cost of one change of possession

Genuine short possessions (an interception and an immediate loss, which can last about a second)
have a touch or control behind them and are kept. An unsupported
short spell is absorbed by its context, never across a stoppage:

    X  Y  X       (flicker)              Y -> X
    D  Y  Z       (restart not taken)    Y -> Z   the restart went straight to Z
    Z  Y  D       (touch before a stop)  Y -> Z
    D  Y  D       (blip in a stoppage)   Y -> D   no possession is invented between two stops
    D  Y  end     (blip at half end)     Y -> D

A D frame is never turned into H or A, and nothing is merged across a D: two possessions of the
same side separated by a stoppage stay two possessions. The pass repeats until nothing changes.

    python -m pipeline4.s07_temporal [--results-name my_team]

Outputs, in work/s07/:

    labels.csv      the final labels (frame,label), checked with the official validator
    changes.csv     every spell that was changed, its support and the reason
    segments.csv    the runs of the final labels
    meta.json       parameters, statistics and checks
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline4 import common as C  # noqa: E402
from pipeline4.s00_timeline import Checks  # noqa: E402
from pipeline4.s02_touches import load_contacts  # noqa: E402
from pipeline4.s05_possession import SWITCH_COST, load_possession  # noqa: E402
from pipeline4.s06_assemble import segment_table  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

STAGE = "s07"
F = C.FPS

# --------------------------------------------------------------------------------------------
# parameters

MAX_SHORT_S = 2.0               # only spells shorter than this are questioned at all
CONFIDENT_SCORE = 0.75          # stage-02 attribution score that counts as a confident touch
SUPPORT_EVIDENCE = SWITCH_COST  # evidence equal to one change of possession in stage 05


def spell_support(label, a, b, touches, control, evidence, backdates):
    """(supported, reason) for a possession spell of `label` over frames [a, b]."""
    accepted = backdates[(backdates["to"] == label) &
                         (backdates["candidate_type"].isin(["reliable_touch", "ambiguous_impulse", "weak_touch_control"])) &
                         (backdates["corrected_switch_frame"] == a) &
                         (backdates["corrected_switch_frame"] <= b) &
                         (backdates["original_switch_frame"] >= a) &
                         (backdates["original_switch_frame"] <= b)]
    if len(accepted):
        return True, "accepted_backdate"
    tt = touches[(touches["frame"] >= a) & (touches["frame"] <= b) & (touches["side"] == label)]
    conf = tt[(tt["side_score"].abs() >= CONFIDENT_SCORE) & (~tt["contested"].astype(bool))]
    if len(conf):
        return True, "confident_touch"
    cc = control[(control["end"] >= a) & (control["start"] <= b) & (control["side"] == label)]
    if len(cc):
        return True, "close_control"
    signed = float(evidence[a:b + 1].sum()) * (1.0 if label == "H" else -1.0)
    if signed >= SUPPORT_EVIDENCE:
        return True, "evidence"
    return False, "unsupported (evidence {:.2f}, {} own touches)".format(signed, len(tt))


def smooth(labels, blocks, touches, control, evidence, backdates):
    """Absorb unsupported short spells one at a time until none is left.

    One change can create a new neighbour pattern (two spells of the same side now adjacent to
    a third), so the run table is rebuilt after every change. Each change removes a spell, so
    the loop ends after at most as many changes as there were spells.
    """
    labels = labels.copy()
    changes = []
    while True:
        changed = False
        for h, first, last in blocks:
            seg = segment_table(labels[first:last + 1].copy(), [(h, 0, last - first)])
            for row in seg.itertuples(index=False):
                if row.label not in ("H", "A") or row.frames >= MAX_SHORT_S * F:
                    continue
                a, b = first + int(row.start), first + int(row.end)
                ok, why = spell_support(row.label, a, b, touches, control, evidence, backdates)
                if ok:
                    continue
                prev, nxt = row.prev or "end", row.next or "end"
                if prev in ("H", "A") and prev == nxt:
                    new, pattern = prev, "flicker"
                elif prev == "D" and nxt in ("H", "A"):
                    new, pattern = nxt, "restart_not_taken"
                elif prev in ("H", "A") and nxt in ("D", "end"):
                    new, pattern = prev, "touch_before_stop"
                elif prev in ("D", "end") and nxt in ("D", "end"):
                    new, pattern = "D", "blip_inside_stoppage"
                else:
                    continue
                labels[a:b + 1] = new
                changes.append({"half": h, "start": a, "end": b, "frames": b - a + 1,
                                "duration_s": round((b - a + 1) / F, 2), "was": row.label,
                                "now": new, "prev": prev, "next": nxt, "pattern": pattern,
                                "support": why})
                changed = True
                break  # the run table of this half is stale now
            if changed:
                break
        if not changed:
            return labels, changes


def build(work_dir=None, results_name=None):
    t0 = time.time()
    checks = Checks()
    c = load_contacts(work_dir)
    p = load_possession(work_dir)
    root = work_dir or C.WORK_DIR
    s06 = pd.read_csv(os.path.join(root, "s06", "labels.csv"), keep_default_na=False)
    backdate_path = os.path.join(root, "s05", "backdating.csv")
    backdates = pd.read_csv(backdate_path) if os.path.isfile(backdate_path) else pd.DataFrame(
        columns=["corrected_switch_frame", "original_switch_frame", "to", "candidate_type"])
    before = s06["label"].to_numpy().astype("<U1")
    n = len(before)
    blocks = C.games.half_video_blocks(C.GAME)
    g = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        g[a:b + 1] = True
    print("loaded s02-s06")

    labels, changes = smooth(before, blocks, c.touches, c.control, p.evidence, backdates)

    # -- checks ---------------------------------------------------------------------------------
    checks.add("no D frame of stage 06 became H or A", bool(np.all(labels[before == "D"] == "D")))
    checks.add("only X outside the halves, never X inside",
               bool(np.all(labels[~g] == "X")) and not bool((labels[g] == "X").any()))
    checks.add("only short unsupported spells changed",
               all(ch["duration_s"] < MAX_SHORT_S and ch["support"].startswith("unsupported") for ch in changes))
    table = pd.DataFrame({"frame": np.arange(n), "label": labels})
    reason = pm.validate_submission(table, n)
    checks.add("official validate_submission() accepts the file", reason is None, reason or "OK")

    graded_codes = np.where(g, 0, pm.LABELS.index("X")).astype(np.int8)
    valid = pm.valid_runs(graded_codes)
    st_before = pm.possession_stats(pm.label_codes(before), valid)
    st_after = pm.possession_stats(pm.label_codes(labels), valid)
    seg = segment_table(labels, blocks)
    remaining_short = seg[seg["label"].isin(["H", "A"]) & (seg["frames"] < F)]
    checks.add("possessions under 1 s kept because they are supported", True,
               "{}".format(len(remaining_short)), severity="warning")
    print("\nchecks")
    checks.report()
    if checks.failed("error"):
        raise SystemExit("\ncritical check failed; nothing written.")

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    path = os.path.join(out, "labels.csv")
    table.to_csv(path, index=False)
    pd.DataFrame(changes, columns=["half", "start", "end", "frames", "duration_s", "was", "now", "prev",
                                   "next", "pattern", "support"]).to_csv(os.path.join(out, "changes.csv"), index=False)
    seg.to_csv(os.path.join(out, "segments.csv"), index=False)
    patterns = pd.Series([ch["pattern"] for ch in changes]).value_counts().to_dict() if changes else {}
    stats = {
        "spells_changed": len(changes),
        "frames_changed": int((labels != before).sum()),
        "patterns": patterns,
        "possessions_before": st_before["n_spells"], "possessions_after": st_after["n_spells"],
        "possession_mean_s_before": round(st_before["mean_duration_s"], 2),
        "possession_mean_s_after": round(st_after["mean_duration_s"], 2),
        "home_share_after": round(st_after["home_share"], 3),
        "label_counts_graded": {str(k): int(v) for k, v in zip(*np.unique(labels[g], return_counts=True))},
    }
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE,
        "parameters": {"MAX_SHORT_S": MAX_SHORT_S, "CONFIDENT_SCORE": CONFIDENT_SCORE,
                       "SUPPORT_EVIDENCE": SUPPORT_EVIDENCE},
        "stats": stats, "checks": checks.items})
    if results_name:
        dest = os.path.join(C.ROOT, "evaluation", "results", "{}.csv".format(results_name))
        shutil.copyfile(path, dest)
        print("\ncopied to {}".format(dest))
    print("\nstats")
    for k, v in stats.items():
        print("  {:<26} {}".format(k, v))
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 07: temporal consistency of possession spells.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    parser.add_argument("--results-name", default=None,
                        help="also copy labels.csv to evaluation/results/<name>.csv for local scoring")
    args = parser.parse_args(argv)
    build(args.work_dir, args.results_name)


if __name__ == "__main__":
    main()
