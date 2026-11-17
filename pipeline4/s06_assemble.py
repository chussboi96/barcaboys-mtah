"""Stage 06 -- combine everything into one label per frame and enforce the state rules (20, 21).

Precedence (rule 20), applied in this order so a later source can never overwrite an earlier one:

    1. X     frames outside the two graded halves (stage 00)
    2. D     frames inside a confirmed stoppage, [onset, restart) (stage 03)
    3. H/A   live frames, from the possession model (stage 05)
    4. fill  any live frame still without a side: the restarting side of its live stretch
             (stage 04), else pass 1 (stage 02), else the previous label in the half. Never X:
             an abstention is not a substitute for an uncertain decision (rule 21)

Rules enforced and checked (rule 21):

    - no X inside a graded half, no H/A/D outside one
    - every stoppage is D from its onset up to, not including, its restart: the restarting side
      is never given possession during the stoppage before it
    - every D -> H/A change sits on a stage-04 restart, every H/A -> D change on a stage-03 onset
    - a turnover in live play (H -> A) and a stoppage followed by a new possession (H -> D -> A,
      or H -> D -> H) stay different things: nothing here merges across a D
    - the halves are encoded separately, so nothing can run across half-time

Every repair is counted and every change of label is listed, so the file can be audited.

    python -m pipeline4.s06_assemble [--results-name my_team]

Outputs, in work/s06/:

    labels.csv        the submission file (frame,label; 199,319 rows), checked with the official
                      validator from evaluation/possession_metrics.py
    segments.csv      every maximal run of one label inside a half, with its neighbours
    transitions.csv   every change of label inside a half, typed: turnover / stoppage / restart
    meta.json         repairs, statistics (possession count and duration as the scorer computes
                      them), and checks

--results-name NAME also copies labels.csv to evaluation/results/NAME.csv (gitignored), so
`python evaluation/run_evaluation.py` scores it next to the baselines.
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
from pipeline4.s00_timeline import Checks, load_timeline  # noqa: E402
from pipeline4.s02_touches import load_contacts  # noqa: E402
from pipeline4.s03_stoppages import load_deadball  # noqa: E402
from pipeline4.s04_restarts import load_restarts  # noqa: E402
from pipeline4.s05_possession import load_possession  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

STAGE = "s06"
F = C.FPS


def letters(side_codes):
    return np.where(side_codes == C.HOME, "H", np.where(side_codes == C.AWAY, "A", "?"))


def segment_table(labels, blocks):
    rows = []
    for h, first, last in blocks:
        seg = labels[first:last + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = [0] + change.tolist() + [len(seg)]
        for k in range(len(bounds) - 1):
            a, b = first + bounds[k], first + bounds[k + 1]
            rows.append({"half": h, "start": a, "end": b - 1, "frames": b - a,
                         "duration_s": round((b - a) / F, 2), "label": labels[a],
                         "prev": labels[a - 1] if a > first else "",
                         "next": labels[b] if b <= last else ""})
    return pd.DataFrame(rows)


def build(work_dir=None, results_name=None):
    t0 = time.time()
    checks = Checks()
    t = load_timeline(work_dir)
    c = load_contacts(work_dir)
    d = load_deadball(work_dir)
    r = load_restarts(work_dir)
    p = load_possession(work_dir)
    n = t.n_frames
    g = t.graded
    blocks = C.games.half_video_blocks(C.GAME)
    print("loaded s00-s05")

    # -- rule 20: precedence ------------------------------------------------------------------
    labels = np.full(n, "?", dtype="<U1")
    labels[~g] = "X"                                         # 1. outside the halves
    dead = d.dead & g
    labels[dead] = "D"                                       # 2. stoppages
    live = g & ~dead
    from_model = live & (p.poss_side >= 0)
    labels[from_model] = letters(p.poss_side[from_model])    # 3. possession model

    # 4. fill, never with X
    repairs = {"filled_from_restart": 0, "filled_from_pass1": 0, "filled_forward": 0}
    hole = live & (labels == "?")
    if hole.any():
        prior = r.segment_prior_side
        use = hole & (prior >= 0)
        labels[use] = letters(prior[use])
        repairs["filled_from_restart"] = int(use.sum())
        hole = live & (labels == "?")
        use = hole & (c.p1_side >= 0)
        labels[use] = letters(c.p1_side[use])
        repairs["filled_from_pass1"] = int(use.sum())
        hole = live & (labels == "?")
        for h, first, last in blocks:
            for f in np.flatnonzero(hole[first:last + 1]) + first:
                prev = labels[f - 1] if f > first else "?"
                labels[f] = prev if prev in ("H", "A") else "H"
                repairs["filled_forward"] += 1

    # -- rule 21: checks ------------------------------------------------------------------------
    checks.add("no X, ? inside the graded halves", not bool(np.isin(labels[g], ["X", "?"]).any()),
               "{} frames".format(int(np.isin(labels[g], ["X", "?"]).sum())))
    checks.add("only X outside the graded halves", bool(np.all(labels[~g] == "X")))

    st = d.stoppages
    inside_bad = 0
    for row in st.itertuples(index=False):
        seg = labels[int(row.onset):int(row.restart)]
        inside_bad += int((seg != "D").sum())
    checks.add("every stoppage is D from onset to restart (restarting side never given the ball early)",
               inside_bad == 0, "{} frames inside stoppages not D".format(inside_bad))

    onsets = set(int(x) for x in st["onset"])
    restarts = r.restarts
    restart_frames = set(int(x) for x in restarts.loc[restarts["type"] != "end_of_half", "restart"])
    restart_side = {int(row.restart): row.side for row in restarts.itertuples(index=False)}

    trans = []
    for h, first, last in blocks:
        seg = labels[first:last + 1]
        for k in np.flatnonzero(seg[1:] != seg[:-1]) + 1:
            f = first + int(k)
            a, b = labels[f - 1], labels[f]
            kind = "turnover" if {a, b} == {"H", "A"} else "stoppage" if b == "D" else "restart"
            trans.append({"frame": f, "half": h, "from": a, "to": b, "kind": kind,
                          "on_stage03_onset": f in onsets if kind == "stoppage" else "",
                          "on_stage04_restart": f in restart_frames if kind == "restart" else "",
                          "restarting_side": restart_side.get(f, "") if kind == "restart" else "",
                          "takes_restart_side": (restart_side.get(f, "") == b) if kind == "restart" else ""})
    trans = pd.DataFrame(trans, columns=["frame", "half", "from", "to", "kind", "on_stage03_onset",
                                         "on_stage04_restart", "restarting_side", "takes_restart_side"])
    stp = trans[trans["kind"] == "stoppage"]
    rst = trans[trans["kind"] == "restart"]
    checks.add("every H/A -> D change is a stage-03 onset", bool(stp["on_stage03_onset"].astype(bool).all()),
               "{} of {}".format(int(stp["on_stage03_onset"].astype(bool).sum()), len(stp)))
    checks.add("every D -> H/A change is a stage-04 restart", bool(rst["on_stage04_restart"].astype(bool).all()),
               "{} of {}".format(int(rst["on_stage04_restart"].astype(bool).sum()), len(rst)))
    took = int(rst["takes_restart_side"].astype(bool).sum())
    checks.add("restarts where the restarting side gets the first possession", True,
               "{} of {} (the rest were overturned by evidence after the restart, rule 19)".format(took, len(rst)),
               severity="warning")

    # half boundaries: half 2 is graded before its kickoff -> D; half 1 is graded from its kickoff
    for h, first, last in blocks:
        checks.add("half {} first graded frame".format(h), True, "label {}".format(labels[first]), severity="warning")

    # format, with the official validator
    table = pd.DataFrame({"frame": np.arange(n), "label": labels})
    reason = pm.validate_submission(table, n)
    checks.add("official validate_submission() accepts the file", reason is None, reason or "OK")

    # possession statistics exactly as the scorer computes them
    codes = pm.label_codes(labels)
    graded_codes = np.where(g, 0, pm.LABELS.index("X")).astype(np.int8)
    valid = pm.valid_runs(graded_codes)
    pstats = pm.possession_stats(codes, valid)
    spells = pm.possession_spells(codes, valid)
    seg = segment_table(labels, blocks)
    short_live = seg[(seg["label"].isin(["H", "A"])) & (seg["frames"] < F)]
    checks.add("possessions shorter than 1 s (handled by stage 07)", True,
               "{} of {}".format(len(short_live), len(spells)), severity="warning")

    print("\nchecks")
    checks.report()
    if checks.failed("error"):
        raise SystemExit("\ncritical check failed; nothing written.")

    # -- write ----------------------------------------------------------------------------------
    out = C.stage_dir(STAGE, work_dir)
    path = os.path.join(out, "labels.csv")
    table.to_csv(path, index=False)
    seg.to_csv(os.path.join(out, "segments.csv"), index=False)
    trans.to_csv(os.path.join(out, "transitions.csv"), index=False)
    counts = {str(k): int(v) for k, v in zip(*np.unique(labels[g], return_counts=True))}
    stats = {
        "label_counts_graded": counts,
        "possessions": pstats["n_spells"],
        "possession_mean_s": round(pstats["mean_duration_s"], 2),
        "home_share_of_live": round(pstats["home_share"], 3),
        "stoppages": int((seg["label"] == "D").sum()),
        "turnovers": int((trans["kind"] == "turnover").sum()),
        "restarts": int(len(rst)),
        "restarts_taking_restart_side": took,
        "repairs": repairs,
    }
    C.write_json(os.path.join(out, "meta.json"), {"stage": STAGE, "stats": stats, "checks": checks.items})
    if results_name:
        dest = os.path.join(C.ROOT, "evaluation", "results", "{}.csv".format(results_name))
        shutil.copyfile(path, dest)
        print("\ncopied to {}".format(dest))
    print("\nstats")
    for k, v in stats.items():
        print("  {:<30} {}".format(k, v))
    print("\nwrote {} in {:.1f}s".format(out, time.time() - t0))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 06: assemble labels and enforce state rules.")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    parser.add_argument("--results-name", default=None,
                        help="also copy labels.csv to evaluation/results/<name>.csv for local scoring")
    args = parser.parse_args(argv)
    build(args.work_dir, args.results_name)


if __name__ == "__main__":
    main()
