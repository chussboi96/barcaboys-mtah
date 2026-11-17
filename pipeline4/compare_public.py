"""Side-by-side check of a labels file against the PUBLIC 5-minute ground truth.

    python -m pipeline4.compare_public                      # work/s07/labels.csv
    python -m pipeline4.compare_public --labels work/s05/s05_labels.csv

Prints, for the graded public window only (video frames 11185-18684):
  - per-frame agreement and a confusion table (rows = ground truth, columns = yours)
  - both timelines as runs, with the video clock, so disagreements can be found in the video

This is a correctness check of the pipeline. RULES.md: the public window is not training data and
not a tuning set -- do not change thresholds until these 5 minutes match. For the official
numbers (macroF1, spellF1) use evaluation/run_evaluation.py.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline4 import common as C  # noqa: E402

GT = os.path.join(C.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def clock(frame):
    s = frame / float(C.FPS)
    return "{:d}:{:04.1f}".format(int(s // 60), s % 60)


def run_list(labels, a, b):
    out, s = [], a
    for f in range(a + 1, b + 1):
        if f == b or labels[f] != labels[s]:
            out.append((labels[s], s, f - 1))
            s = f
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=os.path.join(C.WORK_DIR, "s07", "labels.csv"))
    args = parser.parse_args(argv)

    gt = pd.read_csv(GT, dtype={"label": str})["label"].to_numpy()
    mine = pd.read_csv(args.labels, dtype={"label": str})["label"].to_numpy()
    graded = np.flatnonzero(gt != "X")
    a, b = int(graded[0]), int(graded[-1]) + 1
    g, m = gt[a:b], mine[a:b]

    print("public window {}-{} ({} - {}), {} frames\n".format(a, b - 1, clock(a), clock(b - 1), b - a))
    print("per-frame agreement: {:.1f}%".format(100 * float((g == m).mean())))
    tab = pd.crosstab(pd.Series(g, name="truth"), pd.Series(m, name="yours"))
    print(tab.to_string(), "\n")

    print("{:<34} {:<34}".format("ground truth", "yours"))
    rg, rm = run_list(gt, a, b), run_list(mine, a, b)
    for k in range(max(len(rg), len(rm))):
        left = "{} {:>6}-{:<6} {}-{}".format(rg[k][0], rg[k][1], rg[k][2], clock(rg[k][1]), clock(rg[k][2])) if k < len(rg) else ""
        right = "{} {:>6}-{:<6} {}-{}".format(rm[k][0], rm[k][1], rm[k][2], clock(rm[k][1]), clock(rm[k][2])) if k < len(rm) else ""
        print("{:<34} {:<34}".format(left, right))
    print("\nofficial scores: python evaluation/run_evaluation.py   (after --results-name in s06/s07)")


if __name__ == "__main__":
    main()
