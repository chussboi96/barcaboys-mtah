# barca boys — More Than A Hack 2027

Frame-level possession labels (`H` home, `A` away, `D` dead ball, `X` outside the halves) for
BAR-ATH, built from the provided tracking, Smart Tagging and the tactical video's audio.

There is no trained model and no labelled data. Two hidden Markov models, decoded with Viterbi,
make the two decisions: is the ball dead, and which team has it. Their evidence is hand-specified
from physical quantities (distances, speeds, touches, tags, the referee's whistle). The technical
document submitted with the form describes the method and every rule.

## Requirements

- Python 3.11 (tested on 3.11.9)
- `pip install -r requirements.txt` (tested with numpy 2.4.6 and pandas 3.1.0)
- ffmpeg on the PATH, for the whistle stage

## Data

The match data is not in this repository and must not be added to it. Unpack it as described in
[DATA.md](DATA.md). The folder passed below must contain `tactical/` with the tracking, homography,
Smart Tagging and video files.

## Run

Windows:

```
run_pipeline4.cmd "D:\path\to\data folder"
```

Any OS, the same steps:

```
python -m pipeline4.s00_timeline --data-dir "<data folder>"
python -m pipeline4.s01w_whistle --data-dir "<data folder>"
python -m pipeline4.s01_evidence
python -m pipeline4.s02_touches
python -m pipeline4.s03_stoppages
python -m pipeline4.s04_restarts
python -m pipeline4.s05_possession
python -m pipeline4.s06_assemble
python -m pipeline4.s07_temporal --results-name barca_boys
python evaluation/validate_submission.py evaluation/results/barca_boys.csv
```

The output is `evaluation/results/barca_boys.csv`. The run is deterministic. Intermediate outputs go
to `work_pipeline4/`, which is git-ignored because it holds frame-level data derived from the match.

To score it against the public 5-minute window: `python evaluation/run_evaluation.py`.

## Code

| Module | Decides |
|---|---|
| `pipeline4/s00_timeline.py` | One record per video frame; graded halves |
| `pipeline4/s01_evidence.py` | Clean ball track, pitch geometry, eligible and nearest players, tag hints, ball trust |
| `pipeline4/s01w_whistle.py` | Referee whistle strength per frame from the audio |
| `pipeline4/s02_touches.py` | Touches and their side, close control, slow carry, goalkeeper holds, completed passes |
| `pipeline4/s03_stoppages.py` | Dead-ball intervals: onset and restart |
| `pipeline4/s04_restarts.py` | Restart type and restarting side |
| `pipeline4/s05_possession.py` | Home/away in live play |
| `pipeline4/s06_assemble.py` | Final label per frame, consistency checks |
| `pipeline4/s07_temporal.py` | Removes short spells with nothing behind them; writes the submission file |

Each stage reads the previous stages' outputs and writes its own, with a `meta.json` holding every
parameter and the stage's self-checks. Any stage can be rerun on its own.

## Switches

The defaults reproduce the submission. Set a variable before running to turn a rule off (or on):
`MTAH_WHISTLE`, `MTAH_STILL_BALL`, `MTAH_LOWCOUNT`, `MTAH_GOAL_CHAIN`, `MTAH_CONFIRMED_HOLD`,
`MTAH_STALE`, `MTAH_FIRST_TOUCH`, `MTAH_ORPHAN`, `MTAH_BALL_TRUST`, `MTAH_FAST_FLIGHT` (on by
default, `=0` turns off); `MTAH_INFER_OUT`, `MTAH_BALL_TRUST_DROP` (off by default, `=1` turns on).

## Challenge documents

[CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md),
[RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md), [FAQ.md](FAQ.md) and
[OVERVIEW.md](OVERVIEW.md) are the organisers' and unchanged. The `evaluation/` folder is theirs too.
