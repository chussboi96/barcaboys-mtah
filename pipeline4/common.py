"""Shared constants, paths and I/O for every pipeline stage.

Everything that names a file, a frame count or a pitch size comes from games.py, so the pipeline
and the official tooling cannot disagree about them.

Outputs go to work/ (or MTAH_WORK_DIR). work/ is gitignored: it holds frame-level tracking
derived from the gated data, which the terms of use forbid publishing.
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games  # noqa: E402

GAME = games.GAME
FPS = int(games.GAMES[GAME]["fps"])
N_FRAMES = int(games.source_frames("tactical", GAME))
PITCH_LENGTH, PITCH_WIDTH = (float(v) for v in games.GAMES[GAME]["pitch"])
PITCH_SCALE = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)

WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(ROOT, "work_pipeline4")

# Sides, as codes and as submission letters.
HOME, AWAY = 0, 1

# Period codes. Index into PERIOD_NAMES.
PRE, H1, HT, H2, POST = range(5)
PERIOD_NAMES = ("pre_match", "half_1", "half_time", "half_2", "post_match")

# Initial label codes written by s00. Later stages only ever replace UNDECIDED.
LABEL_X = "X"
LABEL_UNDECIDED = "?"


def stage_dir(stage, work_dir=None):
    """work/<stage>/, created on demand."""
    path = os.path.join(work_dir or WORK_DIR, stage)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_assets(data_dir=None, source="tactical"):
    """games.asset_paths(), plus a tolerant lookup for the video.

    The download names videos BAR-ATH_tactical-001.mp4 / BAR-ATH_broadcast-002.mp4 while
    games.py expects BAR-ATH_tactical.mp4. Renaming is the clean fix; until then, any single
    <stem>*.mp4 in the source folder is accepted. Tracking, homography and tagging files are
    required to have the exact names games.py builds.
    """
    paths = games.asset_paths(GAME, data_dir, source)
    video = paths.get("video")
    if video and not os.path.isfile(video):
        stem = os.path.splitext(os.path.basename(video))[0]
        candidates = sorted(glob.glob(os.path.join(os.path.dirname(video), stem + "*.mp4")))
        paths["video"] = candidates[0] if len(candidates) == 1 else None
    return paths


def require_files(paths, names):
    """Raise one readable error listing every missing required file."""
    missing = [paths[name] for name in names if not paths.get(name) or not os.path.isfile(paths[name])]
    if missing:
        raise IOError("match data not found:\n  " + "\n  ".join(str(m) for m in missing) +
                      "\nPass --data-dir or set MTAH_DATA_DIR (see DATA.md).")


def to_metres(xy):
    """Normalised (0-1) pitch coordinates -> metres. Works on (..., 2) arrays; NaN stays NaN."""
    return np.asarray(xy, dtype=np.float32) * PITCH_SCALE


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("not JSON serialisable: {!r}".format(type(value)))
