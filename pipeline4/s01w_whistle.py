"""Stage 01w -- referee whistle strength from the tactical video's audio track.

    python -m pipeline4.s01w_whistle --data-dir "D:\\More Than A Hack 2027 - Initial Challenge"
    python -m pipeline4.s01w_whistle --video path\\to\\BAR-ATH_tactical.mp4

Needs ffmpeg on the PATH (the same one viz_possession.py uses).

Writes work/s01w/:
    whistle.npz   strength (N,) float32: whistle strength in dB on every video frame
    whistles.csv  one row per burst at or above WHISTLE_REPORT_DB (for inspection only)
    meta.json

A referee's whistle is a loud, narrow tone in the 3.5-4.5 kHz range lasting a few tenths of a
second. Crowd noise is broadband: it raises that band and the bands either side of it together.
So the measure is the energy in a whistle band relative to the bands either side of it, in dB, taken
relative to its own running median (the crowd of that minute), and a whistle has to hold that
contrast for WHISTLE_MIN_HOLD_S. Two overlapping whistle bands are
measured and the stronger is kept, because referees' whistles differ in pitch.

This stage only measures. Stage 03 decides where a whistle is used (stoppage onsets with no ball-out
and no set-piece tag). Nothing here is fitted to any ground truth.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline4 import common as C  # noqa: E402

STAGE = "s01w"
F = C.FPS

SAMPLE_RATE = 24000                      # Hz; covers the whistle bands and their flanks
FFT_SIZE = 1024                          # 43 ms window
HOP = SAMPLE_RATE // (4 * F)             # 10 ms: four hops per video frame
WHISTLE_BANDS = ((3600, 4000), (3900, 4400))   # Hz
FLANK_GAP, FLANK_WIDTH = 200, 700        # Hz: flanks start 200 Hz outside a band and are 700 Hz wide
WHISTLE_MIN_HOLD_S = 0.16                # the contrast must hold this long
WHISTLE_REPORT_DB = 10.0                 # bursts listed in whistles.csv
# Adaptive level: the band contrast is measured against its own running median over
# +/- BACKGROUND_HALF_WINDOW_S, so a whistle is judged against the crowd noise of that moment
# rather than against a fixed level for the whole match.
BACKGROUND_HALF_WINDOW_S = 30.0
BACKGROUND_STEP = 10                     # hops between median samples (0.1 s)
CHUNK_S = 60


def band_ratio(power, freqs):
    """Per hop: max over WHISTLE_BANDS of 10 log10(band mean / mean of the two flanks)."""
    best = np.full(power.shape[0], -np.inf)
    for lo, hi in WHISTLE_BANDS:
        band = power[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
        below = power[:, (freqs >= lo - FLANK_GAP - FLANK_WIDTH) & (freqs < lo - FLANK_GAP)].mean(axis=1)
        above = power[:, (freqs >= hi + FLANK_GAP) & (freqs < hi + FLANK_GAP + FLANK_WIDTH)].mean(axis=1)
        ratio = 10.0 * np.log10((band + 1e-12) / (0.5 * (below + above) + 1e-12))
        best = np.maximum(best, ratio)
    return best


def hop_ratios(video):
    """Decode the audio with ffmpeg and return the band ratio of every hop."""
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    window = np.hanning(FFT_SIZE).astype(np.float32)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    out = []
    carry = np.zeros(0, dtype=np.float32)
    chunk_bytes = SAMPLE_RATE * CHUNK_S * 2
    while True:
        raw = proc.stdout.read(chunk_bytes)
        if not raw:
            break
        x = np.concatenate([carry, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0])
        n = (len(x) - FFT_SIZE) // HOP + 1
        if n <= 0:
            carry = x
            continue
        idx = np.arange(FFT_SIZE)[None, :] + HOP * np.arange(n)[:, None]
        power = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
        out.append(band_ratio(power, freqs))
        carry = x[n * HOP:]
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to decode the audio of {}".format(video))
    return np.concatenate(out) if out else np.zeros(0)


def relative_to_background(ratio):
    """Band contrast minus its running median over +/- BACKGROUND_HALF_WINDOW_S."""
    coarse = ratio[::BACKGROUND_STEP]
    half = int(round(BACKGROUND_HALF_WINDOW_S * SAMPLE_RATE / HOP / BACKGROUND_STEP))
    med = pd.Series(coarse).rolling(2 * half + 1, center=True, min_periods=half // 6).median().to_numpy()
    med = np.repeat(np.nan_to_num(med, nan=0.0), BACKGROUND_STEP)[:len(ratio)]
    return ratio - med


def frame_strength(ratio, n_frames):
    """Whistle strength per video frame: the contrast held for WHISTLE_MIN_HOLD_S starting there."""
    ratio = relative_to_background(ratio)
    hold = max(1, int(round(WHISTLE_MIN_HOLD_S * SAMPLE_RATE / HOP)))
    sustained = pd.Series(ratio).rolling(hold, min_periods=hold).min().shift(-(hold - 1)).to_numpy()
    sustained = np.nan_to_num(sustained, nan=-99.0)
    per_frame = int(round(SAMPLE_RATE / HOP / F))
    usable = min(len(sustained) // per_frame, n_frames)
    strength = np.full(n_frames, -99.0, dtype=np.float32)
    strength[:usable] = sustained[:usable * per_frame].reshape(usable, per_frame).max(axis=1)
    return strength


def bursts(strength, level):
    rows = []
    above = strength >= level
    edges = np.diff(np.r_[0, above.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        rows.append({"start_frame": int(a), "end_frame": int(b) - 1,
                     "time": "{}:{:05.2f}".format(int(a / F // 60), a / F % 60),
                     "peak_db": round(float(strength[a:b].max()), 2)})
    return pd.DataFrame(rows, columns=["start_frame", "end_frame", "time", "peak_db"])


def build(data_dir=None, video=None, work_dir=None):
    t0 = time.time()
    if video is None:
        video = C.resolve_assets(data_dir, "tactical").get("video")
    if not video or not os.path.isfile(video):
        raise SystemExit("tactical video not found -- pass --data-dir or --video")
    n = int(C.N_FRAMES)
    print("decoding audio of {}".format(video))
    ratio = hop_ratios(video)
    strength = frame_strength(ratio, n)
    table = bursts(strength, WHISTLE_REPORT_DB)
    out = C.stage_dir(STAGE, work_dir)
    np.savez(os.path.join(out, "whistle.npz"), strength=strength)
    table.to_csv(os.path.join(out, "whistles.csv"), index=False)
    C.write_json(os.path.join(out, "meta.json"), {
        "stage": STAGE, "video": os.path.basename(video), "n_frames": n,
        "parameters": {k: v for k, v in globals().items()
                       if k.isupper() and isinstance(v, (int, float, tuple, str))},
        "bursts_at_report_level": int(len(table)),
        "strength_percentiles_db": {str(p): round(float(np.percentile(strength[strength > -99], p)), 2)
                                    for p in (50, 90, 99, 99.9)},
    })
    print("{} bursts >= {} dB; wrote {} in {:.0f}s".format(len(table), WHISTLE_REPORT_DB, out, time.time() - t0))
    return out


def load_whistle(work_dir=None):
    """Per-frame whistle strength in dB, or None when stage 01w has not been run."""
    path = os.path.join(work_dir or C.WORK_DIR, STAGE, "whistle.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as data:
        return data["strength"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="folder of the match data (as for s00)")
    parser.add_argument("--video", default=None, help="tactical video file (overrides --data-dir)")
    parser.add_argument("--work-dir", default=None, help="output root (default: ./work)")
    args = parser.parse_args(argv)
    build(args.data_dir, args.video, args.work_dir)


if __name__ == "__main__":
    main()
