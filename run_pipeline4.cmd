@echo off
REM barca boys - More Than A Hack 2027. Regenerates evaluation\results\barca_boys.csv from the match data.
REM Usage:  run_pipeline4.cmd "<data folder>"
REM Needs:  pip install -r requirements.txt, and ffmpeg on the PATH (whistle detection reads the audio).
REM Intermediate outputs go to work_pipeline4\ (git-ignored).
set DATA=%~1
if "%DATA%"=="" (
    echo Usage: run_pipeline4.cmd "<data folder>"
    exit /b 1
)
python -m pipeline4.s00_timeline --data-dir "%DATA%" || exit /b 1
python -m pipeline4.s01w_whistle --data-dir "%DATA%" || exit /b 1
python -m pipeline4.s01_evidence   || exit /b 1
python -m pipeline4.s02_touches    || exit /b 1
python -m pipeline4.s03_stoppages  || exit /b 1
python -m pipeline4.s04_restarts   || exit /b 1
python -m pipeline4.s05_possession || exit /b 1
python -m pipeline4.s06_assemble   || exit /b 1
python -m pipeline4.s07_temporal --results-name barca_boys || exit /b 1
python evaluation\validate_submission.py evaluation\results\barca_boys.csv
