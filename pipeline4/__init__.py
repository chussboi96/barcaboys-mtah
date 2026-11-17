"""Frame-level possession pipeline for More than a Hack 2027.

Stages, each a module runnable on its own and each writing to work/<stage>/:

    s00_timeline   one record per video frame; ATD, ball, players, homography and Smart Tagging
                   aligned to the zero-based submission index; out-of-period frames marked X
"""
