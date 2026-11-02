"""Full Match Mode: a whole recorded match analysed as many bounded segment runs of the unchanged pipeline, run in
parallel by local worker processes and assembled into one match.

    plan.py        the segment plan: periods (halves), owned frames, processed frames with context
    manifest.py    the match manifest on disk: settings, segments, their state and artifacts, atomic updates
    system.py      what this computer can run: free memory, GPU memory, CPUs, the AUTO worker count
    worker.py      a worker process: loads the models once, runs segment jobs one after another
    scheduler.py   the scheduler in the server: the worker pool, the queue, retries, pause/stop/resume, progress, ETA
    calibration.py the match's calibration: keyframes shared by every segment, discovered anchors, coverage
    identities.py  match-wide identities: teams across segments, players stitched over the overlaps
    assemble.py    the assembled match: ownership, the ball at the boundaries, the match index, the per-segment views
    analytics.py   match-level analytics from the assembled segments
    corrections.py the coach's roster corrections of a match: merges, exclusions, team changes, merge suggestions
    coaching.py    the match's coaching report: each segment measured over its owned frames, one merged obs, the report
    routes.py      the server's /api/match routes, the match answers of analytics, possession and team shape, and the
                   coaching routes of a match

Short Segment mode does not use any of this: a run of at most 180 s through the page goes the old way.
"""
