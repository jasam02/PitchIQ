"""Key moments: the coaching review of a match built from player relationships in short situations (who was where,
who covered whom, what space opened), not from team-level statistics.

    data        MatchData: the match (or a Segment run) as compact 5 Hz arrays, built once from the saved views and cached
    context     attacking directions, the phase (who had the ball, as the possession record estimates it), the camera's
                view of the pitch, team units
    roles       tactical roles per player and window, from pairwise relative positions (works with partial views)
    geometry    time to reach, zones, visibility, gaps between players
    detectors   the situation detectors, one module each, all returning candidate episodes
    moments     episodes -> moments with before / during / after windows -> scoring, grouping, selection
    suggest     alternative positions (zones), checked against reach and against opening a new gap
    language    the coach's words for a moment, per level band
    report      the report (summary, strengths, improvements, key moments, patterns, training, metrics, coverage) and
                the evidence records the page's annotation drawing uses

Nothing here re-runs detection, tracking, calibration or identities: everything is read from what the analysis saved.
"""
