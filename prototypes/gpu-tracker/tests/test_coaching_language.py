"""The words of the coaching report (soccer/coaching/language.py): every template is plain ASCII, free of digits,
blame, exaggeration, intent and phases of play; every rendered text passes the fact check against the observation it
came from; context levels change words, not numbers; a short segment never claims a repeated pattern; and the fact
check catches what a careless engine would invent.

Fix round 1 (the coach and fact reviews of the first reports on the four test runs) adds: a strength states the
professional median and a "to build on it" line and never says "within the professional upper level"; development
headlines name no professional yardstick and development text has no "median", "standard deviation" or "(est.)";
every compactness and spacing finding carries the phase hedge; a pooled strength names its strictest context; neutral
notes say their state (no spell, one spell with its times, several within the professional count, measured too
briefly); formation readings of one family are one shape, role stability is "N of M players kept one role"; a
player's support-distance review is a role check; the VIDEO line says "and others"; a not-assessed line carries its
own reasons only (the run's are in the data-quality note), the small-scene veto names its rule; an id that is no
longer in the roster is marked as such.

The Observation records below follow the contract's schema; their values are shaped after the audited runs
(broadcast 0-60 s, broadcast 42-222 s, amateur 118-182 s) and the Metrica game 1 reference, so the sentences read
as they will on real data. test_coaching_report.py builds its reports from the same builders.

Run:  python -m unittest test_coaching_language          (from this directory)
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import drills, language  # noqa: E402
from soccer.coaching.episodes import stable_id  # noqa: E402

LEVELS = language.LEVELS
TEXT_FIELDS = ('headline', 'whatHappened', 'whyItMatters', 'whatToImprove', 'howToPractice', 'video', 'priority',
               'confidenceNote', 'note', 'limitation')
CATEGORY = {key: cat for key, cat, _ in language.DETECTORS.values()}


def occurrence(t0, t1, metric, peak, mean=None, players=(), censored=False, fps=30.0, **more):
    peak_t = round((t0+t1)/2, 3)
    occ = {'id': stable_id('occ', 'x', t0, t1), 't0': t0, 't1': t1, 'peakT': peak_t,
           'frame0': int(round(t0*fps)), 'frame1': int(round(t1*fps)), 'peakFrame': int(round(peak_t*fps)),
           'censored': censored, 'players': list(players),
           'metrics': {metric: {'peak': peak, 'mean': peak-1.0 if mean is None else mean}}}
    for name, (p, m) in more.items():
        occ['metrics'][name] = {'peak': p, 'mean': m}
    return occ


def baseline(enter, exit_, p50=.29, p90=.37, share=.10, context='all'):
    return {'reference': 'metrica-sample-game-1@0123456789abcdef', 'rule': 'default', 'context': context,
            'threshold': {'enter': enter, 'exit': exit_}, 'professionalRatePerMinute': {'p10': .1, 'p50': p50,
                                                                                         'p90': p90},
            'professionalShareAbove': share}


def strong(median, p50, p75, share75=0.0, **more):
    """A strength's metrics as detectors.strength_check leaves them: the median against the professional median and
    the share of time beyond the professional p75 of the same context."""
    return dict({'median': median, 'professionalP50': p50, 'professionalP75': p75,
                 'shareAboveProfessionalP75': share75, 'favourableSide': True, 'upperBoundShareAboveP75': .2,
                 'strengthFailed': []}, **more)


def observation(detector, classification, pattern=None, team='A', t0=0.0, t1=59.97, measured=57.4, eligible=.957,
                longest=26.0, occurrences=(), metrics=None, base=None, confidence='high', reasons=(), players=None,
                severity=None, evidence=(), drill_ids=None, status='reported', segment_start=0.0, key=None,
                category=None, **extra):
    rec_key, cat, default_pattern = language.DETECTORS[detector]
    pattern = pattern or default_pattern
    occurrences = list(occurrences)
    if players is None:
        players = sorted({p for o in occurrences for p in o.get('players') or ()})
    sev = severity
    if sev is None:
        sev = {'strength': 'observed_strength', 'improvement': 'potential_improvement',
               'review': 'review_opportunity'}.get(classification)
    count = len(occurrences)
    obs = {'id': stable_id('obs', detector, pattern, team, t0, t1), 'engineVersion': '1', 'team': team,
           'detector': detector, 'category': category or cat, 'pattern': pattern, 'classification': classification,
           'severity': sev,
           'window': {'t0': t0, 't1': t1, 'segmentT0': round(t0-segment_start, 3),
                      'segmentT1': round(t1-segment_start, 3)},
           'measuredSeconds': measured, 'eligibleShare': eligible, 'longestStretchSeconds': longest,
           'occurrences': occurrences, 'occurrenceCount': count,
           'ratePerMeasuredMinute': round(count/(measured/60.0), 3) if measured else None,
           'baseline': base, 'metrics': metrics or {}, 'players': list(players),
           'confidence': {'tracking': confidence, 'calibration': confidence, 'identity': confidence, 'ball': 'n/a',
                          'overall': confidence, 'reasons': list(reasons)},
           'evidence': [e['id'] for e in evidence], 'recommendationKey': key or rec_key,
           'drillIds': list(drill_ids) if drill_ids is not None else [],
           'status': status, 'suppressionReason': None if status == 'reported' else 'duplicate'}
    obs.update(extra)
    if 'id' in extra:
        obs['id'] = extra['id']
    return obs


def evidence_for(obs, occ, players=None, metric=None, fps=30.0, segment_start=0.0):
    """An evidence record (contract section 6) at the occurrence's peak frame, attached to obs."""
    frame = occ['peakFrame']
    t = round(frame/fps, 3)
    metric = metric or next(iter(occ['metrics']))
    ev = {'id': stable_id('ev', obs['id'], occ['id']), 'observation': obs['id'], 'occurrence': occ['id'],
          'time': t, 'frame': frame, 'segmentTime': round(t-segment_start, 3), 'seek': round((frame+.5)/fps, 4),
          'fps': fps, 'players': list(players if players is not None else occ['players']),
          'metrics': {metric: occ['metrics'][metric]['peak']}, 'confidence': 'high', 'tier': 'pinned',
          'calibration': {'confidence': .82, 'paintResidualPx': 4.1, 'disagreement': None},
          'diagram': {'preset': 'defensive_gap', 'title': 'x', 't': t, 'frame': frame, 'team': obs['team'],
                      'attack': 'right', 'metresEstimated': True, 'pitch': {'length': 105, 'width': 68},
                      'elements': [{'type': 'player', 'id': 'A-03', 'team': 'A', 'x': .41, 'y': .32, 'kind': 'O'}]},
          'videoOverlay': None, 'caption': 'Defence to midfield gap at its largest.'}
    obs['evidence'].append(ev['id'])
    return ev


def moment(obs, time, players, metric, value, fps=30.0):
    """A typical-moment evidence record (a strength's or a neutral's), attached to obs."""
    frame = int(round(time*fps))
    ev = {'id': stable_id('ev', obs['id'], time), 'observation': obs['id'], 'occurrence': None, 'time': time,
          'frame': frame, 'segmentTime': time, 'seek': round((frame+.5)/fps, 4), 'fps': fps, 'players': list(players),
          'metrics': {metric: value}, 'confidence': 'high', 'tier': 'world', 'kind': 'typical',
          'calibration': {'confidence': .9, 'paintResidualPx': 2.0, 'disagreement': None}, 'diagram': None,
          'videoOverlay': None, 'caption': None}
    obs['evidence'].append(ev['id'])
    return ev


# --- Fixture sets -------------------------------------------------------------------------------------------------

def broadcast_strengths(team='A'):
    """Broadcast 0-60 s, the defending team: compact and level (the run's own numbers: D-M gap median 8.0 m against a
    professional median of 13.0 m, back-line spread 0.9 m against 3.2 m, width 38.8 m against 43.0 m, 4-4-2 clear
    over 60 s with every player in one role; the strength rule's p75 levels of Metrica game 1)."""
    return [
        observation('unit_gap_dm', 'strength', team=team, measured=59.4,
                    metrics=strong(8.0, 13.0, 16.1, p90=10.2, shareAboveProfessionalP90=0.0),
                    base=baseline(22.5, 20.2, context='third:middle'),
                    players=['A-02', 'A-03', 'A-04', 'A-05', 'A-06', 'A-07', 'A-08', 'A-09']),
        observation('back_line_coordination', 'strength', team=team, measured=60.0,
                    metrics=strong(.94, 3.2, 5.5, p90=1.8, shareAboveProfessionalP90=0.0),
                    base=baseline(9.7, 7.8, context='backline:4'), players=['A-02', 'A-03', 'A-04', 'A-05']),
        observation('compactness', 'strength', 'team_width', team=team, measured=59.6,
                    metrics=strong(38.8, 43.0, 51.9, p90=42.4, shareAboveProfessionalP90=.0),
                    base=baseline(60.1, 57.7)),
        observation('formation_stability', 'strength', 'formation_stable', team=team, t1=320.0, measured=310.0,
                    metrics={'label': '4-4-2', 'labelConfidence': 'clear', 'roleStability': 1.0,
                             'playersKeptRole': 10, 'playersAssessed': 10, 'roleShareThreshold': .75,
                             'defenderSetChanges': 0, 'alternative': '5-4-1'}, confidence='medium',
                    reasons=['formation_unverified']),
    ]


def veo_insufficient(team='A'):
    """Veo 1-31 s: 2-3 players per team identified, nothing measurable at team level."""
    few = ['few_players', 'incomplete_team']
    return [
        observation('unit_gap_dm', 'insufficient', team=team, t0=1.0, t1=30.97, measured=0.0, eligible=0.0,
                    longest=0.0, reasons=['incomplete_units'] + few, confidence='low',
                    insufficientReasons=['short_measured', 'incomplete_units', 'few_players']),
        observation('compactness', 'insufficient', 'team_width', team=team, t0=1.0, t1=30.97, measured=0.6,
                    eligible=.02, longest=0.6, reasons=few, confidence='low',
                    insufficientReasons=['short_measured', 'incomplete_team', 'few_players']),
        observation('back_line_coordination', 'insufficient', team=team, t0=1.0, t1=30.97, measured=0.0,
                    eligible=0.0, longest=0.0, reasons=['no_formation'], confidence='low',
                    insufficientReasons=['short_measured', 'no_formation', 'few_players']),
        observation('formation_stability', 'insufficient', 'formation_stable', team=team, t0=1.0, t1=30.97,
                    measured=0.0, eligible=0.0, longest=0.0, reasons=['no_formation', 'mystery_code_x'],
                    confidence='low', insufficientReasons=['no_formation', 'mystery_code_x', 'few_players']),
    ]


def broadcast_mixed(team='A', start=42.433):
    """Broadcast 42-222 s, the attacking team: an ambiguous 4-3-3 with eight back-line group changes, a D-M gap
    beyond the professional enter level three times, an uneven line, a high line with a gap, a late lateral shift,
    one player far from his team-mates (a role check), a neutral width and the ball context; plus an insufficient
    measurement, a suppressed duplicate and an observation of the other team."""
    t0, t1 = start, 222.4
    out, evs = [], []
    gap_occ = [occurrence(61.2, 68.4, 'gapDM', 23.1, players=['A-04', 'A-08', 'A-05', 'A-06']),
               occurrence(103.0, 109.6, 'gapDM', 24.8, players=['A-04', 'A-08', 'A-05', 'A-06'], censored=True),
               occurrence(151.4, 158.0, 'gapDM', 22.9, players=['A-04', 'A-12', 'A-05', 'A-06'])]
    gap = observation('unit_gap_dm', 'improvement', team=team, t0=t0, t1=t1, measured=132.4, eligible=.735,
                      longest=62.0, occurrences=gap_occ, metrics={'median': 11.8, 'p90': 15.0,
                      'shareAboveProfessionalP90': .14, 'professionalWindowP99': 2.0},
                      base=baseline(22.5, 20.2, context='third:final', p50=.29, p90=.37), segment_start=start,
                      drill_ids=['defensive_compactness'])
    evs += [evidence_for(gap, gap_occ[0], segment_start=start), evidence_for(gap, gap_occ[2], segment_start=start)]
    out.append(gap)
    line_occ = [occurrence(80.0, 88.0, 'lineStd', 10.4, players=['A-04', 'A-08', 'A-12', 'A-02']),
                occurrence(120.0, 127.6, 'lineStd', 11.1, players=['A-04', 'A-08', 'A-12', 'A-02']),
                occurrence(170.2, 176.0, 'lineStd', 9.9, players=['A-04', 'A-08', 'A-03', 'A-02']),
                occurrence(184.0, 190.0, 'lineStd', 10.0, players=['A-04', 'A-08', 'A-03', 'A-02'])]
    line = observation('back_line_coordination', 'improvement', team=team, t0=t0, t1=t1, measured=186.0,
                       eligible=.633, longest=40.0, occurrences=line_occ, metrics={'median': 4.6, 'p90': 7.7,
                       'shareAboveProfessionalP90': .21}, base=baseline(9.7, 7.8, context='backline:4', p50=.2,
                       p90=.45), severity='high_confidence_concern', segment_start=start,
                       drill_ids=['back_line_coordination'])
    evs.append(evidence_for(line, line_occ[1], segment_start=start))
    out.append(line)
    hl_occ = [occurrence(96.0, 104.0, 'gapDM', 24.0, players=['A-04', 'A-08', 'A-12', 'A-02'],
                         lineHeight=(57.0, 55.2))]
    out.append(observation('line_height_and_gap', 'review', team=team, t0=t0, t1=t1, measured=114.0,
                           occurrences=hl_occ, base=baseline(22.5, 20.2), segment_start=start,
                           drill_ids=['back_line_coordination']))
    out.append(observation('formation_stability', 'review', 'formation_ambiguous', team=team, t0=t0, t1=t1,
                           measured=140.0, eligible=.78, metrics={'label': '4-3-3', 'labelConfidence': 'ambiguous',
                           'alternative': '3-5-2', 'roleStability': .5, 'playersKeptRole': 5, 'playersAssessed': 10,
                           'roleShareThreshold': .75, 'defenderSetChanges': 8, 'distinctDefenderSets': 6,
                           'labels': ['4-3-3']}, confidence='medium', reasons=['identity_switches'],
                           segment_start=start, drill_ids=['formation_shadow_play']))
    lat_occ = [occurrence(130.0, 160.0, 'lagDM', 1.9)]
    out.append(observation('lateral_shift', 'improvement', team=team, t0=t0, t1=t1, measured=62.0, eligible=.34,
                           longest=62.0, occurrences=lat_occ, metrics={'lag': 1.6, 'corr': .55},
                           base=baseline(1.2, 0.7), segment_start=start))
    iso_occ = [occurrence(140.0, 152.0, 'nearestTeammate', 21.3, players=['A-11']),
               occurrence(175.0, 183.6, 'nearestTeammate', 19.8, players=['A-11'])]
    iso = observation('isolation', 'review', team=team, t0=t0, t1=t1, measured=120.0, eligible=.68,
                      longest=55.0, occurrences=iso_occ, metrics={'median': 10.4, 'p90': 16.6,
                      'shareAboveProfessionalP90': .22, 'professionalWindowP99': 1.0, 'roleCheck': True},
                      base=baseline(18.0, 16.0, p50=.5, p90=.8, context='unit:F'), players=['A-11'],
                      segment_start=start)
    evs.append(evidence_for(iso, iso_occ[0], segment_start=start))
    out.append(iso)
    out.append(observation('compactness', 'neutral', 'team_width', team=team, t0=t0, t1=t1, measured=104.6,
                           metrics={'median': 55.1, 'p90': 58.7, 'shareAboveProfessionalP90': .25,
                                    'professionalP50': 44.0, 'favourableSide': False,
                                    'strengthFailed': ['unfavourable_side', 'evidence_bound']},
                           base=baseline(60.1, 57.7), segment_start=start, state='absent'))
    out.append(observation('ball_context', 'neutral', team=team, t0=t0, t1=t1, measured=154.8,
                           metrics={'thirds': {'own': .18, 'middle': .45, 'final': .37}}, segment_start=start,
                           state='descriptive'))
    out.append(observation('unit_gap_mf', 'insufficient', team=team, t0=t0, t1=t1, measured=12.0, eligible=.07,
                           longest=4.4, reasons=['incomplete_units'], confidence='low', segment_start=start,
                           insufficientReasons=['short_measured', 'incomplete_units']))
    out.append(observation('compactness', 'improvement', 'team_depth', team=team, t0=t0, t1=t1, measured=100.0,
                           occurrences=[occurrence(70.0, 80.0, 'depth', 47.0), occurrence(90.0, 99.0, 'depth', 48.0)],
                           base=baseline(46.5, 43.1), status='suppressed', segment_start=start))
    other = 'B' if team == 'A' else 'A'
    out.append(observation('unit_gap_dm', 'strength', team=other, t0=t0, t1=t1, measured=136.2,
                           metrics=strong(8.8, 9.4, 12.4, .07), base=baseline(22.5, 20.2), segment_start=start))
    return out, evs


def amateur_set(team='A'):
    """Amateur 118-182 s (keyless MEDIUM registration, small scene): one medium-confidence observation that met the
    rules, the rest insufficient for want of continuous stretches."""
    start = 118.2
    occ = [occurrence(124.0, 130.2, 'gapDM', 19.1, players=['A-02', 'A-05', 'A-06', 'A-09']),
           occurrence(138.4, 144.0, 'gapDM', 18.8, players=['A-02', 'A-05', 'A-06', 'A-09'])]
    gap = observation('unit_gap_dm', 'improvement', team=team, t0=start, t1=182.0, measured=40.0, eligible=.63,
                      longest=12.6, occurrences=occ, metrics={'median': 16.6, 'p90': 18.8,
                      'shareAboveProfessionalP90': .3}, base=baseline(17.6, 13.2, context='third:own', p50=.1,
                      p90=.3), confidence='medium', reasons=['medium_registration', 'small_scene', 'keyless'],
                      segment_start=start)
    ev = [evidence_for(gap, occ[0], segment_start=start)]
    rest = [
        observation('compactness', 'insufficient', 'team_width', team=team, t0=start, t1=182.0, measured=20.4,
                    eligible=.32, longest=6.6, reasons=['short_stretch', 'keyless', 'small_scene'], confidence='low',
                    segment_start=start, insufficientReasons=['short_stretch', 'incomplete_team']),
        observation('back_line_coordination', 'insufficient', team=team, t0=start, t1=182.0, measured=26.8,
                    eligible=.42, longest=5.4, reasons=['short_stretch', 'small_scene', 'keyless'],
                    confidence='medium', segment_start=start,
                    insufficientReasons=['short_stretch', 'incomplete_units']),
        observation('isolation', 'insufficient', team=team, t0=start, t1=182.0, measured=0.0, eligible=0.0,
                    longest=0.0, reasons=['keyless', 'small_scene'], confidence='low', segment_start=start,
                    insufficientReasons=['short_measured', 'small_scene_veto', 'broadcast_only']),
        observation('ball_context', 'insufficient', team=team, t0=start, t1=182.0, measured=0.0, eligible=0.0,
                    longest=0.0, reasons=['keyless', 'small_scene'], confidence='low', segment_start=start,
                    insufficientReasons=['no_ball', 'small_scene_veto']),
        observation('formation_stability', 'review', 'formation_ambiguous', team=team, t0=start, t1=182.0,
                    measured=40.0, metrics={'label': '4-2-3-1', 'labelConfidence': 'likely',
                    'alternative': '4-4-1', 'roleStability': .7, 'defenderSetChanges': 2,
                    'distinctDefenderSets': 2}, confidence='low', segment_start=start),
    ]
    return [gap]+rest, ev


def corrected_set():
    """After corrections: internal id B-12 shows as B-09, and B-05 was withdrawn."""
    occ = [occurrence(20.0, 31.0, 'nearestTeammate', 19.4, players=['B-12']),
           occurrence(40.0, 47.2, 'nearestTeammate', 18.6, players=['B-12'])]
    iso = observation('isolation', 'improvement', team='B', measured=58.0, occurrences=occ,
                      metrics={'median': 12.0, 'p90': 17.0, 'shareAboveProfessionalP90': .2},
                      base=baseline(18.0, 16.0, p50=.5, p90=.8), players=['B-12'])
    ev = [evidence_for(iso, occ[0])]
    return [iso], ev


def compact_block(team='B', start=42.433, pooled=True):
    """Broadcast 42-222 s team B: a compact block, with width, length and area all strengths (pooled over the thirds,
    each quoting its strictest third) and both unit gaps strengths; typical moments for each."""
    t0, t1 = start, 222.4
    obs, evs = [], []
    shapes = (('compactness', 'team_width', 'width', 38.7, 39.3, 40.3, 48.0, .05, 'third:own', 68.2),
              ('compactness', 'team_depth', 'depth', 18.4, 18.9, 31.5, 36.3, .008, 'third:middle', 73.8),
              ('compactness', 'team_area', 'hull', 491.0, 505.0, 644.0, 878.0, .09, 'third:own', 68.0),
              ('unit_gap_dm', 'space_between_defence_and_midfield', 'gapDM', 8.5, 8.8, 9.4, 12.4, .12, 'third:own',
               68.0),
              ('unit_gap_mf', 'space_between_midfield_and_forwards', 'gapMF', 7.0, 7.5, 11.7, 15.0, 0.0,
               'third:middle', 66.4))
    for k, (det, pat, metric, med, all_med, p50, p75, share, ctx, comp) in enumerate(shapes):
        more = dict(pooledMedian=all_med, strictestContext=ctx, comparisonSeconds=comp) if pooled else {}
        o = observation(det, 'strength', pat, team=team, t0=t0, t1=t1, measured=143.0-k, segment_start=start,
                        metrics=strong(med, p50, p75, share, **more), confidence='medium',
                        reasons=['medium_registration'], base=baseline(50.0, 45.4, context=ctx))
        evs.append(moment(o, round(60.0+10*k, 3), [f'{team}-0{k+1}', f'{team}-0{k+2}'], metric, med))
        obs.append(o)
    return obs, evs


def every_finding():
    """One observation per (recommendation key, classification) the templates support, plus neutral notes."""
    obs, evs = broadcast_mixed()
    obs = [o for o in obs if o['team'] == 'A' and o['status'] == 'reported']
    obs += broadcast_strengths()
    obs.append(observation('lateral_shift', 'strength', measured=63.0,
                           metrics={'lag': -0.2, 'corr': .9, 'professionalP50': 0.2, 'professionalP75': 0.4,
                                    'shareAboveProfessionalP75': 0.0, 'lagResolutionSeconds': 0.2,
                                    'windowSeconds': 20.0},
                           base=baseline(1.2, 0.7)))
    obs.append(observation('isolation', 'strength', measured=55.0, players=['A-07'],
                           metrics=strong(8.0, 9.5, 12.6, .02), base=baseline(18.0, 16.0)))
    obs.append(observation('compactness', 'strength', 'team_area', measured=58.0, metrics=strong(490.0, 943.0, 1220.0),
                           base=baseline(1528.0, 1379.0)))
    obs.append(observation('formation_stability', 'review', 'formation_change', measured=150.0,
                           metrics={'labels': ['4-4-2', '3-5-2'], 'labelSeconds': {'4-4-2': 80.0, '3-5-2': 70.0},
                                    'roleStability': .62, 'playersKeptRole': 6, 'playersAssessed': 10,
                                    'roleShareThreshold': .75, 'defenderSetChanges': 3, 'distinctDefenderSets': 3}))
    obs.append(observation('formation_stability', 'neutral', 'formation_stable', measured=50.0, state='under_measured',
                           metrics={'label': '3-4-3', 'labelConfidence': 'likely', 'labels': ['3-4-3'],
                                    'strengthSeconds': 300.0}))
    obs.append(observation('formation_stability', 'neutral', 'formation_ambiguous', measured=35.0,
                           state='related_shapes', metrics={'label': '4-1-4-1', 'labelConfidence': 'ambiguous',
                                                             'alternative': '4-3-3', 'labels': ['4-1-4-1'],
                                                             'playersKeptRole': 10, 'playersAssessed': 10,
                                                             'roleShareThreshold': .75}))
    obs.append(observation('compactness', 'neutral', 'team_width', measured=58.0, state='one_spell',
                           metrics={'median': 57.0, 'professionalP50': 43.0, 'favourableSide': False,
                                    'spell': {'t0': 11.0, 't1': 29.4, 'peakT': 14.4, 'peak': 62.3, 'seconds': 18.6}},
                           occurrences=[occurrence(11.0, 29.4, 'width', 62.3)], base=baseline(60.3, 58.0)))
    obs.append(observation('line_height_and_gap', 'neutral', measured=60.0, state='absent',
                           metrics={'lineHeight': 30.6, 'lineHeightSeconds': 60.0, 'opponentLineHeight': 50.1,
                                    'opponentLineHeightSeconds': 60.0, 'professionalLineHeightP50': 38.1}))
    long_occ = [occurrence(10.0+60*i, 18.0+60*i, 'gapMF', 23.0+i) for i in range(5)]
    obs.append(observation('unit_gap_mf', 'improvement', measured=300.0, t1=320.0, occurrences=long_occ,
                           base=baseline(24.5, 20.7), metrics={'median': 13.0, 'professionalWindowP99': 2.0}))
    obs.append(observation('compactness', 'improvement', 'team_area', measured=50.0,
                           occurrences=[occurrence(5.0, 25.0, 'hull', 1650.0)], base=baseline(1528.0, 1379.0)))
    return obs, evs


def render(obs, evs=(), level='competitive', team='A', kind='segment', library=None, label=None, run_codes=()):
    library = library or drills.load_drills()
    drill = drills.choose(library, obs) if obs['classification'] in ('improvement', 'review') else None
    ev = tuple(e for e in evs if e['id'] in obs['evidence'])
    if label is None:
        label = language.LABELS.get(obs.get('severity'))
    ctx = language.FindingContext(obs, ev, level, team, None, kind, drill, True, label, tuple(run_codes))
    return language.TemplateEngine().render(ctx), language.facts_for(obs, ev, drill), drill


def numbers_in(text):
    return sorted(re.findall(r'\d+(?:\.\d+)?', text or ''))


class TemplateTests(unittest.TestCase):
    def test_every_template_string_is_printable_ascii_and_states_no_number(self):
        strings = language.all_template_strings()
        self.assertGreater(len(strings), 150)
        for path, text in strings:
            self.assertRegex(text, r'^[ -~]*$', path)
            self.assertIsNone(re.search(r'\d', text), f'{path} has a digit: {text}')
            words = set(re.findall(r'[a-z]+', text.lower()))
            self.assertFalse(words & set(language.NUMBER_WORDS), f'{path} has a number word')
            self.assertEqual(language.wording_problems(text), [], path)
            text.encode('cp1252')

    def test_reasons_and_caveats_read_well_in_a_short_segment(self):
        # They appear in every report, short ones included: no repeated-pattern words, no raw codes.
        lines = list(language.REASONS.values())+list(language.SECTION_REASONS.values())
        lines += list(language.LIMIT_WORDS.values())+list(language.VETO_WORDS.values())
        lines += language.caveat_notes({'pitch': {'length': 105.0, 'width': 68.0}, 'footage': 'amateur',
                                        'smallScene': True, 'keyless': True, 'corrections': 1,
                                        'registration': {'high': .2, 'medium': .7, 'low': .1}}, 'B')
        lines += language.data_quality({'footage': 'amateur', 'smallScene': True, 'keyless': True},
                                       ('keyless', 'small_scene'), help_spell=True, vetoed=('isolation',))
        lines += language.data_quality({'footage': 'wide'}, ('few_players',), observed={'A': 2.0, 'B': 2.6})
        lines += language.summary_sentences('segment', 'A', None, 0.0, 59.97, 60.0, {'strength': 1}, 'x',
                                            'improvement', 'medium', 2, 'y')
        for text in lines:
            self.assertEqual(language.wording_problems(text, short=True), [], text)
            self.assertNotRegex(text, r'\b[a-z]+_[a-z]+\b', text)

    def test_every_template_has_all_three_levels(self):
        for key, by_cls in language.TEMPLATES.items():
            for cls, tpl in by_cls.items():
                self.assertIn(cls, language.FINDING_CLASSES)
                fields = ['headline', 'whyItMatters', 'whatToImprove'] + (['priority'] if cls != 'strength' else [])
                for f in fields:
                    self.assertEqual(set(tpl[f]), set(LEVELS), f'{key}.{cls}.{f}')
                self.assertTrue(tpl['whatHappened'])
        for pat, info in language.PATTERNS.items():
            for name in ('noun', 'topic', 'stretch', 'formationHeadline', 'units', 'measure', 'short',
                         'strengthName'):
                if name in info:
                    self.assertEqual(set(info[name]), set(LEVELS), f'{pat}.{name}')
            self.assertIn(info['key'], language.RECOMMENDATION_KEYS)
        for name, g in language.GROUPS.items():
            self.assertEqual(set(g['headline']), set(LEVELS), name)
            self.assertEqual(set(g['lead']), set(LEVELS), name)

    def test_every_detector_maps_to_a_key_and_a_pattern(self):
        for det, (key, cat, pat) in language.DETECTORS.items():
            self.assertIn(key, language.RECOMMENDATION_KEYS, det)
            self.assertEqual(language.PATTERNS[pat]['key'], key, det)
            self.assertIn(cat, ('spacing', 'defensive_line', 'formation', 'movement', 'support', 'ball_context'))

    def test_strengths_say_maintained_reviews_say_to_review(self):
        for key, by_cls in language.TEMPLATES.items():
            for level in LEVELS:
                if 'strength' in by_cls:
                    self.assertTrue(by_cls['strength']['headline'][level].startswith('Maintained'), key)
                    self.assertTrue(by_cls['strength']['whatToImprove'][level].startswith('To build on it'), key)
                if 'review' in by_cls:
                    self.assertTrue(by_cls['review']['headline'][level].startswith('To review:'), key)
                    self.assertTrue(by_cls['review']['whatToImprove'][level].startswith('To review'), key)
        for g in language.GROUPS.values():
            self.assertTrue(all(h.startswith('Maintained') for h in g['headline'].values()))

    def test_no_professional_upper_level_and_no_yardstick_in_development_headlines(self):
        # "within the professional upper level" hid how far below the range a team sat (broadcast 0-60 s team A's
        # length: 17.8 m against a professional median of 31.5 m) and is jargon; a youth team is not measured
        # against professionals in its headline.
        for path, text in language.all_template_strings():
            self.assertNotIn('upper level', text, path)
            self.assertNotIn('reference range', text if 'strength' in path else '', path)
        for key, by_cls in language.TEMPLATES.items():
            for cls, tpl in by_cls.items():
                self.assertNotIn('professional', tpl['headline']['development'].lower(), (key, cls))
        for g in language.GROUPS.values():
            self.assertNotIn('professional', g['headline']['development'].lower())


class RenderTests(unittest.TestCase):
    def test_every_rendered_text_passes_the_fact_check_at_every_level(self):
        obs, evs = every_finding()
        rendered = 0
        for o in obs:
            short = o['measuredSeconds'] < language.SHORT_SEGMENT_SECONDS
            for level in LEVELS:
                out, facts, _ = render(o, evs, level)
                for f in TEXT_FIELDS:
                    text = out.get(f)
                    if text is None:
                        continue
                    rendered += 1
                    self.assertEqual(language.validate_text(text, facts), [], f'{o["detector"]} {level} {f}: {text}')
                    self.assertEqual(language.wording_problems(text, short=short, practice=f == 'howToPractice'),
                                     [], f'{o["detector"]} {level} {f}: {text}')
                    self.assertNotIn('{', text)
                    self.assertNotIn('None', text)
                    self.assertNotIn('  ', text)
                    text.encode('ascii')
        self.assertGreater(rendered, 250)

    def test_levels_change_words_not_numbers(self):
        obs, evs = every_finding()
        for o in obs:
            outs = {level: render(o, evs, level)[0] for level in LEVELS}
            for f in ('whatHappened', 'video', 'howToPractice', 'note'):
                nums = {level: numbers_in(outs[level].get(f)) for level in LEVELS}
                self.assertEqual(nums['development'], nums['competitive'], f'{o["detector"]} {f}')
                self.assertEqual(nums['competitive'], nums['advanced'], f'{o["detector"]} {f}')
            if o['classification'] in language.FINDING_CLASSES:
                self.assertEqual(len({outs[lv]['whyItMatters'] for lv in LEVELS}), 3, o['detector'])
                self.assertEqual(len({outs[lv]['headline'] for lv in LEVELS}), 3 if o['classification'] != 'review'
                                 or o['detector'] != 'formation_stability' else 3, o['detector'])

    def test_development_text_is_plain(self):
        # No "median", "standard deviation" or "(est.)" at the development level (the coach review: development
        # readers got "The spread is the standard deviation of the defenders' distances" and "beyond 7.8 m").
        obs, evs = every_finding()
        for o in obs:
            out = render(o, evs, 'development')[0]
            for f, text in out.items():
                if not text or f == 'howToPractice':
                    continue
                low = text.lower()
                for word in ('median', 'standard deviation', '(est.)', 'quartile', 'p90', 'percentile'):
                    self.assertNotIn(word, low, f'{o["detector"]} {f}: {text}')

    def test_improvement_has_what_why_improve_practice_and_video(self):
        obs, evs = broadcast_mixed()
        gap = obs[0]
        out, facts, drill = render(gap, evs)
        self.assertEqual(drill['id'], 'defensive_compactness')
        for f in ('headline', 'whatHappened', 'whyItMatters', 'whatToImprove', 'howToPractice', 'video', 'priority'):
            self.assertTrue(out[f], f)
        self.assertIn("In 2.2 minutes of measured play with the team's centre in the final third, the distance "
                      "between the back line and midfield went beyond 22.5 m 3 times, at 01:01.20, 01:43.00 "
                      "and 02:31.40.", out['whatHappened'])          # the professional level is real metres
        self.assertIn('average positions of both units', out['whatHappened'])
        self.assertIn("with the team's centre in the final third", out['whatHappened'])
        self.assertIn('The largest was 24.8 m (est.) at 01:46.30.', out['whatHappened'])
        self.assertIn('Professional teams rarely showed more than 2 such spells over the same measured time.',
                      out['whatHappened'])
        self.assertIn('cut off by a gap in the measurement', out['whatHappened'])
        self.assertEqual(out['howToPractice'], 'Use this exercise: Compact shape small-sided game (8 to 18 '
                         'players, 15 to 20 minutes). Aim: Keep the distances between team-mates and between the '
                         'lines short, so the team defends as one compact group. See Training.')
        self.assertTrue(out['video'].startswith('Moments to watch: 01:04.80 (A-04, A-08, A-05 and others)'))
        self.assertIn('if the opponents had the ball', out['whatToImprove'])
        self.assertIn('This matters most when the opponents have the ball; the tracking does not record who had it, '
                      'so check the moments below.', out['whyItMatters'])

    def test_short_segment_never_claims_a_repeated_pattern(self):
        obs, evs = broadcast_mixed()
        gap = obs[0]                                        # 3 occurrences in 132 s: short
        for level in LEVELS:
            out = render(gap, evs, level)[0]
            text = ' '.join(v for v in out.values() if v)
            self.assertEqual(language.wording_problems(text, short=True, practice=True), [], level)
            self.assertIn('can vary by chance', out['whatHappened'])
        long_obs = every_finding()[0]
        mf = next(o for o in long_obs if o['detector'] == 'unit_gap_mf' and o['classification'] == 'improvement')
        out = render(mf)[0]
        self.assertIn('In 5.0 minutes of measured play, the distance between midfield and the front line went '
                      'beyond 24.5 m 5 times, at 00:10.00, 01:10.00, 02:10.00 and later; this is a repeated '
                      'pattern.', out['whatHappened'])
        self.assertIn('and later', out['whatHappened'])
        self.assertNotIn('can vary by chance', out['whatHappened'])
        capped = dict(gap, occurrenceCount=6)               # the record lists fewer occurrences than it counts
        self.assertIn('6 times, at 01:01.20, 01:43.00, 02:31.40 and later.', render(capped, evs)[0]['whatHappened'])

    def test_one_occurrence_is_one_observed_instance(self):
        obs = every_finding()[0]
        area = next(o for o in obs if o['pattern'] == 'team_area' and o['classification'] == 'improvement')
        out = render(area)[0]
        self.assertIn('once, from 00:05.00 to 00:25.00 (one observed instance).', out['whatHappened'])
        self.assertNotIn('times per measured minute', out['whatHappened'])
        self.assertIn('1650 square metres (est.)', out['whatHappened'])

    def test_strength_states_the_professional_median_and_how_to_build_on_it(self):
        for o in broadcast_strengths():
            out = render(o)[0]
            self.assertTrue(out['headline'].startswith('Maintained'), out['headline'])
            self.assertRegex(out['whatHappened'], r'\d')
            self.assertTrue(out['whatToImprove'].startswith('To build on it'), out['whatToImprove'])
            self.assertIsNone(out['howToPractice'])
            self.assertNotIn('upper level', out['headline']+out['whatHappened'])
        gap = render(broadcast_strengths()[0])[0]
        self.assertEqual(gap['headline'], 'Maintained a short distance between the back line and midfield: median '
                                          '8.0 m (est.), professional median 13.0 m')
        self.assertIn("Over 59 s of measured play with the team's centre in the middle third, Team A's distance from "
                      "the back line to midfield had a median of 8.0 m (est.), against a professional median of "
                      "13.0 m.", gap['whatHappened'])
        self.assertIn('It was beyond 16.1 m, a level professional teams exceed about a quarter of the time, for 0% '
                      'of the time.', gap['whatHappened'])
        self.assertEqual(render(broadcast_strengths()[0], level='development')[0]['headline'],
                         'Maintained a short distance between the defenders and the midfielders')
        line = render(broadcast_strengths()[1])[0]
        self.assertIn('in which the whole back line was measured with a back line of four', line['whatHappened'])
        dev = render(broadcast_strengths()[1], level='development')[0]['whatHappened']
        self.assertIn('typical difference in depth between the defenders was 0.9 m (estimated), against 3.2 m for a '
                      'typical professional team', dev)
        form = render(broadcast_strengths()[3])[0]
        self.assertIn('read as 4-4-2 (clear)', form['whatHappened'])
        self.assertIn('10 of 10 players kept one role at least 75% of the time.', form['whatHappened'])
        self.assertIn('The same players formed the back line throughout.', form['whatHappened'])
        self.assertIn("not been checked against a coach's view", form['whatHappened'])

    def test_a_pooled_strength_names_its_strictest_context(self):
        obs, evs = compact_block()
        width = obs[0]
        out = render(width, evs, team='B')[0]
        self.assertIn("Over 2.4 minutes of measured play, Team B's width had a median of 39.3 m (est.). The strictest "
                      "comparison is the moments with the team's centre in its own third (68 s): a median of 38.7 m "
                      "(est.) against a professional median of 40.3 m.", out['whatHappened'])
        self.assertIn('for 5.0% of those moments', out['whatHappened'])
        self.assertTrue(out['confidenceNote'].startswith('Medium confidence: the pitch registration'))

    def test_the_phase_hedge_is_on_every_compactness_and_spacing_finding(self):
        obs, evs = every_finding()
        for o in obs:
            if o['recommendationKey'] not in language.HEDGED_KEYS or o['classification'] not in ('strength',
                                                                                                 'improvement'):
                continue
            for level in LEVELS:
                why = render(o, evs, level)[0]['whyItMatters']
                self.assertIn('the tracking does not record who had', why.lower(), (o['detector'], level))
                self.assertIn('check the moments below' if o['evidence'] else 'check the phase on the video', why)

    def test_review_is_framed_as_a_question_to_review(self):
        obs, evs = broadcast_mixed()
        hl = next(o for o in obs if o['detector'] == 'line_height_and_gap')
        out = render(hl, evs)[0]
        self.assertTrue(out['headline'].startswith('To review:'))
        self.assertIn('The back line stood 55.2 m (est.) from the team\'s own goal while the distance to midfield '
                      'reached 24.0 m (est.).', out['whatHappened'])
        self.assertIn('This was one observed moment, from 01:36.00 to 01:44.00.', out['whatHappened'])
        self.assertTrue(out['howToPractice'].startswith('If the review confirms it'))
        form = next(o for o in obs if o['detector'] == 'formation_stability')
        out = render(form, evs)[0]
        self.assertIn('read most like 4-3-3, with 3-5-2 fitting almost as well (reading: ambiguous)',
                      out['whatHappened'])
        self.assertIn('5 of 10 players kept one role at least 75% of the time.', out['whatHappened'])
        self.assertIn('changed 8 times between complete readings, with 6 different groups', out['whatHappened'])
        self.assertTrue(out['confidenceNote'].startswith('Medium confidence: some player identities may have'))
        self.assertIn('unrelated formations', out['whyItMatters'])          # an ambiguity is not "changes"
        self.assertNotIn('frequent', out['whyItMatters'])
        change = next(o for o in every_finding()[0] if o['pattern'] == 'formation_change')
        out = render(change)[0]
        self.assertIn('read as 4-4-2 for 80 s and 3-5-2 for 70 s.', out['whatHappened'])
        self.assertIn('one the players did not plan', out['whyItMatters'])

    def test_lateral_and_isolation_wording(self):
        obs, evs = broadcast_mixed()
        lat = next(o for o in obs if o['detector'] == 'lateral_shift')
        out = render(lat, evs, 'development')[0]
        self.assertIn('the defenders moved sideways 1.6 s later than the midfielders on average',
                      out['whatHappened'])
        self.assertIn('rarely more than 1.2 s', out['whatHappened'])
        strength = next(o for o in every_finding()[0] if o['detector'] == 'lateral_shift' and
                        o['classification'] == 'strength')
        happened = render(strength)[0]['whatHappened']
        self.assertIn('moved sideways at about the same time as midfield', happened)   # -0.2 s: one sample step
        self.assertIn('Delays are measured in steps of 0.2 s.', happened)
        self.assertIn('the back line moves, at the median, at about the same time as its midfield', happened)
        iso = next(o for o in obs if o['detector'] == 'isolation')
        out = render(iso, evs)[0]
        self.assertEqual(out['headline'], 'To review: distance between A-11 and the nearest team-mate (role check)')
        self.assertIn('was in camera view', out['whatHappened'])
        self.assertIn('Professional players rarely showed more than 1 such spells', out['whatHappened'])
        self.assertIn('can be the role itself', out['whyItMatters'])
        self.assertTrue(out['whatToImprove'].startswith('To review'))

    def test_neutral_notes_say_their_state(self):
        obs, _ = broadcast_mixed()
        width = next(o for o in obs if o['classification'] == 'neutral' and o['detector'] == 'compactness')
        out = render(width)[0]
        self.assertEqual(set(out), {'note'})
        self.assertEqual(out['note'], 'Team width: median 55.1 m (est.) over 105 s, wider than the professional '
                                      'median of 44.0 m. No spell beyond 60.1 m. A wide shape is typical for a team '
                                      'with the ball; without it, it leaves space between players. Check the phase '
                                      'on the video.')
        spell = next(o for o in every_finding()[0] if o.get('state') == 'one_spell')
        self.assertIn('One spell beyond 60.3 m, from 00:11.00 to 00:29.40 (peak 62.3 m (est.)): one observed '
                      'instance', render(spell)[0]['note'])
        absent = observation('compactness', 'neutral', 'team_width', measured=59.6, state='absent',
                             metrics={'median': 39.1, 'professionalP50': 43.0, 'favourableSide': True,
                                      'strengthFailed': ['evidence_bound']}, base=baseline(60.3, 58.0))
        self.assertIn('narrower than the professional median of 43.0 m. No spell beyond 60.3 m. Too few independent '
                      'moments in this time to call it a strength.', render(absent)[0]['note'])
        under = dict(absent, state='under_measured', metrics=dict(absent['metrics'], strengthFailed=[
            'short_for_strength']))
        self.assertIn('Measured too briefly to call it a strength.', render(under)[0]['note'])
        several = dict(absent, state='within_range', occurrenceCount=2, occurrences=[
            occurrence(5.0, 12.0, 'width', 61.0), occurrence(30.0, 36.0, 'width', 60.9)])
        self.assertIn('2 spells beyond 60.3 m, within what professional teams showed over the same measured time.',
                      render(several)[0]['note'])
        one_team = dict(several, metrics=dict(several['metrics'], improvementBlocked='one_team_baseline'))
        self.assertIn('has one team only, so the count is not compared', render(one_team)[0]['note'])
        # a pooled strength that failed in one third names it (detectors.pooled_strength, metrics.contexts)
        fails = dict(absent, metrics=dict(absent['metrics'], strengthFailed=['context_fails'], contexts={
            'third:own': {'enoughForStrength': True, 'passesStrength': False},
            'third:middle': {'enoughForStrength': True, 'passesStrength': True},
            'third:final': {'enoughForStrength': False, 'passesStrength': False}}))
        self.assertIn("It did not meet the strength rule with the team's centre in its own third.",
                      render(fails)[0]['note'])
        unnamed = dict(absent, metrics=dict(absent['metrics'], strengthFailed=['context_fails']))
        self.assertIn('It did not meet the strength rule in every third of the pitch it was measured in.',
                      render(unnamed)[0]['note'])
        for o in (width, spell, absent, under, several, one_team):
            self.assertNotIn('no strength or improvement met the evidence rules', render(o)[0]['note'])
        ball = next(o for o in obs if o['detector'] == 'ball_context')
        note = render(ball)[0]['note']
        self.assertIn("own third 18%, middle third 45%, final third 37%", note)
        self.assertIn('not which team had it', note)

    def test_formation_notes_and_role_stability(self):
        obs = every_finding()[0]
        related = next(o for o in obs if o.get('state') == 'related_shapes')
        note = render(related)[0]['note']
        self.assertIn('read as 4-1-4-1, with 4-3-3 almost as close: both have the same back line and one holding '
                      'midfielder in front of it; they differ only in how high the other central midfielders and the '
                      'wide players stand. Read from the tracked positions over 35 s; not a finding, so no drill.',
                      note)
        self.assertIn('10 of 10 players kept one role at least 75% of the time.', note)
        short = next(o for o in obs if o.get('state') == 'under_measured' and o['detector'] == 'formation_stability')
        note = render(short)[0]['note']
        self.assertIn('read as 3-4-3 (likely) throughout the 50 s measured, from the tracked positions only.', note)
        self.assertIn('only called a strength over at least 5.0 minutes of measured play', note)
        for pair in language.RELATED_SHAPES:            # every family of team_shape.RELATED has its words
            self.assertTrue(language.RELATED_SHAPES[pair])
        from soccer import team_shape
        self.assertEqual(set(language.RELATED_SHAPES), set(team_shape.RELATED))

    def test_block_height_note(self):
        block = next(o for o in every_finding()[0] if o['detector'] == 'line_height_and_gap'
                     and o['classification'] == 'neutral')
        self.assertEqual(render(block)[0]['note'], "Block height: Team A's back line stood a median 30.6 m (est.) "
                                                   "from its own goal over 60 s (professional median 38.1 m); the "
                                                   "other team's back line (Team B) stood 50.1 m (est.) from its own "
                                                   "goal over 60 s.")
        low = dict(block, confidence=dict(block['confidence'], overall='low'))
        self.assertIn('treat these heights as rough', render(low)[0]['note'])

    def test_insufficient_lists_what_was_measured_and_its_own_reasons(self):
        for o in veo_insufficient():
            out, facts, _ = render(o, run_codes=('few_players', 'small_scene'))
            text = out['limitation']
            self.assertIn('not assessed', text)
            self.assertEqual(language.validate_text(text, facts), [])
            self.assertNotIn('mystery_code_x', text)
            self.assertNotIn('_', text)
            self.assertNotIn('only a few players', text)          # run-level: stated once in the data-quality note
        gap = render(veo_insufficient()[0], run_codes=('few_players',))[0]['limitation']
        self.assertEqual(gap, 'Back line to midfield distance: not assessed; nothing could be measured; the players '
                              'of the lines involved were not all in view together for part of the time.')
        width = render(veo_insufficient()[1], level='advanced')[0]['limitation']
        self.assertEqual(width, 'Horizontal compactness: not assessed; measured 0.6 s (2.0% of the time), longest '
                                'continuous stretch 0.6 s; not every outfield player was in view at the same time; '
                                'only a few players of each team were in view.')
        obs = amateur_set()[0]
        ball = next(o for o in obs if o['detector'] == 'ball_context')
        text = render(ball, run_codes=('keyless', 'small_scene'))[0]['limitation']
        self.assertEqual(text, "Ball location: not measured on this footage; the ball's location is only used on "
                               "broadcast-style footage, a fixed rule for this footage type rather than a check of "
                               "the ball tracking in this clip.")
        self.assertNotIn('not reliable', text)                    # the cause the first reports gave was never tested
        iso = next(o for o in obs if o['detector'] == 'isolation')
        self.assertIn('support distances are only measured on broadcast-style footage',
                      render(iso, run_codes=('keyless',))[0]['limitation'])
        low = observation('unit_gap_dm', 'insufficient', measured=101.0, eligible=.56, longest=29.0,
                          reasons=['medium_registration', 'stabilizer_flags', 'ambiguous_formation'],
                          insufficientReasons=['low_confidence', 'incomplete_units', 'not_calibrated'],
                          confidence='medium')
        low['confidence']['tracking'] = low['confidence']['overall'] = 'low'
        reasons = language.limitation_reasons(low)
        self.assertEqual(reasons[0], 'the units come from a formation reading that was ambiguous')   # decisive first
        self.assertLessEqual(len(reasons), language.MAX_LIMIT_REASONS)

    def test_missing_facts_drop_sentences_never_fill_defaults(self):
        obs, evs = broadcast_mixed()
        base = dict(obs[0]['baseline'], professionalRatePerMinute=None)
        gap = dict(obs[0], baseline=base, players=[])
        out, facts, _ = render(gap, evs)
        self.assertNotIn('times per measured minute', out['whatHappened'])
        self.assertNotIn('Players involved', out['whatHappened'])
        with self.assertRaises(language.NoWording):           # the first measured sentence has no threshold
            render(dict(obs[0], baseline=None))
        with self.assertRaises(language.NoWording):
            render(dict(obs[0], measuredSeconds=None))
        with self.assertRaises(language.NoWording):
            render(dict(obs[0], recommendationKey='press_higher'))
        with self.assertRaises(language.NoWording):
            render(dict(obs[0], classification='review'))       # no review wording for unit gaps
        plain = observation('compactness', 'strength', 'team_depth', measured=60.0, metrics={'median': 17.8},
                            base=baseline(46.5, 43.1))
        out = render(plain)[0]                                 # no professional median: no comparison is made up
        self.assertEqual(out['headline'], 'Maintained a short team from back to front')
        self.assertNotIn('professional', out['whatHappened'])

    def test_high_confidence_concern_is_said_only_with_its_label(self):
        obs, evs = broadcast_mixed()
        line = next(o for o in obs if o['detector'] == 'back_line_coordination' and o['classification'] ==
                    'improvement')
        hcc = render(line, evs, label=language.LABELS['high_confidence_concern'])[0]
        self.assertIn('high-confidence concern', hcc['whatHappened'])
        plain = render(line, evs, label=language.LABELS['potential_improvement'])[0]
        self.assertNotIn('high-confidence concern', plain['whatHappened'])

    def test_the_video_line_says_and_others(self):
        obs, evs = broadcast_mixed()
        out = render(obs[0], evs)[0]
        self.assertIn('(A-04, A-08, A-05 and others)', out['video'])      # the moment involves four players
        two = dict(evs[0], players=['A-04', 'A-08'])
        ctx = language.FindingContext(obs[0], (two,))
        self.assertEqual(language._video(ctx), 'Moments to watch: 01:04.80 (A-04, A-08).')

    def test_data_quality_is_stated_once_with_what_would_help(self):
        amateur = language.data_quality({'footage': 'amateur', 'smallScene': True, 'keyless': True},
                                        ('keyless', 'small_scene'), help_spell=True, vetoed=('isolation',
                                                                                             'ball_context'))
        self.assertEqual(len(amateur), 3)
        self.assertIn('support distances and ball location are only measured on broadcast-style footage', amateur[0])
        self.assertTrue(amateur[1].startswith('No pitch keyframes were set'))
        self.assertEqual(amateur[2], 'What would help: set 2-3 pitch keyframes on frames where the pitch lines are '
                                     'clearly visible, choose a spell where both lines of the team are in view or use '
                                     'a higher or wider camera view.')
        veo = language.data_quality({'footage': 'wide'}, ('few_players',), observed={'A': 2.0, 'B': 2.6})
        self.assertEqual(veo[0], 'Team analysis was not possible: on average only 2.0 players of Team A and 2.6 of '
                                 'Team B were in view, and the team measurements need every outfield player of a team '
                                 'in view at the same time. This is a camera-view limit, not a judgement on the team.')
        self.assertIn('higher or wider camera view', veo[1])
        self.assertEqual(language.data_quality({'footage': 'broadcast'}, (), help_spell=True),
                         ['To measure more: choose a spell where both lines of the team are in view.'])
        self.assertEqual(language.data_quality({'footage': 'broadcast'}, ()), [])

    def test_confidence_notes_leave_run_level_reasons_to_the_data_quality_note(self):
        gap = amateur_set()[0][0]
        note = render(gap)[0]['confidenceNote']                 # no data-quality note: every reason is named
        self.assertIn('no pitch keyframes were set', note)
        note = render(gap, run_codes=('keyless', 'small_scene'))[0]['confidenceNote']
        self.assertEqual(note, 'Medium confidence because of the footage limits in the data-quality note.')


class FactCheckTests(unittest.TestCase):
    def setUp(self):
        obs, evs = broadcast_mixed()
        self.obs = obs[0]
        self.facts = language.facts_for(self.obs, [e for e in evs if e['observation'] == self.obs['id']])

    def test_supported_statements_pass(self):
        ok = ('The gap reached 24.8 m (est.) at 01:46.30, 3 times in 132 s (2.2 minutes), 14% of the time, '
              'involving A-04 and A-08.')
        self.assertEqual(language.validate_text(ok, self.facts), [])
        self.assertEqual(language.validate_text('About 25 m and 24.80 m.', self.facts), [])   # shown precision

    def test_fabrications_are_caught(self):
        bad = {'a number': 'The gap reached 31.0 m.', 'a rounded-away number': 'The gap reached 24.7 m.',
               'a time': 'At 01:47.00 the gap opened.', 'a player': 'Watch A-09 here.',
               'a goalkeeper': 'GK-B stood deep.', 'a formation': 'They played 3-5-2.',
               'a percentage': 'For 40% of the time.', 'a number word': 'It happened seven times.',
               'a digit in a word': 'Above the p90 of play.'}
        for what, text in bad.items():
            self.assertTrue(language.validate_text(text, self.facts), what)

    def test_durations_inside_a_seconds_dict_are_facts_in_minutes(self):
        facts = language.facts_for({'metrics': {'labelSeconds': {'4-4-2': 150.0}}})
        self.assertEqual(language.validate_text('4-4-2 for 2.5 minutes', facts), [])

    def test_displays_are_accepted_only_when_given(self):
        resolved = language.resolve_players('Watch B-12 and B-09.', {'B-12': 'B-09', 'B-09': 'B-07'})
        self.assertEqual(resolved, 'Watch B-09 and B-07.')               # one pass: never chained
        resolved = language.resolve_players('Watch B-12.', {'B-12': 'B-09'})
        self.assertEqual(resolved, 'Watch B-09.')
        obs, evs = corrected_set()
        facts = language.facts_for(obs[0], evs)
        self.assertTrue(language.validate_text(resolved, facts))
        facts = language.facts_for(obs[0], evs, displays={'B-12': 'B-09'})
        self.assertEqual(language.validate_text(resolved, facts), [])
        self.assertEqual(language.resolve_players('A 4-4-2 shape.', {'B-12': 'B-09'}), 'A 4-4-2 shape.')
        self.assertEqual(language.resolve_players('B-05 left.', {}), 'B-05 left.')   # no label map: ids as they are

    def test_an_id_no_longer_in_the_roster_is_marked(self):
        # A BC60 copy with A-03 excluded: A-04 now shows as A-03, so a bare "A-03" in a stale report would name the
        # wrong player.
        labels = {'A-02': 'A-02', 'A-04': 'A-03', 'A-07': 'A-06'}
        self.assertEqual(language.resolve_players('00:36.20 (A-02, A-03, A-07)', labels),
                         '00:36.20 (A-02, not in the roster now (was A-03), A-06)')
        labels['A-03'] = 'excluded (was A-03)'                              # store.labels_of's own marking wins
        self.assertEqual(language.resolve_players('A-03 and A-04', labels), 'excluded (was A-03) and A-03')

    def test_literal_names_are_skipped(self):
        facts = language.facts_for(self.obs, extra={'teamName': 'U12 Lions'})
        self.assertEqual(language.validate_text('U12 Lions kept 3 times.', facts), [])

    def test_wording_check(self):
        self.assertTrue(language.wording_problems('A terrible mistake by the back line.'))
        self.assertTrue(language.wording_problems('They chose to sit deep.'))
        self.assertTrue(language.wording_problems('The team was pressing high.'))
        self.assertTrue(language.wording_problems('Lost possession twice.'))
        self.assertEqual(language.wording_problems('Pass the ball.', practice=True), [])
        self.assertTrue(language.wording_problems('A recurring gap.', short=True))
        self.assertEqual(language.wording_problems('A recurring gap.'), [])
        self.assertEqual(language.wording_problems('Pressure on the ball; passive defenders.'), [])
        self.assertTrue(language.wording_problems('Caf\u00e9 shape'))


class FormatTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(language.fmt_time(42.4333), '00:42.43')
        self.assertEqual(language.fmt_time(59.999), '01:00.00')
        self.assertEqual(language.fmt_time(222.4), '03:42.40')
        self.assertEqual(language.fmt_duration(5.44), '5.4 s')
        self.assertEqual(language.fmt_duration(57.4), '57 s')
        self.assertEqual(language.fmt_duration(180.0), '3.0 minutes')
        self.assertEqual(language.fmt_percent(.957), '96%')
        self.assertEqual(language.fmt_percent(.034), '3.4%')
        self.assertEqual(language.fmt_percent(0.0), '0%')
        self.assertEqual(language.fmt_rate(.37), '0.37')
        self.assertEqual(language.fmt_rate(3.14), '3.1')
        self.assertEqual(language.fmt_value(21.43, 'm'), '21.4 m (est.)')
        self.assertEqual(language.fmt_value(21.43, 'm', estimated=False), '21.4 m')
        self.assertEqual(language.fmt_value(21.43, 'm', plain=True), '21.4 m (estimated)')
        self.assertEqual(language.team_full('A'), 'Team A (as labelled in this run)')
        self.assertEqual(language.team_full('A', 'Lions'), 'Lions')

    def test_engine_protocol(self):
        self.assertIsInstance(language.TemplateEngine(), language.TextEngine)


if __name__ == '__main__':
    unittest.main()
