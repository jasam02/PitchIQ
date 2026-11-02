"""The situation detectors. Each module has detect(engine) -> [episode], an episode being a dict:

    detector, kind          what it is about ('wide_cover': 'exposed_channel' / 'rotated_to_cover' / ...)
    team, category          'A' / 'B'; 'positive' / 'improvement' / 'neutral' / 'insufficient'
    t0, t1, tKey, sKey      the situation's span (s) and its key time / sample (the moment a coach would pause on)
    before, after           seconds of lead-up and follow-up to show (the moment player's window)
    players                 {'focus': [pid], 'support': [pid], 'opponents': [pid]}  (person indices of MatchData)
    roles                   {pid: {'role', 'confidence'}} as inferred at the key time
    measures                the numbers behind the call (metres, seconds), at the key time and over the episode
    zones                   {'channel': pitch polygon, 'coverPoint': (x, y), ...} in metres
    phase                   {'state': 'own' / 'opp' / 'unknown', 'settled': bool, 'confidence'} at the key time (estimated)
    outcome                 what was observed in the follow-up (never used to judge the decision)
    reliability             'high' / 'medium' / 'low': how well the people, positions and area were observed
    interpretation          'clear' / 'plausible' / 'uncertain': how certain the tactical reading is
    reasons                 short codes for what lowered either of them
    score                   ordinal importance used for ranking (not a probability)

    module          detector               kinds (category)
    wide_cover      1 wide defender high   exposed_channel (improvement), covered_advance / rotated_to_cover (positive)
    pressure        2 pressure and cover   press_and_cover (positive), double_press_free / press_no_cover (improvement)
    recovery        3 after losing it      recovery_run (positive), slow_reorganisation (improvement)
    runs            4 off-ball movement    run_in_behind / created_separation (positive)
    support         5 the ball carrier     offered_support (positive), carrier_isolated (improvement, a review)
    back_line       6 / 7a the defence     stepped_out_open / gap_attacker / dropped_off (improvement),
                                           stepped_out_covered / stepped_up_together (positive)
    between_lines   6b                     free_between_lines (improvement)
"""
from . import back_line, between_lines, pressure, recovery, runs, support, wide_cover

DETECTORS = {'wide_cover': wide_cover, 'pressure': pressure, 'recovery': recovery, 'runs': runs, 'support': support,
             'back_line': back_line, 'between_lines': between_lines}


def detect_all(engine, names=None):
    out = []
    for name, module in DETECTORS.items():
        if names and name not in names:
            continue
        out.extend(module.detect(engine))
    return out
