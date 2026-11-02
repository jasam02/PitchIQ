"""The coach's words for a moment, a pattern, the summary: built from the moment's own measurements, never from a
statistic. Three registers by coaching level (soccer/coaching profiles, levels 1-7):

    development   levels 1-3   short, plain words; level 1 no numbers, level 2 no metres
    intermediate  levels 4-5   roles, distances in metres, support and cover
    advanced      levels 6-7   unit and structure words (rest defence, rotation, line height), the trade-offs
The SAME observed event is described at every level; only the interpretation and the suggestion change.

People are named by their match label (A-34) and, when known, their inferred role ("left back A-34"); the page shows the
label the roster has now. Possession is an estimate: sentences about who had the ball say "we had the ball" / "they had
the ball" only when the moment's phase was settled, and the reliability line names the estimate.

Nothing here says what a player thought or how hard they tried: only what was observed.
"""
from . import roles as rl

BANDS = {1: 'development', 2: 'development', 3: 'development', 4: 'intermediate', 5: 'intermediate', 6: 'advanced',
         7: 'advanced'}

TITLES = {
    'rotated_to_cover': 'Covered the space behind an advancing full back',
    'covered_advance': 'Full back pushed on with the space behind covered',
    'exposed_channel': 'Space left behind the advancing full back',
    'stepped_out_open': 'Stepped out of the back line, nobody filled the gap',
    'stepped_out_covered': 'Stepped out to engage with the line covering behind',
    'gap_attacker': 'Gap in the back line with an attacker in it',
    'dropped_off': 'Deeper than the back line with a runner in front',
    'stepped_up_together': 'The back line stepped up together',
    'free_between_lines': 'Opponent free between defence and midfield',
    'press_and_cover': 'Pressure on the ball with cover behind',
    'double_press_free': 'Two players pressing one, an opponent left free',
    'press_no_cover': 'Pressure on the ball without cover',
    'recovery_run': 'Recovery run after losing the ball',
    'slow_reorganisation': 'Players stayed ahead of the ball after the loss',
    'run_in_behind': 'Run in behind the defence',
    'created_separation': 'Movement that lost the marker',
    'offered_support': 'Offered an angle to a teammate under pressure',
    'carrier_isolated': 'Ball carrier under pressure without a nearby option',
}
PATTERN_TITLES = {
    'rotated_to_cover': 'Covering behind the advancing full backs',
    'covered_advance': 'Full backs pushing on with cover behind',
    'exposed_channel': 'Space behind the advancing full backs',
    'stepped_out_open': 'Defenders stepping out without the line adjusting',
    'stepped_out_covered': 'Stepping out to engage with cover',
    'gap_attacker': 'Gaps in the back line with an attacker in them',
    'dropped_off': 'A defender deeper than the line with runners in front',
    'stepped_up_together': 'The back line stepping up as a unit',
    'free_between_lines': 'Opponents free between defence and midfield',
    'press_and_cover': 'Pressure with cover',
    'double_press_free': 'Two pressing one, leaving an opponent free',
    'press_no_cover': 'Pressure without cover',
    'recovery_run': 'Recovery runs after losing the ball',
    'slow_reorganisation': 'Slow to get goal-side after a loss',
    'run_in_behind': 'Runs in behind the defence',
    'created_separation': 'Movement to lose markers',
    'offered_support': 'Giving the ball carrier an angle',
    'carrier_isolated': 'Ball carriers left without a nearby option',
}
POINTERS = {
    'rotated_to_cover': {'development': 'When a teammate goes forward, someone fills the space they left.',
                         'intermediate': 'Full back goes, the nearest midfielder or centre-back tucks into the channel.',
                         'advanced': 'Rest defence: rotate into the vacated channel as the full back goes, not after the ball is lost.'},
    'covered_advance': {'development': 'Going forward is fine when a teammate is ready behind you.',
                        'intermediate': 'Advance when the channel behind you is covered; check before the run.',
                        'advanced': 'Keep the cover behind the full back in place before the run.'},
    'exposed_channel': {'development': 'If you go forward, check that a teammate is ready behind you.',
                        'intermediate': 'Full back advances only when the holding midfielder or nearest centre-back can cover the channel.',
                        'advanced': 'One covering player for each advanced full back, set before the run: rest defence first.'},
    'stepped_out_open': {'development': 'If one defender goes to the ball, the others move across.',
                         'intermediate': 'Step out, neighbour tucks in: never leave the gap open behind the step.',
                         'advanced': 'Press from the back line only with a cover trigger: the neighbour shifts or the 6 drops in.'},
    'stepped_out_covered': {'development': 'Go to the ball while a teammate helps behind you.',
                            'intermediate': 'Step out with the neighbour tucking in behind: keep doing it.',
                            'advanced': 'Correct trigger and cover: the line kept its shape behind the step.'},
    'gap_attacker': {'development': 'Stay close to the defender next to you when an attacker is near.',
                     'intermediate': 'Keep the distances between defenders short when an attacker is near: the nearest defender narrows.',
                     'advanced': 'Back-line spacing follows the attacker, not the touchline: close the channel the striker stands in.'},
    'dropped_off': {'development': 'Defenders stay in a line together.',
                    'intermediate': 'Step up with the line unless you are marking a runner.',
                    'advanced': 'Hold the line height unless you are tracking a runner; one deep defender plays everyone onside.'},
    'stepped_up_together': {'development': 'Move up together, move back together.',
                            'intermediate': 'Up together, back together: the line keeps its shape as it moves.',
                            'advanced': 'Coordinated line height on the backward pass: the block stayed compact.'},
    'free_between_lines': {'development': 'Watch the player behind you, not only the ball.',
                           'intermediate': 'Keep the lines close; the nearest player goes to the one between the lines.',
                           'advanced': 'Compact the vertical distance and pass on the receiver between the lines.'},
    'press_and_cover': {'development': 'One goes to the ball, one helps behind.',
                        'intermediate': 'First defender presses, second defender covers behind and inside.',
                        'advanced': 'Pressure and cover in the right order: the press was supported.'},
    'double_press_free': {'development': 'One goes to the ball, the other helps behind.',
                          'intermediate': 'One presses, one covers: two on the ball leaves someone free.',
                          'advanced': 'Pressing roles: one presses, the next covers the nearest passing option.'},
    'press_no_cover': {'development': 'When a teammate goes to the ball, help behind them.',
                       'intermediate': 'Second defender: behind and inside the first, on the way to goal.',
                       'advanced': 'No press without a cover player on the line to goal.'},
    'recovery_run': {'development': 'Lose it, run back inside.',
                     'intermediate': 'On a loss, recover inside and goal-side of the ball.',
                     'advanced': 'Transition: recovery runs to the central lane first.'},
    'slow_reorganisation': {'development': 'Lose the ball, get back behind it.',
                            'intermediate': 'On a loss the nearest player presses; everyone else gets goal-side quickly.',
                            'advanced': 'Counter-press or recover: the players ahead of the ball pick one immediately.'},
    'run_in_behind': {'development': 'Run into the space behind the defenders.',
                      'intermediate': 'Time the run off the last defender\u2019s shoulder.',
                      'advanced': 'Depth runs pin the back line and open the space in front of it.'},
    'created_separation': {'development': 'Move away from your marker to get free.',
                           'intermediate': 'Check away, then move: separation creates the option.',
                           'advanced': 'Movement to separate from the marker created a receiving option.'},
    'offered_support': {'development': 'Help the player with the ball: be close and free.',
                        'intermediate': 'Give the pressed carrier an angle: close enough, and not behind an opponent.',
                        'advanced': 'Support angle under pressure: the lane was opened at the right moment.'},
    'carrier_isolated': {'development': 'Help the player with the ball: be close and free.',
                         'intermediate': 'When a teammate is pressed, give them an angle within 15-20 m.',
                         'advanced': 'Support structure around the ball: two options on different lines for a pressed carrier.'},
}
WHY = {
    'rotated_to_cover': {'own': 'Had the ball been lost there, that channel is where a counter-attack goes first; it was covered.',
                         'opp': '{opp} was near that space; with the cover in place a ball into the channel would have met a defender.',
                         'unknown': 'The space behind the advanced full back stayed protected.'},
    'covered_advance': {'own': 'The team kept its balance on that side while the full back joined the attack.',
                        'opp': 'The space behind the full back was protected while they engaged high up.',
                        'unknown': 'The space behind the full back stayed protected.'},
    'exposed_channel': {'own': 'Had the ball been lost there, {opp} had a free run into that space.',
                        'opp': 'With the opponents on the ball, that channel was open for a pass in behind.',
                        'unknown': 'Whoever had the ball, that channel was open (who had it could not be confirmed).'},
}
REASON_WORDS = {
    'phase_unknown': 'who had the ball was not named at that moment',
    'phase_unsettled': 'it was close to a change of possession',
    'role_medium': 'the player\u2019s role was inferred with medium confidence',
    'possession_low_confidence': 'the possession estimate had low confidence there',
    'ball_from_carrier': 'the ball\u2019s position comes from the named carrier',
    'ball_not_placed': 'the ball was not placed on the pitch at the key moment',
    'change_low_confidence': 'the change of possession had low confidence',
    'ball_from_player': 'the ball\u2019s position comes from a player',
    'identity_medium': 'one identity was medium confidence',
    'identity_low': 'one identity was low confidence',
    'presence_medium': 'a player was not visible for part of the moment',
    'presence_low': 'a player was out of view for much of the moment',
    'calibration_medium': 'the pitch registration was medium confidence',
    'calibration_low': 'the pitch registration was weak',
    'view_medium': 'part of the area was out of the camera\u2019s view',
    'view_low': 'much of the area was out of the camera\u2019s view',
    'position_medium': 'positions were less precise than usual',
    'position_low': 'positions were imprecise',
    'person_not_seen_at_key': 'a player was not seen at the key moment',
}
SUGGEST = {
    'rotate_cover': '{mover} could have moved into the channel behind {fb} (the green area) as {fb} went forward: about {reach} s away.',
    'hold_deeper': '{fb} could have held a deeper position until a teammate covered the channel.',
    'shift_across': '{mover} could have shifted across towards the channel, with the other centre-back tucking in.',
    'neighbour_narrows': '{mover} could have closed into the space {fb} left in the line (the green area).',
    'midfielder_drops': '{mover} could have dropped into the space (the green area) to hold the shape.',
    'hold_the_line': '{fb} could have held the line and let the midfield press the ball.',
    'narrow_the_gap': '{mover} could have narrowed towards the gap (the green area).',
    'step_up_level': '{mover} could have stepped up level with the line.',
    'defender_steps_in': '{mover} could have stepped in to the player between the lines, with the line covering.',
    'second_presser_covers': '{mover} could have dropped off the ball to cover {free}.',
    'cover_the_presser': '{mover} could have taken a covering position about 7 m behind the presser (the green area).',
    'recover_central': '{mover} could have recovered into the central lane goal-side of the ball (the green area).',
    'offer_an_angle': '{mover} could have moved into the green area, where the passing line was open.',
}


def cap(text):
    """First letter upper case, the rest untouched (labels such as A-34 keep their case)."""
    return text[:1].upper()+text[1:] if text else text


def band(level):
    return BANDS.get(int(level or 5), 'intermediate')


def clock(t):
    t = max(0.0, float(t))
    m = int(t//60)
    return f'{m:02d}:{int(t-60*m):02d}'


def metres(v, level):
    return '' if v is None or int(level) <= 2 else f'{abs(v):.0f} m'


class Names:
    """Player and team words for one report: labels, roles, kit colours."""

    def __init__(self, data, team, kits, roles):
        self.data, self.team, self.kits, self.roles = data, team, kits or {}, roles or {}

    def team_word(self, team=None):
        t = team or self.team
        kit = self.kits.get(t)
        return f'Team {t} ({kit})' if kit else f'Team {t}'

    def person(self, pid, with_role=True):
        if pid is None or pid < 0:
            return 'a player'
        p = self.data.people[int(pid)]
        label = p.get('display') or p['id']
        r = self.roles.get(int(pid))
        if p.get('team') != self.team:
            kit = self.kits.get(p.get('team'))
            role = rl.role_word(r['role']) if with_role and r else None
            return f"opponent {label}" + (f" ({role})" if role and role != 'player' else '') if not kit else \
                f"{kit} {label}" + (f" ({role})" if role and role != 'player' else '')
        if with_role and r:
            return f"{rl.role_word(r['role'])} {label}"
        return label


def phase_key(ep):
    ph = ep.get('phase') or {}
    return ph.get('state') if ph.get('settled') and ph.get('state') in ('own', 'opp') else 'unknown'


def reliability_text(ep):
    obs = {'high': 'Players and positions well observed', 'medium': 'Players and positions observed with some uncertainty',
           'low': 'Players or positions poorly observed'}[ep['reliability']]
    reasons = [REASON_WORDS[r] for r in ep.get('reasons') or [] if r in REASON_WORDS]
    interp = {'clear': 'clear reading', 'plausible': 'plausible reading', 'uncertain': 'uncertain reading'}[ep['interpretation']]
    tail = f" ({'; '.join(reasons[:3])})" if reasons else ''
    possession = ' Who had the ball is PitchIQ\u2019s estimate from the tracking.' if ep.get('phase', {}).get('state') in ('own', 'opp') else ''
    return f'{obs}; {interp}{tail}.{possession}'


def moment_text(ep, names, level, suggestion=None, alternatives=()):
    """{'title', 'finding', 'what', 'why', 'assessment', 'suggestion', 'alternatives', 'pointer', 'reliability'}."""
    kind = ep['kind']
    b = band(level)
    m = ep.get('measures') or {}
    P = ep['players']
    f = [names.person(p) for p in P.get('focus', [])]
    s = [names.person(p) for p in P.get('support', [])]
    o = [names.person(p) for p in P.get('opponents', [])]
    t = clock(ep['tKey'])
    ph = phase_key(ep)
    lv = int(level)
    num = lv >= 3
    out = {'title': TITLES.get(kind, kind.replace('_', ' ').capitalize())}
    fb = f[0] if f else 'A player'
    opp = o[0] if o else 'an opponent'
    if kind in ('rotated_to_cover', 'covered_advance'):
        cov = s[0] if s else 'a teammate'
        side = m.get('side', '')
        adv = f" ({metres(m.get('advanceM'), lv)} higher than the centre-backs)" if num else ''
        moved = f", moving {metres(m.get('covererClosedM'), lv)} towards it" if kind == 'rotated_to_cover' and num and m.get('covererClosedM') else ''
        out['what'] = (f"At {t}, {fb} pushed up on the {side}{adv}. {cap(cov)} covered the space behind{moved}"
                       + (f", with {opp} {metres(m.get('opponentToSpaceM'), lv)} from it." if num and o else '.'))
        out['finding'] = (f"{cap(cov)} {'moved across to protect' if kind == 'rotated_to_cover' else 'was in place to protect'}"
                          f" the space behind {fb}.")
        out['why'] = WHY[kind][ph].format(opp=opp)
        out['assessment'] = 'Done well: the run forward was supported, so the team kept its balance on that side.'
    elif kind == 'exposed_channel':
        side = m.get('side', '')
        adv = f" {metres(m.get('advanceM'), lv)} higher than the centre-backs" if num else ' high up'
        margin = m.get('timeMarginS')
        reach = (f" and would have got there about {margin:.1f} s before any teammate" if num and margin else '')
        out['what'] = (f"At {t}, {fb} was{adv} on the {side}. The nearest opponent, {opp}, was "
                       f"{metres(m.get('opponentToSpaceM'), lv) + ' ' if num else ''}from the space behind{reach}.")
        out['finding'] = f"Nobody covered the {side} channel behind {fb} while {opp} was close to it."
        out['why'] = WHY[kind][ph].format(opp=opp)
        out['assessment'] = 'The forward run can be useful; what was missing was a teammate covering behind it.'
    elif kind in ('stepped_out_open', 'stepped_out_covered'):
        step = f" {metres(m.get('stepM'), lv)}" if num else ''
        if kind == 'stepped_out_open':
            out['what'] = (f"At {t}, {fb} stepped{step} out of the back line towards the ball. Nobody filled the space left in "
                           f"the line, and {opp} was {metres(m.get('opponentToPointM'), lv) + ' ' if num else ''}from it.")
            out['finding'] = f"{cap(fb)} stepped out and the line did not adjust behind."
            out['why'] = 'Stepping out is right when someone covers; here the space behind the step was the shortest way to goal.'
            out['assessment'] = 'The step towards the ball was positive; the line did not move to cover it.'
        else:
            helper = s[0] if s else 'a teammate'
            out['what'] = f"At {t}, {fb} stepped{step} out to engage {opp}, and {helper} closed the space behind."
            out['finding'] = f"{cap(fb)} stepped out to engage with {helper} covering behind."
            out['why'] = 'The pressure came from the back line without opening a gap in it.'
            out['assessment'] = 'Done well: step out with cover is exactly how the line should work.'
    elif kind == 'gap_attacker':
        pair = ' and '.join(f[:2]) if len(f) >= 2 else fb
        gap = f" were {metres(m.get('gapM'), lv)} apart" if num else ' were far apart'
        out['what'] = (f"At {t}, {pair}{gap}, and {opp} stood unmarked in that gap"
                       + (f", with the ball {metres(m.get('ballToPointM'), lv)} away." if num and m.get('ballToPointM') else '.'))
        out['finding'] = f"{cap(opp)} was unmarked in a gap between {pair}."
        out['why'] = f"A pass through that gap would have put {opp} through between the defenders."
        out['assessment'] = 'The distances in the back line did not follow the attacker.'
    elif kind == 'dropped_off':
        drop = f" {metres(m.get('dropM'), lv)}" if num else ''
        out['what'] = (f"At {t}, {fb} was{drop} deeper than the rest of the back line while {opp}, unmarked and running "
                       f"forward, was between them and the line.")
        out['finding'] = f"{cap(fb)} was deeper than the line with a runner in front."
        out['why'] = 'One deep defender plays every attacker in front of them onside.'
        out['assessment'] = ('Worth a look rather than a verdict: PitchIQ cannot see the moment of a pass, so the offside '
                             'line itself is not judged.')
    elif kind == 'stepped_up_together':
        rise = f" {metres(m.get('riseM'), lv)}" if num else ''
        n = m.get('defenders')
        out['what'] = (f"At {t}, the back line moved up{rise} together"
                       + (f" ({n} defenders, level with each other)" if lv >= 2 and n else '') + '.')
        out['finding'] = 'The back line stepped up as one unit.'
        out['why'] = 'Stepping up together shrinks the space between the lines and leaves attackers behind the line.'
        out['assessment'] = 'Done well: the line moved as a unit and stayed level.'
    elif kind == 'free_between_lines':
        mid = f[0] if f else 'the nearest midfielder'
        defn = s[0] if s else 'the nearest defender'
        out['what'] = (f"At {t}, {opp} stood unmarked between the midfield and the defence"
                       + (f" ({metres(m.get('freeM'), lv)} from the nearest player; the lines were {metres(m.get('linesGapM'), lv)} apart)" if num else '')
                       + ' while the ball was in front of the midfield.')
        out['finding'] = f"{cap(opp)} was free between the lines, between {mid} and {defn}."
        out['why'] = 'A pass into that pocket lets the receiver turn and face the back line.'
        out['assessment'] = 'The distance between the lines was too big to control the player in it.'
    elif kind == 'press_and_cover':
        cov = s[0] if s else 'a teammate'
        out['what'] = (f"At {t}, {fb} pressed {opp}" + (f" ({metres(m.get('pressDistanceM'), lv)})" if num else '')
                       + f", and {cov} covered behind on the way to goal"
                       + (f" ({metres(m.get('coverDistanceM'), lv)} from the ball)." if num and m.get('coverDistanceM') else '.'))
        out['finding'] = f"{cap(fb)} pressed the ball with {cov} covering behind."
        out['why'] = 'The ball carrier had pressure in front and a second defender behind: no easy way forward.'
        out['assessment'] = 'Done well: pressure and cover in the right order.'
    elif kind == 'double_press_free':
        pair = ' and '.join(f[:2]) if len(f) >= 2 else fb
        free = o[1] if len(o) > 1 else 'another opponent'
        out['what'] = f"At {t}, {pair} both pressed {opp} while {free} was free nearby."
        out['finding'] = f"Two players pressed {opp}, leaving {free} free."
        out['why'] = f"One pass to {free} would have beaten both pressers."
        out['assessment'] = 'The pressure was good; the second player was needed as cover, not as a second presser.'
    elif kind == 'press_no_cover':
        out['what'] = (f"At {t}, {fb} pressed {opp}" + (f" ({metres(m.get('pressDistanceM'), lv)})" if num else '')
                       + ' with nobody behind on the way to goal.')
        out['finding'] = f"{cap(fb)} pressed without cover behind."
        out['why'] = 'If the carrier gets past the presser, the way to goal is open.'
        out['assessment'] = 'The pressure itself was right; the cover behind it was missing.'
    elif kind == 'recovery_run':
        back = f" {metres(m.get('backM'), lv)}" if num else ''
        out['what'] = (f"Just after the ball was lost (at {clock(ep['t0'])}), {fb} ran{back} back towards goal"
                       + (f" ({m.get('sprintS')} s at sprinting speed)" if lv >= 4 and m.get('sprintS') else '')
                       + ' and ended goal-side of the ball in the middle.')
        out['finding'] = f"{cap(fb)} recovered inside and goal-side straight after the loss."
        out['why'] = 'Recovering inside, goal-side of the ball, protects the most dangerous route to goal.'
        out['assessment'] = 'Done well: the recovery run closed the central route.'
    elif kind == 'slow_reorganisation':
        n = m.get('aheadOfBall')
        out['what'] = (f"After the ball was lost (at {clock(ep['t0'])}), several players who were ahead of the ball moved back "
                       f"only a few metres in the next 6 s, while {m.get('opponentsAdvancing')} opponents ran forward"
                       + (f"; {m.get('goalsideAtEnd')} outfield players were goal-side of the ball after it." if lv >= 2 else '.'))
        out['finding'] = 'After the loss the team was slow to get players goal-side of the ball.'
        out['why'] = 'Players left ahead of the ball cannot defend the space the opponents run into.'
        out['assessment'] = 'Movement only: PitchIQ sees where players went, not why.'
    elif kind in ('run_in_behind', 'created_separation'):
        if kind == 'run_in_behind':
            out['what'] = (f"At {t}, {fb} ran" + (f" {metres(m.get('runM'), lv)}" if num else '')
                           + " forward and got beyond the opponents' last defender"
                           + (f" ({metres(m.get('beyondLastM'), lv)})" if num else '') + '.')
            out['finding'] = f"{cap(fb)} made a run in behind the opponents' defence."
            out['why'] = 'Runs in behind stretch the back line and create a passing option.'
        else:
            out['what'] = (f"At {t}, {fb} moved away from {opp}"
                           + (f": {metres(m.get('separationStartM'), lv)} apart at the start, {metres(m.get('separationEndM'), lv)} at the end." if num else '.'))
            out['finding'] = f"{cap(fb)} lost their marker with the movement."
            out['why'] = 'Separation from the marker turns a teammate into a real option.'
        if m.get('laneOpen'):
            out['what'] += ' From where the ball was, the passing line to them was open.'
        out['assessment'] = 'Done well: the movement created an option (whether the pass was on depends on things the camera cannot show).'
    elif kind == 'offered_support':
        carrier = s[0] if s else 'the ball carrier'
        out['what'] = (f"At {t}, with {carrier} under pressure, {fb} moved" + (f" {metres(m.get('movedM'), lv)}" if num else '')
                       + ' into an open passing lane within reach.')
        out['finding'] = f"{cap(fb)} gave {carrier} an angle under pressure."
        out['why'] = 'A pressed carrier with a close, open option can keep the ball.'
        out['assessment'] = 'Done well: the angle appeared when it was needed.'
    elif kind == 'carrier_isolated':
        out['what'] = (f"At {t}, {fb} had {opp} within a few metres and "
                       + (f"{m.get('optionsNear')} open passing option{'s' if m.get('optionsNear') != 1 else ''} within 20 m." if lv >= 2 else 'no easy pass nearby.'))
        out['finding'] = f"{cap(fb)} was pressed with no close option."
        out['why'] = 'Without an angle, the carrier has to beat the opponent or play long.'
        out['assessment'] = 'Worth a review: sometimes this is the plan (a long ball), sometimes support was late.'
    else:
        out.update(what=f'At {t}.', finding=out['title'], why='', assessment='')
    out['pointer'] = POINTERS.get(kind, {}).get(b, '')
    out['suggestion'] = None
    if suggestion:
        mv = names.person(suggestion['mover'])
        free = o[1] if len(o) > 1 else opp
        text = SUGGEST.get(suggestion['label'], '{mover} could have moved into the green area.')
        # cap() the whole sentence: some start with the advanced player ('{fb} could have held ...')
        out['suggestion'] = cap(text.format(mover=cap(mv), fb=fb, opp=opp, free=free,
                                            reach=f"{suggestion['reachS']:.1f}" if suggestion.get('reachS') is not None else '?'))
        out['suggestion'] += ' (Assuming everyone else stayed where they were.)' if lv >= 4 else ''
    out['alternatives'] = [cap(SUGGEST.get(a['label'], '').format(mover=cap(names.person(a['mover'])), fb=fb, opp=opp,
                                                                  free=o[1] if len(o) > 1 else opp,
                                                                  reach=f"{a['reachS']:.1f}" if a.get('reachS') is not None else '?'))
                           for a in alternatives]
    out['reliability'] = reliability_text(ep)
    out['after'] = after_text(kind, ep.get('outcome') or {}, f, s, o, lv)
    return out


def after_text(kind, outc, f, s, o, level):
    """What was observed in the seconds after (the possession record and the tracked ball), one sentence or None. It
    never changes the reading of the decision: a good decision can end badly and a poor one go unpunished."""
    opp = o[0] if o else 'the opponent'
    if kind == 'exposed_channel' and outc.get('opponentReachedSpace') is not None:
        return (f'A few seconds later {opp} did get into that space.' if outc['opponentReachedSpace']
                else 'Nothing came of it this time; the decision is judged, not the outcome.')
    if kind in ('free_between_lines', 'gap_attacker', 'dropped_off') and outc.get('opponentReceived') is not None:
        return (f'Within a few seconds the ball reached {opp}.' if outc['opponentReceived']
                else f'The ball did not reach {opp} this time; the space was still there.')
    if kind == 'double_press_free' and outc.get('opponentReceived') is not None:
        free = o[1] if len(o) > 1 else 'the free player'
        return f'The ball did go to {free}.' if outc['opponentReceived'] else f'The ball did not go to {free} this time.'
    if kind in ('press_and_cover', 'press_no_cover', 'stepped_out_covered', 'stepped_out_open') and outc.get('wonBack') is not None:
        return ('The team won the ball back within a few seconds (as PitchIQ estimates possession).' if outc['wonBack']
                else 'The opponents kept the ball.')
    if kind in ('offered_support', 'run_in_behind', 'created_separation') and outc.get('received') is not None:
        who = f[0] if f else 'the player'
        return f'The ball reached {who} within a few seconds.' if outc['received'] else f'The ball did not go to {who} this time.'
    if kind == 'carrier_isolated' and outc.get('lost') is not None:
        return 'The ball was lost within a few seconds.' if outc['lost'] else 'The team kept the ball.'
    if kind in ('rotated_to_cover', 'covered_advance') and outc.get('lost') is not None:
        return ('The ball was lost soon after: the cover was there when it mattered.' if outc['lost']
                else 'The team kept the ball; the cover stayed in place.')
    return None


def pattern_text(kind, team_word, count, halves, level):
    title = PATTERN_TITLES.get(kind, kind)
    first, second = halves
    split = f" ({first} in the first half, {second} in the second)" if first and second else ''
    if int(level) <= 1:
        return title, f"This happened several times{split.replace(' (', ' (').strip() and ''}."
    return title, f"{count} times{split}."
