"""DEPRECATED (coaching v2, DESIGN v2 11.2): the v1 report, moved here unchanged from report.py on Oct 22
(TEXT wave 5). report.py now builds the v2 report (build_report_v2, the seven sections of DESIGN v2 9.2); this
module serves the v1 report until the server switches (wave 7), and report.py forwards its names (build_report,
SECTIONS, MAX_PRIORITIES, to_json, ...) so old callers keep working.

The original module notes follow.

The coaching report of one team over one analysed range, assembled from the detectors' observations, their
evidence and the drill library: overview, summary, strengths, potential improvements, moments to review, a breakdown
by area, at most three priorities, linked training and the confidence notes.

What the assembly guarantees, because a smaller report of trustworthy observations beats a fuller one with wrong
advice:
- nothing is forced. A report shows the strengths, improvements and reviews the detectors classified, and only those;
  an empty list stays empty (a strength-only or limitations-only report is a valid report), and no section is filled
  with generic text: a breakdown section without a measured observation is marked unavailable with its reason.
  Attacking structure is unavailable in v1 by construction: the result has no possession record, so the team's
  attacking shape cannot be told from its defending shape (team_shape's formation.phases.available is False);
- insufficient observations become limitations, never strengths or weaknesses; a strength, improvement or review
  whose overall confidence is low is withheld the same way (the contract's rule, re-checked here). A 'high-confidence
  concern' label is kept only with high confidence, at least MIN_CONCERN_OCCURRENCES occurrences and
  MIN_CONCERN_SECONDS measured, otherwise it reads 'Potential improvement';
- the limitations open with the run's data-quality note, stated once with what would help (language.data_quality:
  small players, no pitch keyframes, few players per team in view), then one short line per not-assessed measurement
  with its own reasons. On the four test runs the first reports' limitation text was 15-734 words per report (amateur
  118-182 s team B: 11 lines, "no pitch keyframes were set" 9 times; Veo: 12 lines, "only a few players of each team
  were in view" 10 times); a footage that shows a few players per team (Veo) gets one block saying why, and a bare
  "not assessed on this footage" per measurement. Measured on the same runs at the competitive level: 0-389 words
  (amateur team B 389 in 12 lines, Veo 148, broadcast 42-222 s team A 192), "keyframes were set" once per report.
  The breakdown points to them instead of repeating them;
- strengths of one compact block are one finding (language.GROUPS: width, length and area; both unit gaps), with
  every member's numbers and evidence (broadcast 42-222 s team B: five strengths of one block, three of them
  repeating one "why it matters" sentence). The finding keeps its first member as 'observation' (what feedback,
  links and the page key on) and lists all of them in 'observations';
- the descriptive context (block height of both teams, where the ball was) is the overview's context paragraph;
- every text field of every finding, from any engine, passes language.validate_text against its observations' facts
  and language.wording_problems (tone, intent, phases of play, repeated-pattern words in a short segment); a
  finding that fails is withheld and listed, never shown half-checked. Report-level sentences (summary, notes,
  limitations, context paragraph, section notes and reasons) are checked against the report's facts the same way,
  and each evidence caption against its observation and moment (a failing caption is left out);
- priorities (at most MAX_PRIORITIES) come from the potential improvements first (high-confidence concerns, then
  by confidence and by how far the rate exceeds the professional upper rate), then from the moments to review; a
  player's support-distance review is a role check (detectors.detect_isolation), so it links no drill and is never a
  training priority. Each priority links the drill written for its recommendation key (drills.choose), and the
  training section groups those drills;
- a report is a 'segment' report below MATCH_MIN_SECONDS (20 min). A PitchIQ run covers at most 3 minutes
  (README), so every report today is a segment report: its summary says the findings describe that segment only;
- identities are internal ids everywhere (in text and in records); display labels are rendered by the page and
  the PDF from the current result (language.resolve_players, coaching.js, pdf.py), since corrections renumber them;
- context levels change wording only: classifications, numbers, priorities and drills are the same at every level.
JSON from to_json has allow_nan=False and rounded numbers (metres 0.1, seconds 0.01, shares 0.001).
"""
import copy
import datetime
import json
import math
import re
from collections import defaultdict

from soccer.coaching import drills as drill_lib
from soccer.coaching import language_v1 as language  # the v1 words (language.py forwards them too)
from soccer.coaching.episodes import ENGINE_VERSION

REPORT_VERSION = 1
MATCH_MIN_SECONDS = 1200.0      # 20 min of analysed play make a match report; anything shorter is a segment
MAX_PRIORITIES = 3              # a coach works on a few things at a time
MIN_CONCERN_OCCURRENCES = 3     # contract: high_confidence_concern needs high confidence and >= 3 occurrences ...
MIN_CONCERN_SECONDS = language.SHORT_SEGMENT_SECONDS   # ... and (detectors.classify) at least 3 min measured
SECTIONS = ('defensiveOrganization', 'spacing', 'formation', 'attackingStructure', 'movementSupport')
SECTION_OF_CATEGORY = {'defensive_line': 'defensiveOrganization', 'spacing': 'spacing', 'formation': 'formation',
                       'attacking': 'attackingStructure', 'movement': 'movementSupport',
                       'support': 'movementSupport'}
CONFIDENCE_RANK = {'high': 0, 'medium': 1, 'low': 2}
SEVERITY_RANK = {'high_confidence_concern': 0, 'potential_improvement': 1}
FINDING_TEXT = ('headline', 'whatHappened', 'whyItMatters', 'whatToImprove', 'howToPractice', 'video',
                'confidenceNote')
SMALL_FOOTAGE = ('amateur', 'wide')   # footage whose report confidence is at most medium
CONTEXT_KEYS = ('line_height_gap', 'ball_context')   # neutral notes that make the overview's context paragraph
SPELL_HELP = ('incomplete_units', 'incomplete_team', 'short_stretch', 'short_measured')   # a better spell would help
OBSERVED_ROW = re.compile(r'Team ([AB]) (\d+(?:\.\d+)?)')   # __init__.run_meta's 'observedOutfieldPlayers' text row
METRIC_NAME_LEVEL = 'competitive'    # metric rows are named in one level's words: rows are the same at every level


def build_report(observations, evidence, meta, context=language.DEFAULT_LEVEL, team='A', team_name=None,
                 engine=None, library=None, generated=None):
    """The report dict (contract section 7) for team ('A' or 'B') at a context level ('development',
    'competitive', 'advanced'). observations and evidence are the engine's records (both teams allowed; others are
    ignored), meta describes the run: {'fingerprint', 'video': {name, key}, 'run', 'range': {t0, t1[, seconds]},
    'segmentStart' (frames[0].time), 'metresEstimated', 'pitch': {length, width, source}, 'footage':
    broadcast|amateur|wide, 'smallScene', 'keyless', 'registration': {high, medium, low shares}, 'corrections',
    'coverage': {...}, 'observedPlayers': {team: mean observed outfield players} (else read from the coverage row
    'observedOutfieldPlayers'), 'unavailable': {section or category: reason}, 'limitations': [reason], 'teamName'}.
    engine is a language.TextEngine (default TemplateEngine), library the drill library (default drills.json)."""
    if context not in language.LEVELS:
        raise ValueError(f'context must be one of {language.LEVELS}')
    if team not in ('A', 'B'):
        raise ValueError('team must be A or B')
    meta = meta or {}
    team_name = team_name or meta.get('teamName') or None
    engine = engine or language.TemplateEngine()
    library_note = None
    if library is None:
        try:
            library = drill_lib.load_drills()
        except (OSError, ValueError):
            library, library_note = {'drills': []}, 'The drill library could not be loaded, so no training is linked.'
    estimated = bool(meta.get('metresEstimated', True))
    obs = sorted((o for o in observations or () if isinstance(o, dict) and o.get('team') == team),
                 key=lambda o: (_t0(o), str(o.get('id'))))
    ids = {o.get('id') for o in obs}
    ev_index = {e.get('id'): e for e in evidence or () if isinstance(e, dict) and e.get('observation') in ids}
    rng = _range(meta, obs)
    kind = 'match' if rng['seconds'] >= MATCH_MIN_SECONDS else 'segment'
    extra_name = {'teamName': team_name} if team_name else None
    run_codes = _run_codes(meta, obs)
    wide = meta.get('footage') == 'wide'

    shown = {'strength': [], 'improvement': [], 'review': []}
    section_notes, section_measured = defaultdict(list), defaultdict(float)
    not_assessed, withheld_lines, context_notes, dropped = [], [], [], []
    role_titles = []
    suppressed = 0

    for o in obs:
        cls = o.get('classification')
        section = _section(o)
        evs = tuple(ev_index[e] for e in o.get('evidence') or () if e in ev_index)
        if o.get('status') == 'suppressed':
            suppressed += 1
            continue
        conf = (o.get('confidence') or {}).get('overall')
        role = _role_check(o)
        drill = drill_lib.choose(library, o) if cls in ('improvement', 'review') and not role else None
        label = _label(o)
        ctx = language.FindingContext(o, evs, context, team, team_name, kind, drill, estimated, label, run_codes)
        facts = language.facts_for(o, evs, drill, extra=extra_name)
        short = not _num(o.get('measuredSeconds')) or o['measuredSeconds'] < language.SHORT_SEGMENT_SECONDS
        if cls == 'insufficient':
            not_assessed.append((o, ctx, facts, short, section))
            continue
        if cls in language.FINDING_CLASSES and conf not in ('high', 'medium'):
            withheld_lines.append((language.withheld(ctx, 'low_confidence'), facts, short, section, o))
            continue
        if cls not in language.FINDING_CLASSES and cls != 'neutral':
            continue
        try:
            out = engine.render(ctx)
        except language.NoWording as e:
            withheld_lines.append((language.withheld(ctx, str(e) if str(e) in language.REASONS else 'no_wording'),
                                   facts, short, section, o))
            continue
        if cls == 'neutral':
            note = out.get('note')
            if note and _problems(note, facts, short):
                dropped.append(note)
            elif note and o.get('recommendationKey') in CONTEXT_KEYS:
                context_notes.append((CONTEXT_KEYS.index(o['recommendationKey']), note))
            elif note:
                section_notes[section].append(note)
                section_measured[section] = max(section_measured[section], _seconds(o))
            continue
        bad = [p for f in FINDING_TEXT for p in _problems(out.get(f), facts, short, practice=f == 'howToPractice')]
        bad += _problems(out.get('priority'), facts, short) if cls != 'strength' else []
        if not out.get('headline') or not out.get('whatHappened') or bad:
            withheld_lines.append((language.withheld(ctx, 'unverified_text'), facts, short, section, o))
            continue
        finding = {'observation': o.get('id'), 'observations': [o.get('id')], 'label': label,
                   'headline': out['headline'], 'whatHappened': out['whatHappened'],
                   'whyItMatters': out.get('whyItMatters'), 'whatToImprove': out.get('whatToImprove'),
                   'howToPractice': out.get('howToPractice') if cls != 'strength' else None,
                   'drill': drill.get('id') if drill and out.get('howToPractice') and cls != 'strength' else None,
                   'video': out.get('video'), 'evidence': [e.get('id') for e in evs],
                   'metrics': _metrics(o, estimated), 'confidence': conf,
                   'confidenceNote': out.get('confidenceNote')}
        if role:
            if out.get('priority'):
                role_titles.append(out['priority'])
            out = dict(out, priority=None)
        shown[cls].append({'order': _order(cls, o, label), 'finding': finding, 'priority': out.get('priority'),
                           'section': section, 'obs': [o], 'ctx': [ctx], 'facts': facts, 'short': short})
        section_measured[section] = max(section_measured[section], _seconds(o))

    shown['strength'] = _group(shown['strength'], context, estimated, extra_name, dropped)
    for cls in shown:
        shown[cls].sort(key=lambda item: item['order'])
    strengths = [x['finding'] for x in shown['strength']]
    improvements = [x['finding'] for x in shown['improvement']]
    reviews = [x['finding'] for x in shown['review']]
    by_section = defaultdict(list)
    for cls in ('improvement', 'review', 'strength'):
        for x in shown[cls]:
            by_section[x['section']].append(x['finding']['observation'])

    priorities, training, drill_order = [], [], {}
    for x in shown['improvement']+shown['review']:
        f, title_text = x['finding'], x['priority']
        if len(priorities) < MAX_PRIORITIES and title_text:
            priorities.append({'title': title_text, 'finding': f['observation'], 'drill': f['drill']})
        if f['drill']:
            if f['drill'] not in drill_order:
                drill_order[f['drill']] = len(training)
                training.append({'drill': drill_lib.drill_ref(drill_lib.drill(library, f['drill']), context),
                                 'findings': []})
            training[drill_order[f['drill']]]['findings'].append(f['observation'])

    overall = _overall([f['confidence'] for f in strengths+improvements+reviews], meta)
    counts = {'strength': len(strengths), 'improvement': len(improvements), 'review': len(reviews)}
    main, main_kind = (priorities[0]['title'], None) if priorities else (role_titles[0] if role_titles else None,
                                                                        'review')
    if priorities:
        main_kind = 'improvement' if any(f['observation'] == priorities[0]['finding'] for f in improvements) \
            else 'review'

    # Report-level text, checked against the facts of everything the report rests on.
    observed = _observed(meta)
    report_facts = language.Facts()
    for o in obs:
        evs = [ev_index[e] for e in o.get('evidence') or () if e in ev_index]
        report_facts = report_facts.merge(language.facts_for(o, evs, extra=extra_name))
    report_facts = report_facts.merge(language.facts_for(extra={
        'range': rng, 'pitch': meta.get('pitch'), 'registration': meta.get('registration'),
        'corrections': meta.get('corrections'), 'counts': counts, 'teamName': team_name, 'observed': observed,
        'keyframeAdvice': list(language.KEYFRAME_ADVICE)}))
    for d in (t['drill'] for t in training):
        report_facts = report_facts.merge(language.facts_for(drill=d))

    def checked(lines, short=False):
        keep = []
        for line in lines:
            if line and not _problems(line, report_facts, short):
                keep.append(line)
            elif line:
                dropped.append(line)
        return keep

    short_range = rng['seconds'] < language.SHORT_SEGMENT_SECONDS
    # The limitations: the data-quality note once, then one line per measurement not assessed (one line in all on a
    # footage with a few players per team in view), then the findings withheld.
    lines, topics_of, covered = [], defaultdict(list), []
    not_assessed.sort(key=lambda x: _canonical(x[0]))
    for o, ctx, facts, short, section in not_assessed:
        topic = _topic(o, context)
        if o.get('recommendationKey') not in CONTEXT_KEYS:      # the ball has no breakdown section
            topics_of[section].append(topic)
        covered.append({'observation': o.get('id'), 'topic': topic})
        # with a few players per team in view the data-quality block says why for all of them: a bare line each
        # (the PDF adds its own line, with every reason, for an observation whose topic has none)
        text = f'{topic}: not assessed on this footage.' if wide else engine_limitation(engine, ctx)
        if _problems(text, facts, short):
            dropped.append(text)
        elif text not in lines:
            lines.append(text)
    for text, facts, short, section, o in withheld_lines:
        topics_of[section].append(_topic(o, context))
        covered.append({'observation': o.get('id'), 'topic': _topic(o, context)})
        if _problems(text, facts, short):
            dropped.append(text)
        else:
            lines.append(text)
    vetoed = [o.get('detector') for o, *_ in not_assessed if 'small_scene_veto' in (o.get('insufficientReasons') or ())]
    help_spell = any(set(o.get('insufficientReasons') or ()) & set(SPELL_HELP) for o, *_ in not_assessed)
    quality = checked(language.data_quality(meta, run_codes, team, team_name, observed, help_spell, vetoed),
                      short_range)
    meta_limits = [r for r in (language.meta_reason(x) for x in meta.get('limitations') or ()
                               if x not in language.RUN_REASONS) if r]
    overview_limits = quality+checked(lines+meta_limits, short_range)
    summary = checked(language.summary_sentences(kind, team, team_name, rng['t0'], rng['t1'], rng['seconds'],
                                                 counts, main, main_kind, overall,
                                                 len(not_assessed)+len(withheld_lines),
                                                 strengths[0]['headline'] if strengths else None,
                                                 wide=wide and not (strengths or improvements or reviews),
                                                 no_reference='no_reference' in run_codes),
                      short_range)
    unavailable = meta.get('unavailable') or {}
    breakdown = {}
    for section in SECTIONS:
        findings = by_section.get(section, [])
        notes = section_notes.get(section, [])
        available = bool(findings or notes)
        reason = None
        if available:
            notes = [language.section_measured_note(section_measured[section], rng['seconds'])] + notes \
                if section_measured[section] > 0 else list(notes)
            pointer = language.section_pointer(topics_of.get(section, []))
            if pointer:
                notes.append(pointer)
        else:
            cats = [c for c, s in SECTION_OF_CATEGORY.items() if s == section]
            given = [language.meta_reason(unavailable.get(k)) for k in [section]+cats if unavailable.get(k)]
            given = [g for g in given if g]
            reason = given[0] if given else language.section_unavailable(section, topics=topics_of.get(section, []),
                                                                         no_reference='no_reference' in run_codes)
        notes = checked(notes, short_range)
        reason = (checked([reason], short_range) or [language.SECTION_REASONS['default']])[0] if reason else None
        breakdown[section] = {'available': available, 'reason': reason, 'findings': findings, 'notes': notes}
    # Evidence captions are written by evidence.py from the moment's own values: checked like any other text against
    # that observation and moment, and a caption that fails is left out (the moment itself stays).
    evidence_out = []
    for o in obs:
        for e in o.get('evidence') or ():
            if e not in ev_index:
                continue
            rec = copy.deepcopy(ev_index[e])
            caption = rec.get('caption')
            if caption and _problems(caption, language.facts_for(o, [ev_index[e]], extra=extra_name), False):
                dropped.append(caption)
                rec['caption'] = None
            evidence_out.append(rec)
    report_facts.add_number(suppressed)
    conf_notes = language.caveat_notes(meta, team)
    if suppressed:
        conf_notes.append(f'{suppressed} further observation{"s were" if suppressed != 1 else " was"} not reported '
                          f'(suppressed by the analysis).')
    if library_note:
        conf_notes.append(library_note)
    conf_notes = checked(conf_notes, short_range)
    context_lines = checked([n for _, n in sorted(context_notes, key=lambda x: x[0])], short_range)
    if dropped:
        conf_notes.append('Some report text was left out because it could not be checked against the measurements.')

    coverage = dict(meta.get('coverage') or {})
    coverage.update({
        'rangeSeconds': rng['seconds'],
        'observations': {'total': len(obs),
                         'shown': sum(len(f['observations']) for f in strengths+improvements+reviews),
                         'limitations': len(not_assessed)+len(withheld_lines), 'suppressed': suppressed},
        'measuredSeconds': {c: round(max((_seconds(o) for o in obs if o.get('category') == c), default=0.0), 2)
                            for c in sorted({o.get('category') for o in obs if o.get('category')})}})
    report = {
        'version': REPORT_VERSION, 'engineVersion': ENGINE_VERSION, 'textEngine': getattr(engine, 'name', 'custom'),
        'fingerprint': meta.get('fingerprint'),
        'generated': generated or datetime.datetime.now().isoformat(timespec='seconds'),
        'kind': kind, 'context': context, 'team': team, 'teamName': team_name,
        'video': copy.deepcopy(meta.get('video')), 'run': meta.get('run'), 'range': rng,
        # dataQualityLines: how many of the first limitations are the run's data-quality note (the rest are one line
        # per measurement); notAssessed: the observations those lines cover (a reader that adds its own "not assessed"
        # line per observation can skip these)
        'overview': {'title': language.title(kind), 'coverage': coverage, 'limitations': overview_limits,
                     'dataQualityLines': len(quality), 'notAssessed': covered, 'context': context_lines},
        'summary': summary,
        'strengths': strengths, 'improvements': improvements, 'reviews': reviews,
        'breakdown': breakdown, 'priorities': priorities, 'training': training,
        'confidence': {'overall': overall, 'notes': conf_notes},
        'evidence': evidence_out, 'observations': copy.deepcopy(obs), 'feedback': []}
    return _clean(report)


def engine_limitation(engine, ctx):
    """The engine's limitation line of an insufficient observation (TextEngine.render's 'limitation'), the
    template's when the engine has none."""
    try:
        out = engine.render(ctx)
    except language.NoWording:
        out = {}
    return out.get('limitation') or language.limitation(ctx)


def _group(items, level, estimated, extra_name, dropped):
    """The strengths with the members of each language.GROUPS group (at least its 'min' of them) made one finding:
    the first member's id as 'observation', all of them in 'observations', every member's numbers and evidence. A
    group whose text fails the fact check is left as separate findings."""
    out, members = [], defaultdict(list)
    for x in items:
        name = language.group_of(x['obs'][0])
        (members[name] if name else out).append(x)
    for name, xs in members.items():
        g = language.GROUPS[name]
        if len(xs) < g['min']:
            out += xs
            continue
        order = {p: i for i, p in enumerate(g['patterns'])}
        xs = sorted(xs, key=lambda x: order.get(x['obs'][0].get('pattern'), 9))
        try:
            text = language.group_render(name, [x['ctx'][0] for x in xs], level)
        except language.NoWording:
            out += xs
            continue
        facts = language.Facts()
        for x in xs:
            facts = facts.merge(x['facts'])
        short = all(x['short'] for x in xs)
        bad = [p for f in FINDING_TEXT for p in _problems(text.get(f), facts, short)]
        if bad:
            dropped.extend(t for t in (text.get(f) for f in FINDING_TEXT) if t)
            out += xs
            continue
        confs = [x['finding']['confidence'] for x in xs]
        worst = max(confs, key=lambda c: CONFIDENCE_RANK.get(c, 3))
        rows = [_metric_value('Measured time', max(_seconds(x['obs'][0]) for x in xs), 's')]
        for x in xs:
            rows += _member_rows(x['obs'][0], estimated)
        first = xs[0]['finding']
        finding = dict(first, observations=[x['finding']['observation'] for x in xs], headline=text['headline'],
                       whatHappened=text['whatHappened'], whyItMatters=text['whyItMatters'],
                       whatToImprove=text['whatToImprove'], howToPractice=None, drill=None, video=text['video'],
                       evidence=text['evidence'], metrics=rows, confidence=worst,
                       confidenceNote=text['confidenceNote'])
        out.append({'order': min(x['order'] for x in xs), 'finding': finding, 'priority': None,
                    'section': xs[0]['section'], 'obs': [x['obs'][0] for x in xs], 'ctx': [x['ctx'][0] for x in xs],
                    'facts': facts, 'short': short})
    return out


def to_json(report, indent=None):
    """The report as JSON text: strict (allow_nan=False), ASCII."""
    return json.dumps(report, allow_nan=False, ensure_ascii=True, indent=indent,
                      separators=(',', ':') if indent is None else None)


def _problems(text, facts, short, practice=False):
    if not text:
        return []
    return language.validate_text(text, facts)+language.wording_problems(text, short=short, practice=practice)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _t0(o):
    w = o.get('window') or {}
    return float(w['t0']) if _num(w.get('t0')) else 0.0


def _seconds(o):
    return float(o['measuredSeconds']) if _num(o.get('measuredSeconds')) else 0.0


def _section(o):
    s = SECTION_OF_CATEGORY.get(o.get('category'))
    if s is None and o.get('detector') in language.DETECTORS:
        s = SECTION_OF_CATEGORY.get(language.DETECTORS[o['detector']][1])
    return s or 'spacing'


def _canonical(o):
    """An observation's place in the detector registry (language.DETECTORS order, then its pattern), so lists of
    measurements read in the same order in every report."""
    dets = list(language.DETECTORS)
    pats = list(language.PATTERNS)
    return (dets.index(o['detector']) if o.get('detector') in dets else len(dets),
            pats.index(o['pattern']) if o.get('pattern') in pats else len(pats), _t0(o), str(o.get('id')))


def _topic(o, level):
    _, pinfo = language._pattern(o)
    return (pinfo.get('topic') or {}).get(level) or \
        (language.KEY_TOPIC.get(o.get('recommendationKey')) or {}).get(level) or 'A measurement'


def _role_check(o):
    """A player's support-distance review: a role check, never a training need (detectors.detect_isolation)."""
    return o.get('classification') == 'review' and (o.get('detector') == 'isolation' or
                                                     bool((o.get('metrics') or {}).get('roleCheck')))


def _run_codes(meta, obs):
    """The run-level reason codes this report states once (language.data_quality), from the run description and
    the observations' own reasons."""
    codes = {c for c in meta.get('limitations') or () if c in language.RUN_REASONS}
    if meta.get('smallScene') or meta.get('footage') in SMALL_FOOTAGE:
        codes.add('small_scene')
    if meta.get('keyless'):
        codes.add('keyless')
    if meta.get('footage') == 'wide':
        codes.add('few_players')
    for o in obs:
        for c in list(o.get('insufficientReasons') or ())+list((o.get('confidence') or {}).get('reasons') or ()):
            if c in ('keyless', 'small_scene', 'few_players'):
                codes.add(c)
    return tuple(sorted(codes))


def _observed(meta):
    """{team: mean observed outfield players} from meta['observedPlayers'], else from the coverage text row
    'observedOutfieldPlayers' ('Team A 2.0, Team B 2.6 (mean per sample)'); {} when neither is there."""
    given = meta.get('observedPlayers')
    if isinstance(given, dict):
        return {k: float(v) for k, v in given.items() if k in ('A', 'B') and _num(v)}
    row = (meta.get('coverage') or {}).get('observedOutfieldPlayers')
    if isinstance(row, str):
        return {m.group(1): float(m.group(2)) for m in OBSERVED_ROW.finditer(row)}
    return {}


def _label(o):
    """The finding's label: the severity's, with a high-confidence concern kept only when the contract's conditions
    hold on the record itself."""
    cls = o.get('classification')
    sev = o.get('severity') or language.DEFAULT_SEVERITY.get(cls)
    if sev == 'high_confidence_concern':
        conf = (o.get('confidence') or {}).get('overall')
        count = o.get('occurrenceCount') if _num(o.get('occurrenceCount')) else len(o.get('occurrences') or [])
        if cls != 'improvement' or conf != 'high' or count < MIN_CONCERN_OCCURRENCES or \
                _seconds(o) < MIN_CONCERN_SECONDS-1e-9:
            sev = 'potential_improvement'
    if cls == 'improvement' and sev not in SEVERITY_RANK:
        sev = 'potential_improvement'
    if cls == 'strength':
        sev = 'observed_strength'
    if cls == 'review':
        sev = 'review_opportunity'
    return language.LABELS.get(sev)


def _order(cls, o, label):
    conf = CONFIDENCE_RANK.get((o.get('confidence') or {}).get('overall'), 3)
    if cls == 'improvement':
        rate = o.get('ratePerMeasuredMinute')
        p90 = ((o.get('baseline') or {}).get('professionalRatePerMinute') or {}).get('p90')
        excess = float(rate)/float(p90) if _num(rate) and _num(p90) and p90 > 0 else 0.0
        count = o.get('occurrenceCount') if _num(o.get('occurrenceCount')) else 0
        sev = 0 if label == language.LABELS['high_confidence_concern'] else 1
        return (sev, conf, -excess, -count, _t0(o), str(o.get('id')))
    if cls == 'strength':
        return (conf, -_seconds(o), _t0(o), str(o.get('id')))
    return (int(_role_check(o)), conf, _t0(o), str(o.get('id')))


def _overall(levels, meta):
    ranks = [CONFIDENCE_RANK[c] for c in levels if c in CONFIDENCE_RANK]
    if not ranks:
        return 'low'
    worst = max(ranks)
    if worst == 0 and (meta.get('smallScene') or meta.get('keyless') or meta.get('footage') in SMALL_FOOTAGE):
        worst = 1
    return {0: 'high', 1: 'medium', 2: 'low'}[worst]


def _range(meta, obs):
    r = meta.get('range') or {}
    t0, t1 = r.get('t0'), r.get('t1')
    if not (_num(t0) and _num(t1)):
        ws = [o.get('window') or {} for o in obs]
        t0s = [w['t0'] for w in ws if _num(w.get('t0'))]
        t1s = [w['t1'] for w in ws if _num(w.get('t1'))]
        t0, t1 = (min(t0s), max(t1s)) if t0s and t1s else (0.0, 0.0)
    t0, t1 = float(t0), float(t1)
    seg0 = float(meta['segmentStart']) if _num(meta.get('segmentStart')) else t0
    seconds = float(r['seconds']) if _num(r.get('seconds')) else max(0.0, t1-t0)
    return {'t0': round(t0, 3), 't1': round(t1, 3), 'segmentT0': round(t0-seg0, 3), 'segmentT1': round(t1-seg0, 3),
            'seconds': round(seconds, 2)}


def _metric_value(name, value, unit, estimated=False):
    digits = {'m': 1, 'm2': 0, 's': 2, 'share': 3, 'per_min': 2, 'ratio': 2, 'count': 0}.get(unit, 3)
    if isinstance(value, str):
        return {'name': name, 'value': value, 'unit': unit}
    v = round(float(value), digits)
    row = {'name': name, 'value': int(v) if unit == 'count' else v, 'unit': unit}
    if unit in ('m', 'm2'):
        # explicit both ways: a row without the flag reads as the report's metres (estimated), so the page and the
        # PDF printed the professional median of real metres as "3.2 m (est.)" (broadcast 0-60 s team A's back line)
        row['estimated'] = bool(estimated)
    return row


def _member_rows(o, estimated):
    """A grouped finding's rows of one member, named by its topic: median, professional median, share beyond the
    professional p75 (the strength rule's numbers)."""
    _, pinfo = language._pattern(o)
    unit = language.UNITS.get(language._metric(o, pinfo))
    topic = (pinfo.get('topic') or {}).get(METRIC_NAME_LEVEL) or 'Measure'
    m = o.get('metrics') or {}
    rows = []
    for name, value, u, est in ((f'{topic}: median', m.get('median'), unit, estimated),
                                (f'{topic}: professional median', m.get('professionalP50'), unit, False),
                                (f'{topic}: share of time beyond the professional upper quarter',
                                 m.get('shareAboveProfessionalP75'), 'share', False)):
        if _num(value) and u:
            rows.append(_metric_value(name, value, u, est))
    return rows


def _metrics(o, estimated):
    """The key numbers of a finding as {name, value, unit} rows (only those the record has). Professional reference
    values are real metres (never marked estimated); the team's are estimates while the pitch size is assumed."""
    _, pinfo = language._pattern(o)
    metric = language._metric(o, pinfo)
    unit = language.UNITS.get(metric)
    cls = o.get('classification')
    m = o.get('metrics') or {}
    base = o.get('baseline') or {}
    thr = base.get('threshold') or {}
    pro = base.get('professionalRatePerMinute') or {}
    rows = []

    def add(name, value, u, est=True):
        if (_num(value) or isinstance(value, str)) and u:
            rows.append(_metric_value(name, value, u, estimated and est))
    add('Measured time', o.get('measuredSeconds'), 's')
    if o.get('recommendationKey') == 'formation_organisation':
        add('Formation reading', m.get('label'), 'label')
        add('Next closest reading', m.get('alternative'), 'label')
        kept, assessed = m.get('playersKeptRole'), m.get('playersAssessed')
        if _num(kept) and _num(assessed) and assessed > 0:
            add('Players who kept one role', f'{int(kept)} of {int(assessed)}', 'label')
        add('Back-line group changes', m.get('defenderSetChanges'), 'count')
        return rows
    if cls in ('improvement', 'review'):
        add('Occurrences', o.get('occurrenceCount'), 'count')
        add('Rate per measured minute', o.get('ratePerMeasuredMinute'), 'per_min')
        add('Professional count over the same time (upper)', m.get('professionalWindowP99'), 'count')
        add('Professional rate (median)', pro.get('p50'), 'per_min')
        add('Professional rate (upper)', pro.get('p90'), 'per_min')
        add('Threshold', thr.get('enter'), unit, est=False)
        peaks = [(_g(x, 'metrics', metric, 'peak')) for x in o.get('occurrences') or ()]
        peaks = [p for p in peaks if _num(p)]
        if peaks:
            add('Largest value', max(peaks), unit)
    else:
        pooled = _num(m.get('pooledMedian')) and m.get('strictestContext')
        add('Median', m.get('pooledMedian') if pooled else m.get('median'), unit)
        if pooled:
            add('Median in the strictest comparison', m.get('median'), unit)
        add('Professional median', m.get('professionalP50', thr.get('p50')), unit, est=False)
        add('Professional upper quarter', m.get('professionalP75', thr.get('p75')), unit, est=False)
        add('Share of time beyond the professional upper quarter', m.get('shareAboveProfessionalP75'), 'share')
    if metric == 'lagDM':
        add('Correlation', m.get('corr'), 'ratio')
    return rows


def _g(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> floats, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return int(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f if math.isfinite(f) else None
