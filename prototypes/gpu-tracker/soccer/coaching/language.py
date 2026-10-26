"""The fact check of every coaching text: a sentence may state only what the record it describes measured.

Why a gate and not trust in the wording. The coach acts on the report, and a confident false sentence is worse than a
missing one: the v1 report printed false sentences (critic B4: "nobody stood out" over an untested player, a camera
limit named as the cause on the Veo clip), and an optional local model may one day word the claims (DESIGN v2 6.6).
So every text, whoever wrote it, passes validate_text against its facts (facts_for): each number at its shown
precision, each percentage, mm:ss.ss time, formation label and player id must be one of the record's values (a duration
also as minutes, a share also as a percentage, a count also as a number word); any other digit or number word is a
problem. wording_problems refuses blame, exaggeration and intent words (TONE_BANNED), phase words where no phase was
measured (PHASE_BANNED) and, below SHORT_SEGMENT_SECONDS (3 min) of play, the repeated-pattern words (SHORT_BANNED).
explain.py (the v2 explanations) builds its own register and polarity gate on these two checks; resolve_players
renders internal ids (A-07) as the current display labels at view time, in one pass, because corrections renumber
the labels (soccer/labels.py).

What moved out (Oct 22, coaching v2 wave 5, DESIGN v2 11.2): the v1 wording, 555 fixed strings keyed by detector and
classification plus the template engine and the v1 report's sentences, went unchanged to language_v1.py. explain.py
replaces them with claim plans (the audit measured identical findings, counts and drills at every v1 level on all 8
team-runs). They are kept only while the server still serves the v1 report (wave 7 switches it); this module's
__getattr__ forwards every old name (language.TEMPLATES, language.REASONS, language.TemplateEngine, ...) to
language_v1, so the v1 report, pdf.py, the v1 drill validator and the v1 tests run unchanged, while the v2 modules
import only what is defined here (a test pins it, and the v2 interpretation fingerprint no longer hashes the v1
strings: coaching.code_files reads import statements, and the forward is not one).

Plain printable ASCII only: the PDF writes base-14 Helvetica with WinAnsi.
"""
import importlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

SHORT_SEGMENT_SECONDS = 180.0   # below 3 min of measured play: no repeated-pattern wording (DESIGN v2 3.6)

# Words a finding never uses: exaggeration, blame, praise beyond the measurement and intent (TONE_BANNED, every
# report); the phase words of the v1 report, which had no possession record (PHASE_BANNED; the v2 explanations word
# the possession-4 or coach-marked phase as "when we had the ball (estimated)" instead, explain.PHASE_RE); and the
# repeated-pattern words below SHORT_SEGMENT_SECONDS (SHORT_BANNED). Drill instructions (howToPractice) describe an
# exercise and may mention passing; nothing that describes the observed play may.
TONE_BANNED = ('terrible', 'awful', 'horrible', 'mistake', 'mistakes', 'blunder', 'fail', 'failed', 'failure',
               'failing', 'poor', 'poorly', 'bad', 'badly', 'lazy', 'careless', 'sloppy', 'disaster', 'disastrous',
               'shocking', 'unacceptable', 'excellent', 'perfect', 'perfectly', 'brilliant', 'outstanding', 'superb',
               'dominant', 'dominated', 'always', 'never', 'worst', 'must', 'should have', 'chose', 'choose to',
               'decided', 'deliberately', 'intended', 'intentionally', 'wanted', 'refused', 'tried to',
               'not good enough', 'lack of effort')
PHASE_BANNED = ('possession', 'in possession', 'out of possession', 'pass', 'passes', 'passing', 'passed', 'pressing',
                'press', 'presses', 'pressed', 'turnover', 'turnovers', 'counter-attack', 'counterattack',
                'transition', 'transitions')
SHORT_BANNED = ('recurring', 'recurred', 'recur', 'repeated', 'repeatedly', 'pattern', 'patterns', 'habit', 'habits',
                'habitually', 'regularly', 'consistently', 'often', 'frequently', 'tends to', 'tendency',
                'routinely', 'persistent', 'persistently')
NUMBER_WORDS = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'eleven': 11, 'twelve': 12, 'twenty': 20, 'twice': 2, 'thrice': 3}
SMALL_NUMBER_WORDS = {3: 'three', 4: 'four', 5: 'five', 6: 'six'}

TIME_KEYS = {'t0', 't1', 'peakT', 'time', 'segmentTime', 'seek', 'segmentT0', 'segmentT1', 't', 'segmentPeakT'}
TIME_TOKEN = re.compile(r'(?<![\w.:])(\d{1,3}):([0-5]\d)(?:\.(\d{1,3}))?(?!\w|:|\.\d)')
ID_TOKEN = re.compile(r'(?<![\w-])(?:[AB]-(?:\d{1,3}|\?)|GK-[A-Z0-9]{1,3})(?!\w)')
ID_RE = re.compile(r'(?:[AB]-(?:\d{1,3}|\?)|GK-[A-Z0-9]{1,3}|UNK-\d{1,4})')
FORMATION_TOKEN = re.compile(r'(?<![\w.-])\d(?:-\d){2,3}(?![\w-])')
FORMATION_RE = re.compile(r'\d(?:-\d){2,3}')
PERCENT_TOKEN = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)\s?%')
NUMBER_TOKEN = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)(?!\w|\.\d)')
SLOT = re.compile(r'\{(\w+)\}')
PRINTABLE = re.compile(r'[ -~]*')
NOT_IN_ROSTER = 'not in the roster now (was {id})'   # resolve_players: an id the current result no longer has

# ---------------------------------------------------------------------------------------------------------------
# Formatting (the same rules the fact check reads back)

def fmt_time(t):
    """Absolute source seconds as mm:ss.ss, the page's transport format (index.html timestamp); rounded to the
    centisecond first, so 59.999 reads 01:00.00."""
    cs = int(round(float(t)*100))
    m, rest = divmod(cs, 6000)
    return f'{m:02d}:{rest//100:02d}.{rest % 100:02d}'


def fmt_duration(s):
    """A duration: 5.4 s below 10 s, 57 s below 120 s, then minutes with one decimal."""
    s = float(s)
    if s < 10:
        return f'{s:.1f} s'
    if s < 120:
        return f'{s:.0f} s'
    return f'{s/60:.1f} minutes'


def fmt_percent(share):
    """A share as a percentage: one decimal below 10% (but 0%), whole numbers above."""
    v = 100.0*float(share)
    if 0 < abs(v) < 10:
        return f'{v:.1f}%'
    return f'{v:.0f}%'


def fmt_rate(r):
    """A rate per minute: one decimal from 1, two below."""
    r = float(r)
    return f'{r:.1f}' if abs(r) >= .95 else f'{r:.2f}'


def player_token(pid):
    """A player reference in report text: the internal id, shown as the current display label at view time."""
    return str(pid)


def resolve_players(text, displays):
    """The text with every internal id replaced by the player's current display label (displays: {id: display}),
    in one pass (B-12 -> B-09 never chains on to whoever was B-09). With a label map given, an id that is no longer in
    it (excluded, withdrawn or relabelled away by a correction) reads NOT_IN_ROSTER: left bare it would read as the
    renumbered player who now carries that label (a BC60 copy with A-03 excluded showed A-04 as A-03). Without a map
    the ids stay as they are."""
    if not isinstance(text, str):
        return text
    if not displays:
        return text
    return ID_TOKEN.sub(lambda m: str(displays.get(m.group(0)) or NOT_IN_ROSTER.format(id=m.group(0))), text)


TEXT_KEYS = ('summary', 'headline', 'whatHappened', 'whyItMatters', 'whatToImprove', 'howToPractice', 'video',
             'confidenceNote', 'title', 'limitations', 'context', 'notes', 'reason', 'caption', 'note', 'limitation')


def resolve_report(report, displays, _text=False):
    """A copy of a report with the ids in its coach-facing text (TEXT_KEYS) shown as display labels; the records
    (observations, evidence, players, finding and drill ids) keep their ids."""
    if isinstance(report, str):
        return resolve_players(report, displays) if _text else report
    if isinstance(report, dict):
        # overview.context is text (a list of notes); the report's top-level context is the level name
        return {k: resolve_report(v, displays, _text or (k in TEXT_KEYS and (k != 'context' or isinstance(v, list))))
                for k, v in report.items()}
    if isinstance(report, list):
        return [resolve_report(v, displays, _text) for v in report]
    return report


# ---------------------------------------------------------------------------------------------------------------
# Facts and the fact check

@dataclass
class Facts:
    """What a text may state: numbers (compared at the text's shown precision), percentages, times (absolute and
    segment seconds), player ids, display labels, formation labels and literal names (team, drill)."""
    numbers: list = field(default_factory=list)
    percents: list = field(default_factory=list)
    times: list = field(default_factory=list)
    ids: set = field(default_factory=set)
    displays: set = field(default_factory=set)
    formations: set = field(default_factory=set)
    literals: set = field(default_factory=set)

    def merge(self, other):
        return Facts(self.numbers+other.numbers, self.percents+other.percents, self.times+other.times,
                     self.ids | other.ids, self.displays | other.displays, self.formations | other.formations,
                     self.literals | other.literals)

    def add_number(self, x, duration=False):
        if not _num(x):
            return
        x = float(x)
        self.numbers += [x, abs(x)]
        if 0 <= x <= 1:
            self.percents.append(100*x)
        if duration:
            self.numbers.append(x/60)


def facts_for(observation=None, evidence=(), drill=None, extra=None, displays=None):
    """The facts of one observation with its evidence records (diagram and overlay geometry excluded), the linked
    drill (its name, player count and duration) and optional extra values (a dict walked the same way)."""
    f = Facts()
    if observation:
        _walk(observation, f)
        occ = observation.get('occurrences') or []
        for o in occ:
            if _num(o.get('t0')) and _num(o.get('t1')):
                f.add_number(o['t1']-o['t0'], duration=True)
        w = observation.get('window') or {}
        if _num(w.get('t0')) and _num(w.get('t1')):
            f.add_number(w['t1']-w['t0'], duration=True)
        f.add_number(len(observation.get('players') or []))
        f.add_number(len(occ))
    for e in evidence or ():
        _walk({k: v for k, v in e.items() if k not in ('diagram', 'videoOverlay')}, f)
    if drill:
        f.literals.add(drill.get('name') or '')
        from soccer.coaching import drills as _drills
        for x in _drills.drill_numbers(drill):
            f.add_number(x)
    if extra:
        _walk(extra, f)
    if displays:
        f.displays |= {str(v) for v in displays.values() if v}
    f.literals.discard('')
    return f


def _walk(x, f, key=None, seconds=False):
    """Collect the facts of x; a number under a key ending in 'Seconds' (or inside such a dict: labelSeconds) is a
    duration, so its minutes are a fact too."""
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            return
        f.add_number(x, duration=seconds or (bool(key) and (key.endswith('Seconds') or key == 'seconds')))
        if key in TIME_KEYS:
            f.times.append(float(x))
        return
    if isinstance(x, str):
        if FORMATION_RE.fullmatch(x):
            f.formations.add(x)
        elif ID_RE.fullmatch(x):
            f.ids.add(x)
        m = re.fullmatch(r'backline:(\d)', x)
        if m:
            f.add_number(int(m.group(1)))
        return
    if isinstance(x, dict):
        inner = seconds or (bool(key) and key.endswith('Seconds'))
        for k, v in x.items():
            if k == 'teamName' and isinstance(v, str):
                f.literals.add(v)
            if isinstance(k, str) and FORMATION_RE.fullmatch(k):
                f.formations.add(k)          # labelSeconds {'4-4-2': 70.2}
            _walk(v, f, k, inner)
        return
    if isinstance(x, (list, tuple)):
        for v in x:
            _walk(v, f, key, seconds)


def _close(value, decimals, pool):
    tol = .5*10**-decimals+1e-9
    return any(abs(p-value) <= tol for p in pool)


def validate_text(text, facts):
    """The statements in text that the facts do not support: [] when every number (at its shown precision),
    percentage, mm:ss.ss time, formation label and player (internal id, or a display label given in the facts) is
    in the facts and no other digit or number word is left. Literal names (team, drill) are skipped. Works on any
    engine's output."""
    if text is None:
        return []
    problems = []
    s = str(text)
    for lit in sorted(facts.literals, key=len, reverse=True):
        s = s.replace(lit, ' ')

    def sub(pattern, check):
        nonlocal s
        out = []
        last = 0
        for m in pattern.finditer(s):
            msg = check(m)
            if msg:
                problems.append(msg)
            out.append(s[last:m.start()])
            out.append(' ')
            last = m.end()
        out.append(s[last:])
        s = ''.join(out)

    def time(m):
        frac = m.group(3) or ''
        v = int(m.group(1))*60+int(m.group(2))+(float('0.'+frac) if frac else 0.0)
        return None if _close(v, len(frac), facts.times) else f'time {m.group(0)} is not in the facts'

    def bare_id(m):
        return None if m.group(0) in facts.ids or m.group(0) in facts.displays else \
            f'player {m.group(0)} is not in the facts'

    def formation(m):
        return None if m.group(0) in facts.formations else f'formation {m.group(0)} is not in the facts'

    def percent(m):
        v = m.group(1)
        d = len(v.split('.')[1]) if '.' in v else 0
        return None if _close(float(v), d, facts.percents) else f'percentage {m.group(0)} is not in the facts'

    def number(m):
        v = m.group(1)
        d = len(v.split('.')[1]) if '.' in v else 0
        return None if _close(float(v), d, facts.numbers) else f'number {v} is not in the facts'

    sub(TIME_TOKEN, time)
    sub(ID_TOKEN, bare_id)
    sub(FORMATION_TOKEN, formation)
    sub(PERCENT_TOKEN, percent)
    sub(NUMBER_TOKEN, number)
    counts = {round(x) for x in facts.numbers if abs(x-round(x)) < 1e-9}
    for w in re.findall(r'[A-Za-z]+', s):
        n = NUMBER_WORDS.get(w.lower())
        if n is not None and n not in counts:          # a number word is a count: it must be an exact whole fact
            problems.append(f'number word {w!r} is not in the facts')
    left = re.findall(r'\S*\d\S*', s)
    if left:
        problems.append(f'unverified digits in {left!r}')
    return problems


def wording_problems(text, short=False, practice=False):
    """Words the report never uses in this text: exaggeration, blame, intent, and (unless it describes a practice
    exercise) phases of play; with short=True also repeated-pattern words (below SHORT_SEGMENT_SECONDS)."""
    if not text:
        return []
    low = ' '+re.sub(r'[^a-z\-\' ]+', ' ', str(text).lower())+' '
    banned = list(TONE_BANNED)+([] if practice else list(PHASE_BANNED))+(list(SHORT_BANNED) if short else [])
    found = [w for w in banned if re.search(r'(?<![a-z\-])'+re.escape(w)+r'(?![a-z\-])', low)]
    bad = [c for c in str(text) if not PRINTABLE.fullmatch(c)]
    return [f'uses {w!r}' for w in found]+([f'characters outside printable ASCII {sorted(set(bad))!r}'] if bad
                                           else [])


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ---------------------------------------------------------------------------------------------------------------
# Engines

@runtime_checkable
class TextEngine(Protocol):
    """The interface of a wording engine: a name and render(request) -> dict of text fields. The v2 explanations
    (explain.py) send only the finding's claims and the register (DESIGN v2 6.6) and use the text only when it passes
    their gate; the deprecated v1 template engine (language_v1.TemplateEngine) renders one observation. No model is
    shipped and nothing calls the network."""
    name: str

    def render(self, request) -> dict:
        ...


# ---------------------------------------------------------------------------------------------------------------
# The v1 names (language_v1.py, deprecated): forwarded so the v1 report and its callers keep working until wave 7

V1_MODULE = 'soccer.coaching.language_v1'


def __getattr__(name):
    """language.X for a v1 name (TEMPLATES, REASONS, LEVELS, TemplateEngine, ...): the attribute of language_v1."""
    if name.startswith('__'):
        raise AttributeError(name)
    v1 = importlib.import_module(V1_MODULE)
    try:
        return getattr(v1, name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
