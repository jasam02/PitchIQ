"""The page (coaching.js, run in node: skipped without node) and the PDF (soccer/coaching/pdf.py) show a 'moments
worth a look' review (fix round 2, LEAD decision (b): repeated clear failures below the level's verdict bar; assess
limitation 'moments_worth_a_look') as report.REVIEW_PARTS asks (TEXT's request, build/INTERFACES.md 'TEXT (fix round
2)' section 4), on a REAL report (coaching.interpret on the small_field_scale/declared scene at level 4: one D5
moments review, "10 of the 11" judged in 8 spells, through TEXT's fixtures), given the no-objective record assess makes
when no check qualifies (that report has a level objective: it is printed once), and on the same report with a
possible issue added:

- WHAT NEEDS WORK prints the work items, then "Possible issues to review", then "Moments worth a look" with its own
  count and intro (report.REVIEW_PARTS, verbatim up to the page's typographic apostrophe); the section's count is the
  work items and the possible issues;
- the card's badge is "Moments worth a look" (never "Possible issue to review"), the action reads "What to look for."
  (never "What could help."), the counts are the clear moments of the judged ones ("Clear in 10 of the 11 moments
  PitchIQ could judge (3 separate spells)", never "Missing in x of <all>"), and the evidence guardrail it fails by
  construction gives no "Could not be confirmed" note, the suggested action no "No reachable better position" note;
- the key-moment rows and the side column carry the same badge; the side column says nothing about a suggested area;
- a next objective without a skill (assess found none) is its text alone, without "From the level's objectives, not
  from this match.".

Before this round the page and the PDF printed the review under "Possible issues to review" with that part's intro,
the "Possible issue to review" badge, "Missing in 11 of 36 moments", "Could not be confirmed: enough repeated
evidence." and the level's-objectives line under the no-objective text: 9 of these 10 tests fail on the two files as
they were before this round (session scratch coaching2/fix2/eval/view_before.py injects them; the tenth checks the
fixture report).

Run:  python -m unittest test_coaching_moments_view        (from this directory; about 3 s)
"""
import copy
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import coaching_text_fixtures as TF  # noqa: E402
from soccer.coaching import pdf, report as R, schema as S  # noqa: E402
from test_coaching_panel import (CATALOG, NODE, SPANS_EMPTY, run, saved_block, v2_answer, v2_settings,  # noqa: E402
                                 v2job)
from test_coaching_pdf import text_of  # noqa: E402

MOMENTS = 'moments_worth_a_look'
APOS = str.maketrans({chr(0x2019): "'"})     # the page's typographic apostrophe
DOT = chr(0xb7)                                 # the page's count separator


def job(rep, steps=None):
    """A page job on a real report, as test_coaching_panel.V2Real builds one."""
    built = TF.tiers('small_field_scale', 'declared')[0]
    res = built.result
    slim = {'version': 3, 'video': res['video'], 'match': res['match'],
            'frames': [{'time': f['time'], 'frame': f['frame'], 'people': []} for f in (res['frames'][0], res['frames'][-1])]}
    return v2job(report=rep, answers=[v2_answer(rep, drills={})], result=slim, info={'video': res['video'], 'videoKey': '0'*64},
                 profiles=CATALOG, settings=v2_settings(saved_block(ageGroup=rep['profile']['ageGroup'],
                                                                   playingLevel=rep['profile']['level'])),
                 steps=steps or [], spans=SPANS_EMPTY)


def moments_report():
    """The level-4 report of small_field_scale/declared (team A): one D5 'moments worth a look' review."""
    return TF.report('small_field_scale', 'declared', level=4)


def count_line(rep):
    """The count line the page and the PDF print for the report's moments review: its clear moments of the judged
    ones, its spells, and since fix round 3 the failures that were not clear (the counts add up: critic B-3, FACTS
    F-4 of round 2)."""
    m = next(f for f in rep['reviews'] if MOMENTS in f['limitations'])['measurements']
    c, d, k, n = m['clearMoments'], m['decidedMoments'], m['clearSpells'], m['failingMoments']
    line = f'Clear in {"all " + str(d) if c >= d else f"{c} of the {d}"} moments PitchIQ could judge ({k} separate spell{"" if k == 1 else "s"})'
    return line + (f', {n-c} more fell short less clearly' if n > c else '')


def no_objective(rep):
    """The report with the next objective assess gives when no check of the level qualifies (skill None)."""
    r = copy.deepcopy(rep)
    r['nextObjective'] = {'skill': None, 'text': R.no_objective_text(r['profile']), 'fromMatch': False, 'finding': None}
    return r


def with_issue(rep):
    """The report with an old-path possible issue added (a copy of the moments review without the limitation, worded
    as a possible issue), still schema-valid."""
    r = copy.deepcopy(rep)
    mw = next(f for f in r['reviews'] if MOMENTS in f['limitations'])
    issue = copy.deepcopy(mw)
    issue['id'] = S.record_id('fnd', 'issue-copy')
    issue['key'] = issue['key'].replace('|team|', '|left|')
    issue['observation'] = 'wide_defender_advance'
    issue['key'] = 'D3' + issue['key'][2:]
    issue['limitations'] = ['identity_uncertain']
    issue['guardrails'] = dict(issue['guardrails'], identity='fail', evidence='pass')
    for k in ('clearMoments', 'clearSpells', 'decidedMoments', 'failingMoments'):
        issue['measurements'].pop(k, None)
    issue['text'] = {'headline': 'Possible issue to review: an advance on the left', 'short': 'An advance on the left',
                     'explanation': 'PitchIQ could not confirm who advanced on the left.',
                     'action': 'Watch who stayed behind the advance.'}
    r['reviews'] = [issue] + r['reviews']
    return r


class TheReport(unittest.TestCase):
    """The real report the view tests read holds what they assume (if ASSESS's rules change, say so here first)."""

    def test_the_scene_gives_one_moments_review(self):
        rep = moments_report()
        mws = [f for f in rep['reviews'] if MOMENTS in f['limitations']]
        self.assertEqual(len(mws), 1)
        m = mws[0]['measurements']
        # The counts are ASSESS's: (10, 11, 8) clear / judged / spells until fix round 3, (7, 11, 5) with its clear-
        # moment gate (critic B-2); the view tests below read them from the report and check only how they are said.
        self.assertTrue(3 <= m['clearSpells'] <= m['clearMoments'] <= m['failingMoments'] <= m['decidedMoments'], m)
        self.assertEqual(mws[0]['guardrails']['evidence'], 'fail')
        self.assertEqual(rep['nextObjective']['skill'], 'back_line_coordination')
        self.assertEqual(S.validate('report', no_objective(rep)), [])
        self.assertEqual(S.validate('report', with_issue(rep)), [])
        self.assertEqual([p['part'] for p, _ in R.reviews_by_part(with_issue(rep))], ['issues', 'moments'])


class ThePdf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = no_objective(moments_report())
        cls.text = text_of(pdf.build_report_pdf(cls.rep, None, {})[0])
        cls.mixed = text_of(pdf.build_report_pdf(with_issue(cls.rep), None, {})[0])
        cls.level = text_of(pdf.build_report_pdf(moments_report(), None, {})[0])

    def test_the_moments_part_with_its_title_count_and_intro(self):
        part = dict((p['part'], p) for p in R.REVIEW_PARTS)['moments']
        self.assertIn(part['title']+' (1)', self.text)
        self.assertIn(' '.join(part['intro'].split()[:8]), self.text)
        self.assertNotIn('Possible issues to review', self.text)
        self.assertIn('3. What needs work (0)', self.text, 'the section counts work items and possible issues')
        self.assertIn('moment worth a look', self.text, 'its own tile on the cover')
        self.assertNotIn('issue to review 0', self.text.replace('issues to review', 'issue to review'))

    def test_the_card(self):
        self.assertIn('MOMENTS WORTH A LOOK', self.text)
        self.assertNotIn('POSSIBLE ISSUE TO REVIEW', self.text)
        self.assertIn('WHAT TO LOOK FOR', self.text)
        self.assertNotIn('WHAT COULD HELP', self.text)
        self.assertIn(count_line(self.rep), self.text)
        self.assertNotRegex(self.text, r'Missing in \d+ of \d+ moments')
        self.assertNotIn('enough repeated evidence', self.text)
        self.assertNotIn('No reachable better position could be confirmed', self.text)
        self.assertNotIn('No area is drawn', self.text)

    def test_the_next_objective_without_a_skill(self):
        self.assertIn(self.rep['nextObjective']['text'][:40], ' '.join(self.text.split()))
        self.assertNotIn('not from this match.', self.text)
        # fix round 3 (TEXT, FACTS F-9 = COACH F-4): the S-8 objective of small_field_scale/declared at level 4 is
        # chosen from this match's checks and says so once; the old "not from this match" label is gone from the
        # objective box (the optional drill's own label "for the next objective, not from this match's evidence" is
        # an open DRILLS / PAGE item of the round, not checked here)
        flat = ' '.join(self.level.split())
        self.assertEqual(flat.count("Chosen from this match's checks"), 1, 'said once')
        box = flat[flat.index('Next development objective'):flat.index('Sources (')]
        self.assertNotIn('not from this match', box)

    def test_possible_issues_then_moments(self):
        t = self.mixed
        self.assertLess(t.index('Possible issues to review'), t.index('Moments worth a look (1)'))
        self.assertIn('3. What needs work (1)', t)
        self.assertIn('POSSIBLE ISSUE TO REVIEW', t)
        self.assertIn('Could not be confirmed: who the player was.', t)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class ThePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = no_objective(moments_report())
        mixed = with_issue(cls.rep)
        cls.mw = next(f for f in cls.rep['reviews'] if MOMENTS in f['limitations'])
        first = cls.mw['timestamps'][0]['evidence']
        steps = [{'do': 'click', 'in': 'main', 'tag': 'BUTTON', 'data': {'evidence': first}, 'name': 'jump'}]
        cls.out, cls.mixed = run(job(cls.rep, steps), job(mixed))

    def section(self, out, key):
        return next(s['text'] for s in out['sections'] if s['key'] == key)

    def test_the_moments_part_with_its_title_count_and_intro(self):
        nw = self.section(self.out['first'], 'needsWork')
        part = dict((p['part'], p) for p in R.REVIEW_PARTS)['moments']
        self.assertIn(f"{part['title']} {DOT} 1", nw)
        self.assertIn(part['intro'], nw.translate(APOS))
        self.assertNotIn('Possible issues to review', nw)
        self.assertTrue(nw.startswith(f'What needs work {DOT} 0'), nw[:40])

    def test_the_card(self):
        card = next(c for c in self.out['first']['cards'] if c['finding'] == self.mw['id'])
        text = card['text']
        self.assertIn('Moments worth a look', text)
        self.assertNotIn('Possible issue to review', text)
        self.assertIn('What to look for.', text)
        self.assertNotIn('What could help.', text)
        self.assertIn(count_line(self.rep), text)
        self.assertNotRegex(text, r'Missing in \d+ of \d+ moments')
        self.assertNotIn('enough repeated evidence', text)
        self.assertNotIn('No reachable better position could be confirmed', text)
        self.assertNotIn('Drill:', text)

    def test_key_moments_and_the_side_column(self):
        ids = {t['evidence'] for t in self.mw['timestamps']}
        rows = [m for m in self.out['first']['moments'] if m['evidence'] in ids]
        self.assertEqual(len(rows), len(ids))
        for row in rows:
            self.assertIn('Moments worth a look', row['text'])
            self.assertNotIn('Possible issue to review', row['text'])
        side = self.out['jump']['side']['text']
        self.assertIn('Moments worth a look', side)
        self.assertIn('What to look for.', side)
        self.assertNotIn('No suggested area', side)

    def test_the_next_objective_without_a_skill(self):
        prog = self.section(self.out['first'], 'progression')
        self.assertIn(self.rep['nextObjective']['text'], prog)
        self.assertNotIn('not from this match', prog)

    def test_possible_issues_then_moments(self):
        nw = self.section(self.mixed['first'], 'needsWork')
        self.assertLess(nw.index('Possible issues to review'), nw.index(f'Moments worth a look {DOT} 1'))
        self.assertTrue(nw.startswith(f'What needs work {DOT} 1'), nw[:40])
        issue = next(c for c in self.mixed['first']['cards'] if c['finding'] == S.record_id('fnd', 'issue-copy'))
        self.assertIn('Possible issue to review', issue['text'])
        self.assertIn('Could not be confirmed: who the player was.', issue['text'])


if __name__ == '__main__':
    unittest.main()
