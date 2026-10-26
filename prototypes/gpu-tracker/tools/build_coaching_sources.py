"""Build soccer/coaching/sources.json, the registry of the coaching sources the v2 coaching profiles cite, from the three
research passes kept in tools/annotations/coaching/research/ (DESIGN v2 section 2.7).

Why a generated registry. The three research passes of 2026-10-21 (levels 1-2, levels 3-4, levels 5-7) recorded 165
source entries: the same page appears in two or three passes under different ids (US_GM_U1314_PDF in one pass,
USS-GM1314 in another), a page and its PDF were read together in one pass and apart in another, and one England
article was read at two URL paths. A profile that cites
"US_GM_U910_PDF" in one place and "USS-GM1516" in another cannot be checked or shown with one title and one link. So
this script merges the entries by URL (union over each entry's url and extra urls, plus one recorded manual merge for
the same article at two paths), gives each merged source one canonical id of the form <ORG>-<KIND>-<AGE|TOPIC>
(USS-GM-U78, USS-F-7V7, EF-COVER-RECOVER, EF-PTC, UEFA-PRO, CS-CPRO), and keeps every research id as an alias so a
citation written with any of them resolves. Measured on the three files: 165 entries -> 122 sources (90 read in full,
26 in part, 6 not read), 32 of them merged from more than one entry, 483 tagged items, 22 recorded gaps.

What "read" means. Each pass recorded its own status (read_fully / read_partially / not_read in the level 1-4 passes;
read / read-partial / not-fetched in the level 5-7 pass). The registry keeps every pass's status in `readings` and
sets `read` to the best of them (full > partial > not_read): a page that one pass read in full was read. Pages read
through the fetch tool were summarised by a model (the passes say so in `how`); small wording details from them may be
summary artefacts. A source whose `read` is not_read is never cited for content (profiles.validate_profiles checks it).

What is not a source. The `unreadable` list keeps what the passes could not verify (PDFs over the fetch limit, pages
behind registration, merged table cells), so the profile panel can say what is missing, not only what was read.

The output is deterministic (sources sorted by id, ASCII, CRLF), so `--check` can fail a test or a review when
sources.json was edited by hand or the research copies changed without a rebuild. The research copies are compared by
the sha256 of their canonical JSON (sorted keys, no whitespace), so a CRLF or LF checkout gives the same hash.

Usage (from prototypes/gpu-tracker):
    python tools/build_coaching_sources.py            # rebuild soccer/coaching/sources.json
    python tools/build_coaching_sources.py --check    # exit 1 if sources.json is stale or profiles.json is invalid
    python tools/build_coaching_sources.py --import DIR   # copy the three research JSON files from DIR (CRLF), then build
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT/'tools'/'annotations'/'coaching'/'research'
SOURCES_PATH = ROOT/'soccer'/'coaching'/'sources.json'
ACCESSED = '2026-10-21'      # the date all three passes read their pages (each entry also carries its own date)

# The research passes, in citation-priority order (the level 1-4 passes give the clearer titles).
PASSES = (('L1L2', 'L1L2_foundation_early_development.json'),
          ('L3L4', 'L3L4_developing_intermediate.json'),
          ('L57', 'levels5-7.json'))

# Status words of the passes -> the registry's three values.
STATUS = {'read_fully': 'full', 'read': 'full', 'read_partially': 'partial', 'read-partial': 'partial',
          'not_read': 'not_read', 'not-fetched': 'not_read'}
READ_ORDER = {'not_read': 0, 'partial': 1, 'full': 2}

# Canonical ids, keyed by any research id of the merged source. Ids written in the level 5-7 style (ORG-..., no
# underscore) become canonical as they are unless an entry below renames their group; every group must end with
# exactly one canonical id (the build fails otherwise). DESIGN 2.7 names USS-GM-U78, USS-CG-U910, USS-F-7V7,
# EF-COVER-RECOVER, EF-PTC, UEFA-PRO and CS-CPRO; the drills of DESIGN 7.2 name EF-SWIFT-SUPPORT, EF-PROG-7-11,
# EF-PRACTISING-PRESS, EF-SUPPORT-LINE, EF-COVER-RECOVER-SESSION, EF-PROG-PRESS-COVER, USS-F-4V4, USS-F-9V9.
CANON = {
    # U.S. Soccer: age group guide pages (AGG), coaching-guide PDFs (CG), game models (GM), formations (F)
    'US_AGE_INDEX': 'USS-AGG', 'US_AGE_U6': 'USS-AGG-U6', 'US_AGE_U78': 'USS-AGG-U78', 'US_AGE_U910': 'USS-AGG-U910',
    'US_AGE_U1112': 'USS-AGG-U1112', 'US_AGE_U1314': 'USS-AGG-U1314', 'USS-AGG1516': 'USS-AGG-U1516',
    'USS-AGG17': 'USS-AGG-U17',
    'US_CG_U78_PDF': 'USS-CG-U78', 'US_CG_U910_PDF': 'USS-CG-U910', 'US_CG_U1112_PDF': 'USS-CG-U1112',
    'US_CG_U1314_PDF': 'USS-CG-U1314', 'US_CGPAGE_U1112': 'USS-CGPAGE-U1112', 'US_CGPAGE_U1314': 'USS-CGPAGE-U1314',
    'US_GM_INDEX': 'USS-GM', 'US_GM_U78_PDF': 'USS-GM-U78', 'US_GM_U78_PAGE': 'USS-GM-U78-PAGE',
    'US_GM_U910_PDF': 'USS-GM-U910', 'US_GM_U910_PAGE': 'USS-GM-U910-PAGE', 'US_GM_U1112_PDF': 'USS-GM-U1112',
    'US_GM_U1112_PAGE': 'USS-GM-U1112-PAGE', 'US_GM_U1314_PDF': 'USS-GM-U1314',
    'US_GM_U1314_PAGE': 'USS-GM-U1314-PAGE', 'US_GM_U1516_PDF': 'USS-GM-U1516', 'USS-GM17': 'USS-GM-U17',
    'US_GM_USING_PDF': 'USS-GM-USING',
    'US_FORMATIONS': 'USS-F', 'US_F4_PDF': 'USS-F-4V4', 'US_F7_PDF': 'USS-F-7V7', 'US_F7_PAGE': 'USS-F-7V7-PAGE',
    'US_F9_PDF': 'USS-F-9V9', 'US_F9_PAGE': 'USS-F-9V9-PAGE', 'US_F11_PDF': 'USS-F-11V11',
    'US_FORMATION_SELECT_PDF': 'USS-F-SELECT',
    'US_GAME_FORMATS': 'USS-FORMATS', 'US_FORMATS_PDF': 'USS-FORMATS-PDF', 'US_DESIGN_PDF': 'USS-DESIGN',
    'US_COACHING_APPROACH': 'USS-APPROACH', 'US_SESSIONS': 'USS-SESSIONS', 'US_SSG_2017': 'USS-SSG-2017',
    'US_R2R_2015': 'USS-R2R-2015', 'US_CONCUSSION_HUB': 'USS-CONCUSSION', 'US_PDI_2017': 'USS-PDI-2017',
    'US_COURSES_2018': 'USS-COURSES-2018', 'US_PLAYER_U1112': 'USS-PLAYER-U1112',
    'US_PLAYER_U1314': 'USS-PLAYER-U1314',
    # Canada Soccer (CA_ in the level 1-2 pass, CS_ / CS- elsewhere: one prefix here)
    'CA_GRASSROOTS_EDU': 'CS-GCE', 'CA_GRASSROOTS_PAGE': 'CS-GRASSROOTS', 'CA_STANDARDS_PDF': 'CS-STANDARDS-2026',
    'CS_STANDARDS': 'CS-STANDARDS-2024', 'CA_PRINCIPLES_PDF': 'CS-PRINCIPLES', 'CA_GAMES_PDF': 'CS-GAMES',
    'CA_SSG_GUIDE_PDF': 'CS-SSG',
    # England Football (learn.englandfootball.com, englandfootball.com, futurefit)
    'EF_PCB_2023': 'EF-PCB-2023', 'EF_PCB_2023_PDF': 'EF-PCB-2023-PDF', 'EF_PCB_2025': 'EF-PCB25',
    'EF_PROG_7_11': 'EF-PROG-7-11', 'EF_SWIFT_SUPPORT': 'EF-SWIFT-SUPPORT', 'EF_PRACTISING_PRESS': 'EF-PRACTISING-PRESS',
    'EF_SUPPORT_LINE': 'EF-SUPPORT-LINE', 'EF_COVER_RECOVER': 'EF-COVER-RECOVER', 'EF_STEP': 'EF-STEP',
    'EF_12_16_SERIES': 'EF-COVER-RECOVER-SESSION', 'EF_HEADING': 'EF-HEADING', 'EF_PITCH_SIZES': 'EF-PITCH-SIZES',
    'EF_9V9': 'EF-9V9', 'EF_11V11': 'EF-11V11', 'EF_GAME_STATE': 'EF-STATE', 'EF_TACTICS': 'EF-TACTICS',
    'EF_INDIV_TACTICS': 'EF-INDIV-TACTICS', 'EF_PROG_PRESS_COVER': 'EF-PROG-PRESS-COVER',
    'EF_SMALL_NUMBERS_2022': 'EF-DSN22', 'EF_SMALL_NUMBERS_2026': 'EF-DSN26', 'EF_CHALLENGE_RECOVER': 'EF-CRS',
    'EF_PRESSING_ZONES': 'EF-PZ', 'EF_CREATE_SPACE': 'EF-CREATE-SPACE', 'EF_DEFEND_TRANSITION': 'EF-DLE-TRANS',
    'EF_DEFEND_ROLES': 'EF-DLE-ROLES', 'EF_DEFEND_WIDE': 'EF-DLE-WIDE',
    # The FA (thefa.com)
    'FA_TRANSITION_YOUNG': 'FA-TRANSITION-YOUNG', 'FA_DNA_PRIORITIES': 'FA-DNA-PRIORITIES',
    'FA_DNA_FOUNDATION': 'FA-DNA-FOUNDATION', 'FA_HEADING_NEWS_2024': 'FA-HEADING-NEWS-2024',
    'FA_PITCH_U78': 'FA-PITCH-U78', 'FA_PITCH_U910': 'FA-PITCH-U910',
}
# The same England article read at two URL paths (/articles/resources/... and /articles-and-resources/coaching/
# resources/...), with the same title: merged by hand, recorded here.
MANUAL_MERGES = (('EF_DEFEND_WIDE', 'EF-DLE-WIDE'),)
# Spellings DESIGN.md uses that are neither a research id nor its hyphen form.
EXTRA_ALIASES = {'CS-GAMES': ('CA-GAMES',)}

PUBLISHERS = (('documents.ussoccer.com', 'U.S. Soccer'), ('ussoccer.com', 'U.S. Soccer'),
              ('englandfootball.com', 'England Football (The FA)'), ('thefa.com', 'The FA'),
              ('canada-soccer.myshopify.com', 'Canada Soccer'), ('canadasoccer.com', 'Canada Soccer'),
              ('uefa.com', 'UEFA'))
READ_RULE = ('read = the best status any research pass recorded (full > partial > not_read); readings lists each '
             "pass's own status. Pages read through a fetch tool were summarised by a model, so small wording "
             'details from them may be summary artefacts. A not_read source is never cited for content.')
DISCLAIMER = ("PitchIQ's checks and numbers are its own proposals; the sources describe coaching objectives, not "
              "PitchIQ's thresholds. No federation reviewed or endorses PitchIQ.")


def canonical_sha(raw):
    """sha256 of a JSON document's canonical form (sorted keys, no whitespace, ASCII): line endings do not matter."""
    data = json.loads(raw)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                          .encode('ascii')).hexdigest()


def norm_url(url):
    u = url.strip().lower()
    for p in ('https://', 'http://'):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith('www.'):
        u = u[4:]
    return u.rstrip('/')


def publisher_of(url, given=None):
    host = norm_url(url).split('/')[0]
    for dom, name in PUBLISHERS:
        if host == dom or host.endswith('.'+dom):
            return name
    return given or host


def load_research(directory=RESEARCH_DIR):
    """[(pass, data, raw bytes)] of the three research files; raises ValueError on a missing, non-ASCII or broken
    file."""
    out = []
    for name, fname in PASSES:
        p = Path(directory)/fname
        raw = p.read_bytes()
        try:
            raw.decode('ascii')
        except UnicodeDecodeError as error:
            raise ValueError(f'{fname}: not ASCII ({error})') from None
        out.append((name, json.loads(raw), raw))
    return out


def walk_items(node, path=''):
    """(path, item) for every tagged item (a dict with a 'text') of a research file, sources list excluded. Paths
    look like 'levels.L1_FOUNDATION.learning_objectives[0]' or 'levels[0].learningObjectives[3]'."""
    if isinstance(node, dict):
        if isinstance(node.get('text'), str):
            yield path, node
            return
        for k, v in node.items():
            if path == '' and k == 'sources':
                continue
            yield from walk_items(v, f'{path}.{k}' if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_items(v, f'{path}[{i}]')


def research_item(research, pointer):
    """The research item a profiles.json pointer names ('L1L2:levels.L1_FOUNDATION.learning_objectives[0]'), or
    None."""
    if not isinstance(pointer, str) or ':' not in pointer:
        return None
    name, path = pointer.split(':', 1)
    for pname, data, _raw in research:
        if pname == name:
            for p, item in walk_items(data):
                if p == path:
                    return item
    return None


def _entries(research):
    out = []
    for name, data, _raw in research:
        for s in data['sources']:
            urls = [s['url']]+[u for u in (s.get('extraUrls') or []) if isinstance(u, str)]
            out.append({'pass': name, 'id': s['id'], 'urls': urls, 'status': s.get('status'), 'title': s['title'],
                        'publisher': s.get('publisher'), 'how': s.get('how_read'), 'notes': s.get('notes'),
                        'accessed': s.get('accessed') or data.get('accessed') or ACCESSED})
    return out


def merge(research):
    """The merged sources: [{id, title, publisher, url, extraUrls, accessed, read, readings, aliases, notes}] sorted
    by id, and {any alias or research id -> canonical id}. Raises ValueError when a group has no canonical id, two
    canonical ids, or an id that maps to two groups."""
    entries = _entries(research)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in entries:
        keys = [norm_url(u) for u in e['urls']]
        for k in keys[1:]:
            union(keys[0], k)
    first_url = {}
    for e in entries:
        first_url.setdefault(e['id'], norm_url(e['urls'][0]))
    for a, b in MANUAL_MERGES:
        if a not in first_url or b not in first_url:
            raise ValueError(f'manual merge {a} + {b}: id not in the research files')
        union(first_url[a], first_url[b])
    groups = {}
    for e in entries:
        groups.setdefault(find(norm_url(e['urls'][0])), []).append(e)
    sources, alias_map = [], {}
    for members in groups.values():
        explicit = {CANON[m['id']] for m in members if m['id'] in CANON}
        if len(explicit) > 1:
            raise ValueError(f'conflicting canonical ids {sorted(explicit)} for {[m["id"] for m in members]}')
        if explicit:
            cid = explicit.pop()
        else:
            plain = sorted({m['id'] for m in members if '_' not in m['id']})
            if len(plain) != 1:
                raise ValueError(f'no canonical id for {[m["id"] for m in members]} (add it to CANON)')
            cid = plain[0]
        ordered = sorted(members, key=lambda m: [p for p, _ in PASSES].index(m['pass']))
        lead = ordered[0]
        urls, seen = [], set()
        for m in ordered:
            for u in m['urls']:
                if norm_url(u) not in seen:
                    seen.add(norm_url(u))
                    urls.append(u)
        given_pub = next((m['publisher'] for m in ordered if m['publisher']), None)
        publisher = publisher_of(urls[0], given_pub)
        l14 = [m for m in ordered if m['pass'] != 'L57']
        title = l14[0]['title'] if l14 else lead['title']
        if not l14 and not title.lower().startswith(publisher.lower()):
            title = f'{publisher} - {title}'
        readings = []
        for m in ordered:
            r = {'pass': m['pass'], 'id': m['id'], 'status': m['status']}
            if m['how']:
                r['how'] = m['how']
            readings.append(r)
        statuses = [STATUS[m['status']] for m in ordered]
        aliases = set()
        for m in ordered:
            aliases.update((m['id'], m['id'].replace('_', '-')))
        aliases.update(EXTRA_ALIASES.get(cid, ()))
        aliases.discard(cid)
        notes = []
        for m in ordered:
            if m['notes'] and m['notes'] not in notes:
                notes.append(m['notes'])
        sources.append({'id': cid, 'title': title, 'publisher': publisher, 'url': urls[0], 'extraUrls': urls[1:],
                        'accessed': max(m['accessed'] for m in ordered),
                        'read': max(statuses, key=READ_ORDER.get), 'readings': readings,
                        'aliases': sorted(aliases), 'notes': notes})
        for a in list(aliases)+[cid]:
            if alias_map.get(a, cid) != cid:
                raise ValueError(f'id {a} names two sources ({alias_map[a]}, {cid})')
            alias_map[a] = cid
    ids = [s['id'] for s in sources]
    if len(set(ids)) != len(ids):
        raise ValueError('two groups got the same canonical id')
    sources.sort(key=lambda s: s['id'])
    return sources, alias_map


def unreadable(research, alias_map, sources):
    """What the passes could not verify, with canonical ids (the level 5-7 pass names URLs: mapped through the
    registry when the URL is in it)."""
    by_url = {}
    for s in sources:
        for u in [s['url']]+s['extraUrls']:
            by_url[norm_url(u)] = s['id']
    out = []
    for name, data, _raw in research:
        for item in data.get('could_not_verify') or []:
            out.append({'pass': name, 'text': item['text'],
                        'sources': sorted({alias_map[i] for i in item.get('sources') or []})})
        for item in data.get('unverified') or []:
            ids = sorted({by_url[norm_url(u)] for u in item.get('urls') or [] if norm_url(u) in by_url})
            out.append({'pass': name, 'text': f'{item["what"]} ({item["why"]})', 'sources': ids})
    return out


def build(directory=RESEARCH_DIR, research=None):
    """The sources.json document (a dict) built from the research copies."""
    research = research if research is not None else load_research(directory)
    sources, alias_map = merge(research)
    inputs = []
    for (name, fname), (_n, data, raw) in zip(PASSES, research):
        inputs.append({'pass': name, 'file': f'tools/annotations/coaching/research/{fname}',
                       'contentSha256': canonical_sha(raw), 'sources': len(data['sources']),
                       'items': sum(1 for _ in walk_items(data))})
    return {'kind': 'coaching-sources', 'version': 1, 'accessed': ACCESSED,
            'builtBy': 'tools/build_coaching_sources.py (do not edit by hand)', 'readRule': READ_RULE,
            'disclaimer': DISCLAIMER, 'inputs': inputs, 'sources': sources,
            'unreadable': unreadable(research, alias_map, sources)}


def dumps(doc):
    """ASCII JSON with CRLF line endings and a final newline (the repo's text convention)."""
    text = json.dumps(doc, indent=1, ensure_ascii=True, allow_nan=False)
    return (text.replace('\n', '\r\n')+'\r\n').encode('ascii')


def import_research(src_dir, dst_dir=RESEARCH_DIR):
    """Copy the three research files from src_dir, checked as ASCII JSON, written with CRLF line endings."""
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for _name, fname in PASSES:
        raw = (Path(src_dir)/fname).read_bytes()
        raw.decode('ascii')
        json.loads(raw)
        text = raw.decode('ascii').replace('\r\n', '\n').replace('\n', '\r\n')
        tmp = Path(dst_dir)/(fname+'.tmp')
        tmp.write_bytes(text.encode('ascii'))
        os.replace(tmp, Path(dst_dir)/fname)


def check(directory=RESEARCH_DIR, sources_path=SOURCES_PATH):
    """Problems (strings): sources.json differs from a fresh build, or profiles.json does not validate against it."""
    problems = []
    fresh = build(directory)
    try:
        current = json.loads(Path(sources_path).read_bytes())
    except (OSError, ValueError) as error:
        return [f'sources.json unreadable: {error}']
    if current != fresh:
        problems.append('sources.json is stale: run python tools/build_coaching_sources.py')
    sys.path.insert(0, str(ROOT))
    from soccer.coaching import profiles
    problems += profiles.validate_profiles(profiles.read_json(profiles.PROFILES_PATH), current,
                                           research=load_research(directory))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='exit 1 if sources.json is stale or profiles.json invalid')
    ap.add_argument('--import', dest='import_dir', help='copy the three research JSON files from this folder first')
    args = ap.parse_args(argv)
    if args.import_dir:
        import_research(args.import_dir)
    if args.check:
        problems = check()
        for p in problems:
            print('PROBLEM', p)
        print('OK' if not problems else f'{len(problems)} problem(s)')
        return 1 if problems else 0
    doc = build()
    tmp = SOURCES_PATH.with_suffix('.json.tmp')
    tmp.write_bytes(dumps(doc))
    os.replace(tmp, SOURCES_PATH)
    n_read = sum(1 for s in doc['sources'] if s['read'] != 'not_read')
    print(f'wrote {SOURCES_PATH.relative_to(ROOT)}: {len(doc["sources"])} sources ({n_read} read), '
          f'{len(doc["unreadable"])} unverifiable items')
    return 0


if __name__ == '__main__':
    sys.exit(main())
