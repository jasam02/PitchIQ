"""Kit colour names from the team jersey prototypes the tracker learned (soccer/appearance: 4 grey bins at lightness
.1 / .38 / .65 / .92, then 10 hue bins x 2 brightness levels), so the report can say "Team A (white)".

Measured on CAN-ISL: prototype A has 96% of its mass in the grey bins, most of it the lightest (white); B 92% in the two
red hue bins (9 and 0). Every one of the 33 segments' frames agreed (B's shirts redder than A's in all 33)."""

GREYS = (.1, .38, .65, .92)
HUE_NAMES = ('red', 'orange', 'yellow', 'green', 'green', 'light blue', 'blue', 'purple', 'pink', 'red')


def kit_name(prototype):
    """'white', 'red', 'light blue', ... or None when the prototype is empty or unclear."""
    if not prototype or len(prototype) < 24:
        return None
    h = [float(v) for v in prototype[:24]]
    total = sum(h)
    if total <= 0:
        return None
    grey = sum(h[:4])/total
    if grey >= .6:
        light = sum(h[i]*GREYS[i] for i in range(4))/max(sum(h[:4]), 1e-9)
        return 'white' if light >= .75 else 'light grey' if light >= .5 else 'grey' if light >= .25 else 'black'
    names = {}
    for k in range(10):
        names[HUE_NAMES[k]] = names.get(HUE_NAMES[k], 0.0)+h[4+2*k]+h[5+2*k]
    name, mass = max(names.items(), key=lambda kv: kv[1])
    if mass/total < .35:
        return None
    dark = sum(h[4+2*k] for k in range(10) if HUE_NAMES[k] == name)
    bright = sum(h[5+2*k] for k in range(10) if HUE_NAMES[k] == name)
    return f'dark {name}' if dark > 2*bright and name not in ('red',) else name


def team_kits(index_or_result):
    """{'A': name, 'B': name} from a match index ('teams': {'prototypes': {A, B}}) or a run's result ('teams': {a, b})."""
    teams = (index_or_result or {}).get('teams') or {}
    protos = teams.get('prototypes') or {'A': teams.get('a'), 'B': teams.get('b')}
    out = {t: kit_name(protos.get(t)) for t in 'AB'}
    if out['A'] and out['A'] == out['B']:
        return {'A': None, 'B': None}
    return out
