"""A frozen copy of soccer/ball_path.decode as it was before the vectorized rewrite (speed pass 3), for the
side-by-side tests: the new decode must return exactly this path (the same nodes, the same link costs and
speeds bit for bit) on every input. Not used by the pipeline."""
import math

import numpy as np

from soccer.ball_path import (DEAD_STOP, END_COST, FOOT_ENTER, FOOT_FADE, FOOT_START, FOOT_SWITCH, JUMP_COST, JUMP_SPEED, MAX_SKIP_S, MAX_STILL_S,
                              NULL_COST, SKIP_COST, START_COST, STILL_SPEED, TRAVEL_SPEED, UNRELIABLE, V0, _chain)


def decode_reference(layers, times, motions, fps):
    """Viterbi over the layers. Returns per frame either None (not visible) or (node, link) where link
    is {'from': frame index, 'node': Node, 'cost': motion cost, 'speed': m/s, 'skipped': bridged frames}
    or None for a start. A link may bridge up to MAX_SKIP_S of unseen frames, or up to MAX_STILL_S when
    the speed it implies says the ball stayed where it was (STILL_SPEED). A flight into or out of a ball
    standing still with nobody near it (DEAD_STOP), or faster than a shot between two different things
    (JUMP_COST), is paid for: the ball does neither by itself."""
    n = len(layers)
    T, breaks, unreliable = _chain(motions)
    cuts = np.cumsum(np.asarray(breaks, int))
    shaky = np.cumsum(np.asarray(unreliable, int))
    K = max(1, int(round(MAX_SKIP_S*fps)))
    K_still = max(K, int(round(MAX_STILL_S*fps)))
    best_null = [math.inf]*n
    back_null = [None]*n          # ('null', i-1) or ('node', i-1, j)
    best = [np.full(len(l), math.inf) for l in layers]
    back = [[None]*len(l) for l in layers]
    link = [[None]*len(l) for l in layers]
    # Node positions in the coordinates of frame 0: a link from h to i maps them through T[i].
    origin = [np.array([(m.x, m.y, 1.0) for m in l]) @ np.linalg.inv(T[i]).T if l else None for i, l in enumerate(layers)]
    for i in range(n):
        # Not visible at i.
        if i == 0:
            best_null[i], back_null[i] = NULL_COST, None
        else:
            options = [(best_null[i-1], ('null', i-1))]
            if len(layers[i-1]):
                j = int(np.argmin(best[i-1]))
                options.append((best[i-1][j]+END_COST, ('node', i-1, j)))
            best_null[i], back_null[i] = min(options, key=lambda o: o[0])
            best_null[i] += NULL_COST
        if not layers[i]:
            continue
        xy = np.array([(m.x, m.y) for m in layers[i]])
        pxm = np.array([m.pxm for m in layers[i]])
        start = START_COST if i == 0 else best_null[i-1]+START_COST
        for j, m in enumerate(layers[i]):
            best[i][j], back[i][j] = start+(FOOT_START if m.pid is not None else 0.0), ('null', i-1)
        cols = np.arange(len(layers[i]))
        for k in range(1, K_still+1):
            h = i-k
            if h < 0 or cuts[i]-cuts[h] > 0:
                break
            if not layers[h] or not np.isfinite(best[h]).any():
                continue
            prev = origin[h] @ T[i].T
            prev = prev[:, :2]/prev[:, 2:3]
            dt = max(1e-3, times[i]-times[h])
            dist = np.hypot(prev[:, None, 0]-xy[None, :, 0], prev[:, None, 1]-xy[None, :, 1])
            scale = (np.array([m.pxm for m in layers[h]])[:, None]+pxm[None, :])/2
            speed = dist/np.maximum(scale, 1e-3)/dt
            if k > K:
                still = speed <= STILL_SPEED
                if not still.any():
                    continue
            # The displacement an unseen gap must explain grows dearer with the gap: a long flight that
            # nobody saw needs more than a plausible average speed.
            motion = (speed/V0)**2*(1+dt)+(k-1)*SKIP_COST+UNRELIABLE*(shaky[i]-shaky[h])
            cost = motion.copy()
            same = np.zeros(cost.shape, bool)
            if k == 1:
                same = np.array([[ma.tracklet is not None and ma.tracklet == mb.tracklet for mb in layers[i]] for ma in layers[h]], bool).reshape(cost.shape)
                cost = np.where(same, np.minimum(cost, .05+UNRELIABLE*unreliable[i]), cost)
            # Arriving at a player's lower body: free when the causal pass saw it arrive or the ball was
            # already with that player; dear when it comes from another player's feet or reach with
            # nothing seen travelling (an unseen pass stays NOT VISIBLE; a boot never becomes the ball).
            near_a = [ma.near for ma in layers[h]]
            enter = np.array([[mb.pid is not None and na != mb.pid for mb in layers[i]] for na in near_a], bool).reshape(cost.shape) & ~same
            from_feet = np.array([na is not None for na in near_a], bool)[:, None]
            cost = cost+enter*FOOT_ENTER+(enter & from_feet)*(FOOT_SWITCH*max(0.0, 1-dt/FOOT_FADE))
            # A flight cannot leave a ball standing still, nor end on one, with no player within reach of it: the
            # ball does not start from rest or stop dead on the grass by itself.
            if k > 1 or not same.all():
                still_a = np.array([ma.own_speed is not None and ma.own_speed < STILL_SPEED and ma.near is None for ma in layers[h]], bool)[:, None]
                still_b = np.array([mb.own_speed is not None and mb.own_speed < STILL_SPEED and mb.near is None for mb in layers[i]], bool)[None, :]
                flight = speed >= TRAVEL_SPEED
                cost = cost+np.where(flight & (still_a | still_b) & ~same, DEAD_STOP, 0.0)
                # Faster than a hard shot between sightings of two different things: not the same ball.
                cost = cost+np.where((speed > JUMP_SPEED) & ~same, JUMP_COST, 0.0)
            if k > K:
                cost = np.where(still, cost, np.inf)
            total = best[h][:, None]+cost
            a_best = np.argmin(total, axis=0)
            t_best = total[a_best, cols]
            for b in np.flatnonzero(t_best < best[i]):
                a = int(a_best[b])
                best[i][b], back[i][b] = float(t_best[b]), ('node', h, a)
                link[i][b] = {'from': h, 'node': layers[h][a], 'cost': float(motion[a, b] if k == 1 else cost[a, b]), 'speed': float(speed[a, b]), 'skipped': k-1}
        for j, m in enumerate(layers[i]):
            best[i][j] += -math.log(m.q)
    # Backtrack.
    path = [None]*n
    if n == 0:
        return path
    state = ('null', n-1)
    if len(layers[n-1]):
        j = int(np.argmin(best[n-1]))
        if best[n-1][j] < best_null[n-1]:
            state = ('node', n-1, j)
    while state is not None:
        if state[0] == 'null':
            i = state[1]
            if i < 0:
                break
            state = back_null[i]
        else:
            _, i, j = state
            path[i] = (layers[i][j], link[i][j])
            state = back[i][j]
    return path
