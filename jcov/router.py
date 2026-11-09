"""Stage 2 selection rules over common candidate sets (Sec. V-D, V-E):
greedy coverage-aware routing (Algorithm 2), exact ILP, closest-length, random."""
from __future__ import annotations

import time

import numpy as np


def _base(phi_fixed, grid):
    if len(phi_fixed) == 0:
        return np.full(len(grid.area), grid.dbar2)
    return np.min(np.vstack(phi_fixed), axis=0)


def greedy(cands, phis, gaps, phi_fixed, grid):
    """Algorithm 2. cands: {vehicle: [route idx...]}, phis: {vehicle: (n_cand, C) array},
    gaps: {vehicle: array of realization gaps}. Returns {vehicle: chosen candidate index}."""
    cur = _base(phi_fixed, grid).copy()
    left = set(cands)
    choice = {}
    while left:
        best = None
        for i in sorted(left):
            g = (np.maximum(cur[None, :] - phis[i], 0.0) * grid.area[None, :]).sum(axis=1)
            for r in range(len(g)):
                key = (-g[r], gaps[i][r], i, r)          # max gain, then smallest gap
                if best is None or key < best:
                    best = key
        _, _, i, r = best
        choice[i] = r
        cur = np.minimum(cur, phis[i][r])
        left.remove(i)
    return choice


def local_search(choice, phis, gaps, phi_fixed, grid, max_rounds: int = 50):
    """Single-vehicle swap improvement: repeatedly give each vehicle its best candidate given
    the others' choices, until no swap raises F. F never decreases, so a greedy start keeps
    the 1/2 guarantee of Proposition prop:greedy. Ties keep the current route."""
    base = _base(phi_fixed, grid)
    choice = dict(choice)
    for _ in range(max_rounds):
        improved = False
        for i in sorted(choice):
            others = [phis[j][r] for j, r in choice.items() if j != i]
            cur = np.min(np.vstack([base] + others), axis=0) if others else base
            g = (np.maximum(cur[None, :] - phis[i], 0.0) * grid.area[None, :]).sum(axis=1)
            r_best = int(np.lexsort((gaps[i], -g))[0])
            if g[r_best] > g[choice[i]] * (1 + 1e-12) + 1e-9:
                choice[i] = r_best
                improved = True
        if not improved:
            break
    return choice


def greedy_ls(cands, phis, gaps, phi_fixed, grid):
    """Algorithm 2 followed by single-vehicle swap local search."""
    return local_search(greedy(cands, phis, gaps, phi_fixed, grid), phis, gaps, phi_fixed, grid)


def closest_length(cands, gaps):
    return {i: int(np.argmin(gaps[i])) for i in cands}


def random_choice(cands, rng):
    return {i: int(rng.integers(len(cands[i]))) for i in sorted(cands)}


def coverage_gain(choice, phis, phi_fixed, grid):
    base = _base(phi_fixed, grid)
    cur = base.copy()
    for i, r in choice.items():
        cur = np.minimum(cur, phis[i][r])
    return float(np.sum(grid.area * (base - cur)))


def ilp(cands, phis, phi_fixed, grid, time_limit: float = 300.0, mip_rel_gap: float = 0.0):
    """Exact facility-location ILP (Eq. ilp) with scipy/HiGHS. y binary, z continuous.
    Cells whose weights are all zero are dropped. Returns (choice, objective, status)."""
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if not cands:                                              # nothing to select
        return {}, 0.0, 0
    base = _base(phi_fixed, grid)
    pairs = [(i, r) for i in sorted(cands) for r in range(len(cands[i]))]
    W = np.vstack([np.maximum(base - phis[i][r], 0.0) * grid.area for i, r in pairs])  # (N, C)
    keep = np.flatnonzero(W.max(axis=0) > 0)
    W = W[:, keep]
    N, C = W.shape
    ny, nz = N, N * C
    c = np.concatenate([np.zeros(ny), -W.ravel()])            # milp minimizes
    rows, cols, vals, lb, ub = [], [], [], [], []
    r0 = 0
    for i in sorted(cands):                                    # one route per vehicle
        idx = [p for p, (j, _) in enumerate(pairs) if j == i]
        rows += [r0] * len(idx); cols += idx; vals += [1.0] * len(idx)
        lb.append(1.0); ub.append(1.0); r0 += 1
    for cc in range(C):                                        # each cell served at most once
        idx = [ny + p * C + cc for p in range(N)]
        rows += [r0] * N; cols += idx; vals += [1.0] * N
        lb.append(-np.inf); ub.append(1.0); r0 += 1
    for p in range(N):                                         # z <= y
        for cc in range(C):
            rows += [r0, r0]; cols += [ny + p * C + cc, p]; vals += [1.0, -1.0]
            lb.append(-np.inf); ub.append(0.0); r0 += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r0, ny + nz)).tocsr()
    integrality = np.concatenate([np.ones(ny), np.zeros(nz)])
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=(0, 1), options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap})
    if res.x is None:
        return None, np.nan, res.status
    y = res.x[:ny]
    choice = {}
    for p, (i, r) in enumerate(pairs):
        if y[p] > 0.5:
            choice[i] = r
    return choice, float(-res.fun), res.status


def timed(f, *a, **k):
    t0 = time.perf_counter()
    out = f(*a, **k)
    return out, time.perf_counter() - t0
