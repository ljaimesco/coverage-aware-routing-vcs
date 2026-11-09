"""Unit tests: run with  python -m pytest tests -q  (from the repo root)."""
import itertools
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jcov.geometry import frechet, point_polyline_dist2, resample  # noqa: E402
from jcov.mechanism import (best_response_gain, certificate, diversity_weights,  # noqa: E402
                            payments_utilities, stage1_targets)
from jcov.router import coverage_gain, greedy, ilp  # noqa: E402


def test_stage1_is_equilibrium_and_utilities():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = rng.integers(2, 9)
        c, x, R = rng.uniform(0.2, 3, M), rng.uniform(0.1, 1, M), rng.uniform(10, 1000)
        s1 = stage1_targets(c, x, R)
        t = s1["t"]
        _, u = payments_utilities(t, c, x, R)
        assert np.allclose(u, s1["u_star"], atol=1e-8)
        for i in range(M):                      # no profitable unilateral deviation
            def f(ti):
                tt = t.copy(); tt[i] = ti
                return -payments_utilities(tt, c, x, R)[1][i]
            best = -minimize_scalar(f, bounds=(0, 10 * max(t.max(), 1)), method="bounded").fun
            assert best <= u[i] + 1e-6 * max(1, abs(u[i]))
        assert np.isclose(np.dot(x, t), s1["S_star"])          # Proposition 1(c)
        s2 = stage1_targets(c, x, 2 * R)
        assert np.allclose(s2["t"], 2 * t) and (s2["active"] == s1["active"]).all()  # Prop. 1(a)


def test_certificate_bounds_exact_gains():
    rng = np.random.default_rng(1)
    for _ in range(500):
        M = rng.integers(2, 9)
        c, x, R = rng.uniform(0.2, 3, M), rng.uniform(0.1, 1, M), rng.uniform(10, 1000)
        s1 = stage1_targets(c, x, R)
        dmax = s1["S_star"] / (2 * x[s1["active"]].sum())
        delta = rng.uniform(0, 1) * dmax
        t = s1["t"].copy()
        t[s1["active"]] = np.maximum(0, t[s1["active"]] + rng.uniform(-delta, delta, s1["active"].sum()))
        cert = certificate(t, s1, x, c, R)
        assert cert["condition"]
        gains = best_response_gain(t, c, x, R)
        assert np.all(gains <= cert["eps"] + 1e-9)


def test_diversity_bounds():
    rng = np.random.default_rng(2)
    polys = [np.cumsum(rng.normal(size=(20, 2)), axis=0) * 100 for _ in range(6)]
    x, _ = diversity_weights(polys, rho=200.0, sigma=1.0, self_term=True)
    assert np.all(x <= 1.0 + 1e-12) and np.all(x >= 1.0 / 6 - 1e-12)


def test_geometry():
    P = np.array([[0.0, 0.0], [10.0, 0.0]])
    pts = np.array([[5.0, 3.0], [-4.0, 0.0], [13.0, 4.0]])
    assert np.allclose(point_polyline_dist2(pts, P), [9.0, 16.0, 25.0])
    Q = P + np.array([0.0, 2.0])
    assert np.isclose(frechet(resample(P, 1.0), resample(Q, 1.0)), 2.0)


class _Grid:
    def __init__(self, n):
        self.area = np.ones(n)
        self.dbar2 = 100.0


def test_greedy_half_and_ilp_optimal():
    rng = np.random.default_rng(3)
    for _ in range(40):
        C, M, K = 30, rng.integers(2, 5), rng.integers(2, 4)
        grid = _Grid(C)
        phis = {i: np.minimum(rng.uniform(0, 150, (K, C)), grid.dbar2) for i in range(M)}
        cands = {i: list(range(K)) for i in range(M)}
        gaps = {i: rng.uniform(0, 1, K) for i in range(M)}
        fixed = [np.minimum(rng.uniform(0, 150, C), grid.dbar2)]
        g = coverage_gain(greedy(cands, phis, gaps, fixed, grid), phis, fixed, grid)
        opt = max(coverage_gain(dict(zip(range(M), combo)), phis, fixed, grid)
                  for combo in itertools.product(range(K), repeat=M))
        assert g >= 0.5 * opt - 1e-9
        ch, obj, _ = ilp(cands, phis, fixed, grid)
        assert np.isclose(obj, opt, rtol=1e-6, atol=1e-6)
        assert np.isclose(coverage_gain(ch, phis, fixed, grid), opt, rtol=1e-6, atol=1e-6)


def test_second_order_bound():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(500):
        M = rng.integers(2, 9)
        c, x, R = rng.uniform(0.2, 3, M), rng.uniform(0.1, 1, M), rng.uniform(10, 1000)
        s1 = stage1_targets(c, x, R)
        act = s1["active"]
        delta = rng.uniform(0, 1) * s1["S_star"] / (8 * x[act].sum())
        t = s1["t"].copy()
        t[act] = np.maximum(0, t[act] + rng.uniform(-delta, delta, act.sum()))
        cert = certificate(t, s1, x, c, R)
        gains = best_response_gain(t, c, x, R)
        ok = np.isfinite(cert["eps2"])
        checked += ok.sum()
        assert np.all(gains[ok] <= cert["eps2"][ok] + 1e-9)
    assert checked > 100


def test_road_buffer_domain_and_nested_grids():
    from jcov.coverage import CoverageGrid
    road = np.column_stack([np.linspace(0, 1000, 41), np.full(41, 500.0)])   # horizontal road at y = 500
    bbox = (0, 0, 1000, 1000)
    full = CoverageGrid(bbox, 100)
    dom = CoverageGrid(bbox, 100, road_points=road, buffer_m=150)
    assert len(full.area) == 100 and dom.n_box_cells == 100
    assert np.all(np.abs(dom.centers[:, 1] - 500) <= 150)                  # rows y = 450, 550 only
    assert len(dom.area) == 20
    fine = CoverageGrid(bbox, 50)                                            # 50 m centers nest in 100 m cells
    c100 = set(map(tuple, np.floor(full.centers / 100)))
    assert set(map(tuple, np.floor(fine.centers / 100))) == c100


def test_dispersed_scenario_respects_od_bounds():
    from jcov.network import RoadNetwork
    from jcov.scenario import make_scenario
    n = 8                                                                    # 8 x 8 bidirectional grid, 250 m
    ids = [f"{i}_{j}" for i in range(n) for j in range(n)]
    index = {k: t for t, k in enumerate(ids)}
    coords = np.array([[250.0 * i, 250.0 * j] for i in range(n) for j in range(n)])
    out_adj = [[] for _ in ids]; in_adj = [[] for _ in ids]; shape = {}; elen = {}
    for i in range(n):
        for j in range(n):
            for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                if 0 <= i + di < n and 0 <= j + dj < n:
                    u, v = index[f"{i}_{j}"], index[f"{i + di}_{j + dj}"]
                    out_adj[u].append((v, 250.0)); in_adj[v].append((u, 250.0))
                    shape[(u, v)] = coords[[u, v]]; elen[(u, v)] = 250.0
    net = RoadNetwork(ids, index, coords, out_adj, in_adj, shape, elen)
    assert len(net.largest_scc()) == n * n
    ods = make_scenario(net, [{"dispersed": True, "count": 30}], 0, 600, np.random.default_rng(3), max_od=1200)
    dist = [np.linalg.norm(coords[s] - coords[d]) for s, d in ods]
    assert len(ods) == 30 and min(dist) >= 600 and max(dist) <= 1200
    assert net.road_points(25).shape[1] == 2


def test_local_search_never_lowers_gain_and_stays_within_ilp():
    from jcov.coverage import CoverageGrid
    from jcov.router import greedy_ls
    rng = np.random.default_rng(7)
    grid = CoverageGrid((0, 0, 1000, 1000), 100)
    for _ in range(30):
        M, K = rng.integers(2, 5), rng.integers(2, 5)
        cands = {i: list(range(K)) for i in range(M)}
        phis = {i: np.vstack([grid.phi(rng.uniform(0, 1000, (3, 2))) for _ in range(K)]) for i in range(M)}
        gaps = {i: rng.uniform(0, 50, K) for i in range(M)}
        fixed = [grid.phi(rng.uniform(0, 1000, (3, 2)))]
        Fg = coverage_gain(greedy(cands, phis, gaps, fixed, grid), phis, fixed, grid)
        Fl = coverage_gain(greedy_ls(cands, phis, gaps, fixed, grid), phis, fixed, grid)
        _, Fopt, _ = ilp(cands, phis, fixed, grid)
        assert Fl >= Fg - 1e-6 and Fl <= Fopt * (1 + 1e-6) + 1e-6
