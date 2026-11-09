"""Seeded origin-destination scenarios, reused across all methods and budgets (paired design)."""
from __future__ import annotations

import numpy as np


def make_scenario(net, groups, radius: float, min_od: float, rng, max_tries: int = 2000,
                  max_od: float | None = None):
    """Sample vehicles group by group.

    groups: list of {"origin_ref": junction_id, "dest_ref": junction_id, "count": n}
    Each vehicle's origin (destination) is a junction within `radius` meters of the group's
    origin (destination) reference junction. A group {"dispersed": true, "count": n} instead
    draws both ends uniformly from the largest strongly connected component, with straight-line
    distance in [min_od, max_od]; the pair must be connected and at least
    `min_od` meters apart in straight line. Returns a list of (s, d) node indices.
    """
    ods = []
    scc = None
    for g in groups:
        if g.get("dispersed"):
            if scc is None:
                scc = net.largest_scc()
            O = D = scc
        else:
            o_ref = net.coords[net.index[g["origin_ref"]]]
            d_ref = net.coords[net.index[g["dest_ref"]]]
            O = net.nodes_within(o_ref, radius)
            D = net.nodes_within(d_ref, radius)
            O = O[[len(net.out_adj[i]) > 0 for i in O]]
            D = D[[len(net.in_adj[i]) > 0 for i in D]]
        n_ok, tries = 0, 0
        while n_ok < g["count"]:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"could not sample {g['count']} connected OD pairs for group {g}")
            s, d = int(rng.choice(O)), int(rng.choice(D))
            sl = np.linalg.norm(net.coords[s] - net.coords[d])
            if s == d or sl < min_od or (g.get("dispersed") and max_od is not None and sl > max_od):
                continue
            dist, _ = net.dijkstra(s)
            if not np.isfinite(dist[d]):
                continue
            ods.append((s, d))
            n_ok += 1
    return ods


def draw_costs(M: int, spec: dict, rng):
    """Per-meter costs c_i: {"type": "equal", "value": v} or {"type": "lognormal", "mean": m, "sigma": s}."""
    if spec["type"] == "equal":
        return np.full(M, float(spec["value"]))
    if spec["type"] == "lognormal":
        mu = np.log(spec["mean"]) - spec["sigma"] ** 2 / 2       # so that E[c] = mean
        return rng.lognormal(mu, spec["sigma"], M)
    raise ValueError(spec)
