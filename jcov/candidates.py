"""Stage 2 candidate routes (Sec. V-A): pool of simple via-routes, tolerance filter,
deterministic Frechet pruning, and the fallback route."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import frechet, resample


@dataclass
class VehicleRoutes:
    s: int
    d: int
    default: list
    L: float
    dist_s: np.ndarray = field(repr=False)
    pred_s: np.ndarray = field(repr=False)
    dist_d: np.ndarray = field(repr=False)
    pred_d: np.ndarray = field(repr=False)


def prepare_vehicle(net, s: int, d: int) -> VehicleRoutes:
    dist_s, pred_s = net.dijkstra(s)
    dist_d, pred_d = net.dijkstra(d, reverse=True)
    default = net.path_from_pred(pred_s, s, d)
    if not default:
        raise ValueError(f"no route from {s} to {d}")
    return VehicleRoutes(s, d, default, float(dist_s[d]), dist_s, pred_s, dist_d, pred_d)


def via_route(net, vr: VehicleRoutes, v: int):
    first = net.path_from_pred(vr.pred_s, vr.s, v)
    second = net.path_to_target(vr.pred_d, v, vr.d)
    if not first or not second:
        return None
    return first + second[1:]


def is_simple(route) -> bool:
    return len(set(route)) == len(route)


def build_candidates(net, vr: VehicleRoutes, target: float, D: float, delta: float,
                     K: int, eta: float, resample_m: float):
    """Return (candidates, fallback, info).

    candidates: list of (route, deviation) within the tolerance, pruned, at most K
    fallback:   (route, deviation) closest to the target in the simple, detour-feasible pool
                (always exists: the pool contains the default route)
    The pool is the default route plus all simple via-routes with deviation <= D.
    Routes are processed in increasing realization gap, ties by junction index.
    """
    total = vr.dist_s + vr.dist_d
    dev = total - vr.L
    feasible = np.flatnonzero(np.isfinite(dev) & (dev <= D + 1e-9))
    gap = np.abs(dev[feasible] - target)
    order = feasible[np.lexsort((feasible, gap))]           # by gap, then junction index
    # pool entries: (gap, tie-break, junction or None for the default route)
    entries = [(abs(target), -1, None)] + [(float(abs(dev[v] - target)), int(v), int(v)) for v in order]
    entries.sort(key=lambda z: (z[0], z[1]))
    seen = set()
    candidates, kept_shapes = [], []
    fallback = None
    n_simple = 0
    for g, _, v in entries:
        if v is None:
            route, rdev = vr.default, 0.0
        else:
            route = via_route(net, vr, v)
            if route is None or not is_simple(route):
                continue
            rdev = float(dev[v])
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        n_simple += 1
        if fallback is None:          # first simple route in gap order = argmin over the pool
            fallback = (route, rdev)
        if g > delta or len(candidates) >= K:
            break                     # sorted by gap: nothing further is within tolerance
        shape = resample(net.route_polyline(route), resample_m)
        if any(frechet(shape, ks) < eta for ks in kept_shapes):
            continue
        candidates.append((route, rdev))
        kept_shapes.append(shape)
    info = dict(n_pool_scanned=n_simple, n_candidates=len(candidates))
    return candidates, fallback, info
