"""Geometry: polyline resampling, discrete Frechet distance, point-to-polyline distance."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback without numba
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap if not (args and callable(args[0])) else args[0]


def resample(poly: np.ndarray, spacing: float, max_points: int = 200) -> np.ndarray:
    """Resample a polyline at (approximately) uniform arc-length spacing in meters."""
    if len(poly) < 2:
        return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total == 0:
        return poly[:1].copy()
    n = int(min(max_points, max(2, np.ceil(total / spacing) + 1)))
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])


@njit(cache=True)
def _frechet(P, Q):
    n, m = P.shape[0], Q.shape[0]
    ca = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            d = np.sqrt((P[i, 0] - Q[j, 0]) ** 2 + (P[i, 1] - Q[j, 1]) ** 2)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return ca[n - 1, m - 1]


def frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Frechet distance between two polylines (meters)."""
    return float(_frechet(np.ascontiguousarray(P, dtype=np.float64),
                          np.ascontiguousarray(Q, dtype=np.float64)))


def point_polyline_dist2(points: np.ndarray, poly: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Squared distance from each point (N, 2) to the polyline (k, 2), exact per segment."""
    if len(poly) == 1:
        return np.sum((points - poly[0]) ** 2, axis=1)
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None], axis=2) / ab2[None], 0.0, 1.0)
        proj = a[None] + t[..., None] * ab[None]
        d2 = np.sum((p[:, None, :] - proj) ** 2, axis=2)
        out[s:s + chunk] = d2.min(axis=1)
    return out
