"""Incentive mechanism: diversity weights (Sec. III), Stage 1 targets (Sec. IV),
payments and utilities (Eq. payment/utility), and the Theorem 2 certificate (Sec. VI)."""
from __future__ import annotations

import numpy as np

from .geometry import frechet


def diversity_weights(polys_resampled, rho: float, sigma: float = 1.0, self_term: bool = True):
    """Location diversity x_i (Eq. diversity).

    self_term=True : x_i = 1 / (sigma * [1 + sum_{j != i} exp(-dF_ij / rho)])   (journal version)
    self_term=False: x_i = 1 / (sigma * sum_{j != i} exp(-dF_ij / rho))         (ICCVE 2022)
    rho is a fixed length in meters. Returns (x, dF matrix).
    """
    m = len(polys_resampled)
    dF = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            dF[i, j] = dF[j, i] = frechet(polys_resampled[i], polys_resampled[j])
    K = np.exp(-dF / rho)
    np.fill_diagonal(K, 0.0)
    denom = K.sum(axis=1) + (1.0 if self_term else 0.0)
    if np.any(denom <= 0):
        raise ValueError("Original diversity is undefined for a vehicle with no other routes; use self_term=True")
    return 1.0 / (sigma * denom), dF


def stage1_targets(c, x, R):
    """Algorithm 1: unique equilibrium of the continuous deviation game (Theorem 1).

    Returns dict with t (targets, original indexing), active (bool mask), K_H, h, S_star,
    a (a_i for active vehicles, nan otherwise), u_star (equilibrium utilities).
    """
    c = np.asarray(c, float)
    x = np.asarray(x, float)
    M = len(c)
    if M < 2:
        raise ValueError("Stage 1 requires M >= 2")
    k = c / x
    pi = np.argsort(k, kind="stable")             # fixed tie-breaking
    ks = k[pi]
    h, S = 2, ks[0] + ks[1]                        # running sum
    while h < M and ks[h] < (ks[h] + S) / h:
        S += ks[h]
        h += 1
    K_H = S
    t = np.zeros(M)
    a = np.full(M, np.nan)
    active = np.zeros(M, bool)
    for pos in range(h):
        i = pi[pos]
        ai = (h - 1) * k[i] / K_H
        t[i] = ((h - 1) * R / K_H) * (1 - ai) / x[i]
        a[i] = ai
        active[i] = True
    S_star = (h - 1) * R / K_H
    u_star = np.where(active, R * (1 - np.nan_to_num(a)) ** 2, 0.0)
    return dict(t=t, active=active, K_H=K_H, h=h, S_star=S_star, a=a, k=k, u_star=u_star)


def payments_utilities(t, c, x, R):
    """Realized payments (Eq. payment) and utilities (Eq. utility)."""
    t = np.asarray(t, float)
    S = float(np.dot(x, t))
    P = (x * t / S) * R if S > 0 else np.zeros_like(t)
    return P, P - np.asarray(c) * t


def certificate(t_real, s1, x, c, R):
    """Theorem 2: epsilon bounds from realized gaps e_i = |t_real_i - t*_i|.

    Returns dict with gaps, Delta, condition (Delta <= S*/2), eps (per vehicle; nan if the
    condition fails), and the participation check of Corollary 2 for active vehicles.
    """
    t_star = s1["t"]
    e = np.abs(np.asarray(t_real, float) - t_star)
    Delta = float(np.dot(x, e))
    S_star = s1["S_star"]
    cond = Delta <= S_star / 2.0
    eps = np.full(len(e), np.nan)
    ir_ok = np.full(len(e), np.nan)
    if cond:
        for i in range(len(e)):
            if s1["active"][i]:
                eps[i] = (R / S_star) * (Delta / s1["a"][i] + 2 * Delta + 2 * x[i] * e[i]) + c[i] * e[i]
                rhs = (2 * R / S_star) * (x[i] * e[i] + Delta) + c[i] * e[i]
                ir_ok[i] = float(R * (1 - s1["a"][i]) ** 2 > rhs)
            else:
                eps[i] = R * (Delta / S_star) ** 2
    # Proposition 3 (second-order refinement) for active vehicles meeting Eq. (secondcond)
    eps2 = np.full(len(e), np.nan)
    if cond:
        x = np.asarray(x, float); t_real = np.asarray(t_real, float)
        s_hat = x * t_real
        S_hat = s_hat.sum()
        for i in np.flatnonzero(s1["active"]):
            a = s1["a"][i]
            A_star = a * S_star
            if Delta <= 0.5 * min(1.0, (1 - a * a) / (a * a)) * A_star:
                A_hat = S_hat - s_hat[i]
                T = min(S_hat, np.sqrt(R * A_hat / s1["k"][i]))
                L = max(1.0, 1.0 / (np.sqrt(2.0) * a))
                eps2[i] = (R / T ** 2) * (x[i] * e[i] + L * Delta) ** 2
    return dict(gaps=e, Delta=Delta, condition=cond, eps=eps, ir_ok=ir_ok, eps2=eps2)


def compensation_needed(u):
    """Extra payment needed to give every vehicle non-negative utility: sum of max(0, -u_i)."""
    return float(np.sum(np.maximum(0.0, -np.asarray(u, float))))


def best_response_gain(t_real, c, x, R):
    """Exact maximum gain of each vehicle from any unilateral change of deviation t_i >= 0
    (Lemma 3 value minus current utility). Upper-bounds the gain from any route change."""
    t_real = np.asarray(t_real, float)
    _, u = payments_utilities(t_real, c, x, R)
    k = np.asarray(c) / np.asarray(x)
    s = np.asarray(x) * t_real
    gains = np.empty(len(t_real))
    for i in range(len(t_real)):
        A = s.sum() - s[i]
        V = R if A <= 0 else (max(0.0, np.sqrt(R) - np.sqrt(k[i] * A)) ** 2 if k[i] * A < R else 0.0)
        gains[i] = V - u[i]
    return gains
