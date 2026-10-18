"""Catalogue of time scales used in the paper (all closed subsets of R).

Finite unions of compact intervals and points satisfy the finite-gap assumption; the accumulation-point
scale q^{N0} U {0} is handled separately by truncation (see `qacc_solve`)."""
from __future__ import annotations
import numpy as np
from .timescale import TimeScale
from .problems import ts_hybrid, ts_interval, ts_pendulum, ts_qscale, ts_family


def cantor(level=4, a=0.0, b=3.0):
    """level-k middle-thirds prefractal of [a,b]: 2^k intervals, 2^k-1 geometrically shrinking gaps"""
    iv = [(a, b)]
    for _ in range(level):
        new = []
        for p, q in iv:
            d = (q - p) / 3; new += [(p, p + d), (q - d, q)]
        iv = new
    return TimeScale(iv)


def pab(alpha=0.3, beta=0.2, kmax=5):
    """P_{alpha,beta} = U_{k=0..kmax} [k(alpha+beta), k(alpha+beta)+alpha]"""
    return TimeScale([(k * (alpha + beta), k * (alpha + beta) + alpha) for k in range(kmax + 1)])


def random_scale(seed=2026):
    """random union of intervals and isolated points (the scale T_rnd of the paper for seed 2026)"""
    rng = np.random.default_rng(seed)
    comps = [(0.0, 0.35)]; pos = 0.35
    for _ in range(4):
        for _ in range(int(rng.integers(1, 4))):
            pos += rng.uniform(0.08, 0.35); comps.append((pos, pos))
        pos += rng.uniform(0.08, 0.35); ln = rng.uniform(0.2, 0.6); comps.append((pos, pos + ln)); pos += ln
    return TimeScale([(round(p, 6), round(q, 6)) for p, q in comps])


def harmonic(n=400):
    """{H_k : 1 <= k <= n}, mu(H_k) = 1/(k+1)"""
    return TimeScale.from_points(np.cumsum(1.0 / np.arange(1, n + 1)))


def sqrt_scale(n=400):
    """{sqrt(k) : 0 <= k <= n}"""
    return TimeScale.from_points(np.sqrt(np.arange(0, n + 1)))


def irregular():
    """T_irr = [0,0.45] U {0.70,1.03,1.48} U [1.90,2.30] U {2.75} U [3.20,3.55]"""
    return TimeScale([(0, .45), (.70, .70), (1.03, 1.03), (1.48, 1.48), (1.90, 2.30), (2.75, 2.75), (3.20, 3.55)])


SCALES = {
    "interval": lambda: ts_interval(0.0, 2.0),
    "hybrid": ts_hybrid,
    "pendulum": ts_pendulum,
    "qscale": ts_qscale,
    "irr": irregular,
    "cantor": cantor,
    "pab": pab,
    "random": random_scale,
    "harmonic": harmonic,
    "sqrt": sqrt_scale,
}


def get(name, **kw):
    if name.startswith("family"):
        return ts_family(float(name.split(":")[1]) if ":" in name else 0.5)
    if name not in SCALES:
        raise KeyError(f"unknown scale {name!r}; choose from {sorted(SCALES)} , family:<nu> or qacc")
    return SCALES[name](**kw)


def qacc_solve(f, K, q=0.9, n=160):
    """forward recursion on {q^n,...,q,1} of q^{N0} U {0}; the integral over [0,q^n) is dropped.
    Truncation bound (Proposition 8.1): max|X - x| <= K_inf q^n prod(1 + L mu_j)."""
    t = q ** np.arange(n, -1, -1.0); mu = np.concatenate([np.diff(t), [0.0]]); X = np.empty(n + 1)
    for i in range(n + 1):
        X[i] = f(t[i]) + np.sum(mu[:i] * K(t[i], t[:i], t[:i] + mu[:i], X[:i]))
    return t, X, mu
