"""Compact time scales with finitely many gaps and partitions of them.

A compact time scale piece [a,b]_T is stored as a finite, strictly ordered list
of closed components [p_i, q_i] (p_i <= q_i, q_i < p_{i+1}).  Degenerate
components (p_i = q_i) are isolated points.  Every q_i with i < m is
right-scattered with sigma(q_i) = p_{i+1} and graininess g_i = p_{i+1} - q_i.
All other points are right-dense.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


class TimeScale:
    def __init__(self, components):
        comps = sorted((float(p), float(q)) for p, q in components)
        for p, q in comps:
            if q < p:
                raise ValueError("component with q < p")
        for (p1, q1), (p2, q2) in zip(comps[:-1], comps[1:]):
            if not q1 < p2:
                raise ValueError("components must be disjoint and ordered")
        self.comps = comps
        self.p = np.array([c[0] for c in comps])
        self.q = np.array([c[1] for c in comps])
        self.a = comps[0][0]
        self.b = comps[-1][1]

    # ------------------------------------------------------------------ basics
    @classmethod
    def from_points(cls, pts):
        pts = np.unique(np.asarray(pts, float))
        return cls([(t, t) for t in pts])

    @property
    def m(self):
        return len(self.comps)

    def gaps(self):
        """list of (q_i, p_{i+1}) gaps"""
        return [(self.q[i], self.p[i + 1]) for i in range(self.m - 1)]

    def dense_length(self):
        return float(np.sum(self.q - self.p))

    def component_of(self, t):
        i = int(np.searchsorted(self.q, t - 1e-15 * max(1.0, abs(t)), side="left"))
        i = min(i, self.m - 1)
        if not (self.p[i] - 1e-12 <= t <= self.q[i] + 1e-12):
            raise ValueError(f"{t} is not in the time scale")
        return i

    def contains(self, t, tol=1e-12):
        i = int(np.searchsorted(self.q, t - tol, side="left"))
        return i < self.m and self.p[i] - tol <= t <= self.q[i] + tol

    def sigma(self, t):
        i = self.component_of(t)
        if t < self.q[i] - 1e-14 or i == self.m - 1:
            return float(t)
        return float(self.p[i + 1])

    def mu(self, t):
        return self.sigma(t) - t

    def is_left_dense(self, t):
        i = self.component_of(t)
        return t > self.p[i] + 1e-14 or (i == 0 and False)

    def sample(self, n_per_unit=200, include_isolated=True):
        """points of T: uniform samples on dense components + isolated points"""
        pts = []
        for p, q in self.comps:
            if q > p:
                n = max(2, int(np.ceil((q - p) * n_per_unit)) + 1)
                pts.append(np.linspace(p, q, n))
            elif include_isolated:
                pts.append(np.array([p]))
        return np.unique(np.concatenate(pts))

    # --------------------------------------------------------------- integrals
    def pieces(self, s, t):
        """Decompose [s,t)_T (s<=t in T) into ordered pieces
        ('d', c, d)    dense piece: Riemann integral over [c,d]
        ('s', tau, mu) right-scattered point tau with graininess mu."""
        out = []
        if t <= s:
            return out
        i0 = self.component_of(s)
        for i in range(i0, self.m):
            p, q = self.p[i], self.q[i]
            if p >= t:
                break
            c, d = max(p, s), min(q, t)
            if d > c:
                out.append(("d", c, d))
            if i < self.m - 1 and s <= q < t:
                out.append(("s", q, self.p[i + 1] - q))
        return out

    def h2(self, t, s):
        """time-scale monomial h_2(t,s) = int_s^t (tau - s) Delta tau"""
        if t >= s:
            tot = 0.0
            for kind, u, v in self.pieces(s, t):
                if kind == "d":
                    tot += 0.5 * ((v - s) ** 2 - (u - s) ** 2)
                else:
                    tot += v * (u - s)
            return tot
        # h_2(t,s) + h_2(s,t) = (t-s)^2
        return (s - t) ** 2 - self.h2(s, t)

    def h3(self, t, s):
        """h_3(t,s) = int_s^t h_2(tau,s) Delta tau  for t >= s"""
        if t < s:
            raise ValueError("h3 implemented for t >= s")
        tot, acc = 0.0, 0.0  # acc = h_2(current, s)
        for kind, u, v in self.pieces(s, t):
            if kind == "d":
                c, d = u, v
                tot += (d - c) * (acc - 0.5 * (c - s) ** 2) + ((d - s) ** 3 - (c - s) ** 3) / 6.0
                acc += 0.5 * ((d - s) ** 2 - (c - s) ** 2)
            else:
                tau, mu = u, v
                tot += mu * acc
                acc += mu * (tau - s)
        return tot

    def exp_const(self, p, t, s=None):
        """time-scale exponential e_p(t,s) for a constant p >= 0"""
        s = self.a if s is None else s
        val = 0.0
        for kind, u, v in self.pieces(s, t):
            if kind == "d":
                val += p * (v - u)
            else:
                val += np.log1p(p * v)
        return float(np.exp(val))

    def delta_integral(self, F, s, t, n_gauss=20, panel=0.05):
        """int_s^t F(tau) Delta tau with composite Gauss-Legendre on dense pieces.
        F must accept (tau array, sigma(tau) array)."""
        x, w = np.polynomial.legendre.leggauss(n_gauss)
        tot = 0.0
        for kind, u, v in self.pieces(s, t):
            if kind == "d":
                n = max(1, int(np.ceil((v - u) / panel)))
                e = np.linspace(u, v, n + 1)
                for e0, e1 in zip(e[:-1], e[1:]):
                    tau = 0.5 * (e1 - e0) * x + 0.5 * (e1 + e0)
                    tot += 0.5 * (e1 - e0) * np.sum(w * F(tau, tau))
            else:
                tot += v * F(np.array([u]), np.array([u + v]))[0]
        return tot


@dataclass
class Partition:
    """Partition a = t_0 < ... < t_N = b of [a,b]_T with time-scale trapezoid data.

    kind[j] in {'dense','gap','mixed'} for step [t_j, t_{j+1}]
    wm[j], wp[j]: trapezoid weights h_2(t_j,t_{j+1})/D_j and h_2(t_{j+1},t_j)/D_j
    kappa[j]    : Peano constant h_2(t_{j+1},t_j)^2/D_j - h_3(t_{j+1},t_j)
    sigL[j]     : sigma(t_j) (argument used for the left node value)
    sigR[j]     : sigma argument used for the right node value (t_{j+1} on dense
                  steps = left limit; sigma(t_{j+1}) otherwise)
    """
    T: TimeScale
    t: np.ndarray
    kind: list = field(default_factory=list)
    wm: np.ndarray = None
    wp: np.ndarray = None
    kappa: np.ndarray = None
    sigL: np.ndarray = None
    sigR: np.ndarray = None
    sigR_naive: np.ndarray = None

    def __post_init__(self):
        T, t = self.T, np.asarray(self.t, float)
        if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
            raise ValueError("partition requires at least two finite nodes")
        if np.any(np.diff(t) <= 0):
            raise ValueError("partition nodes must be strictly increasing")
        if abs(t[0] - T.a) > 1e-12 or abs(t[-1] - T.b) > 1e-12:
            raise ValueError("partition must contain both time-scale endpoints")
        if not all(T.contains(float(x)) for x in t):
            raise ValueError("partition node outside the time scale")
        self.t = t
        N = len(t) - 1
        self.wm = np.zeros(N); self.wp = np.zeros(N); self.kappa = np.zeros(N)
        self.sigL = np.zeros(N); self.sigR = np.zeros(N); self.sigR_naive = np.zeros(N)
        self.kind = []
        for j in range(N):
            a, b = t[j], t[j + 1]
            D = b - a
            pcs = T.pieces(a, b)
            if len(pcs) == 1 and pcs[0][0] == "d":
                k = "dense"; hp = 0.5 * D * D; h3 = D ** 3 / 6.0
            elif len(pcs) == 1 and pcs[0][0] == "s":
                k = "gap"; hp = 0.0; h3 = 0.0
            else:
                k = "mixed"; hp = T.h2(b, a); h3 = T.h3(b, a)
            self.kind.append(k)
            self.wp[j] = hp / D
            self.wm[j] = D - self.wp[j]
            self.kappa[j] = max(hp * hp / D - h3, 0.0)
            self.sigL[j] = T.sigma(a)
            self.sigR_naive[j] = T.sigma(b)
            self.sigR[j] = b if k == "dense" else T.sigma(b)

    @property
    def N(self):
        return len(self.t) - 1

    @property
    def kappa_total(self):
        return float(np.sum(self.kappa))

    def h_dense(self):
        D = np.diff(self.t)
        msk = np.array([k != "gap" for k in self.kind])
        return float(D[msk].max()) if msk.any() else 0.0

    # ------------------------------------------------------------ constructors
    @classmethod
    def gap_conforming(cls, T: TimeScale, h=None, counts=None):
        """nodes: all component endpoints + uniform refinement of dense components.
        Either a target step h, or explicit per-component step counts."""
        pts = []
        for i, (p, q) in enumerate(T.comps):
            if q > p:
                M = counts[i] if counts is not None else max(1, int(np.ceil((q - p) / h - 1e-12)))
                pts.append(np.linspace(p, q, M + 1))
            else:
                pts.append(np.array([p]))
        return cls(T, np.unique(np.concatenate(pts)))

    @classmethod
    def coarse(cls, T: TimeScale, every):
        """for discrete time scales: every `every`-th point of T plus the last one"""
        if np.any(T.q > T.p) or not isinstance(every, (int, np.integer)) or every < 1:
            raise ValueError("coarse requires a finite discrete scale and a positive integer stride")
        pts = T.p.copy()  # all isolated
        idx = np.arange(0, len(pts), every)
        if idx[-1] != len(pts) - 1:
            idx = np.append(idx, len(pts) - 1)
        return cls(T, pts[idx])
