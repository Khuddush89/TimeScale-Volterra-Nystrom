"""Experiment 7: nonlinear Volterra equations on unusual time scales.

Scales (all closed subsets of R; right-scattered points handled through sigma and mu):
  C4   : level-4 Cantor prefractal of [0,3]  (16 intervals, 15 gaps of four sizes)
  Pab  : P_{0.3,0.2} = U_k [0.5k, 0.5k+0.3], k = 0..5
  Rnd  : random irregular union of intervals and isolated points (seed 2026)
  Harm : harmonic numbers {H_n : 1 <= n <= 400}      (discrete, mu(H_n) = 1/(n+1))
  Sqrt : sqrt(N_0) cap [0,20]                         (discrete, mu = sqrt(n+1)-sqrt(n))
  Qacc : q^{N_0} U {0}, q = 0.9, on [0,1]             (0 is right-dense; infinitely many points)
Kernel K(t,s,x) = lambda exp(-alpha(t-s)) sin x with lambda = alpha = 1 (L = 1).
Run:  python experiments/exp7_unusual.py
"""
from common import *
import csv
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra import nystrom
from tsvolterra.problems import VolterraProblem, kernel_exp_sin, x_star

K, Kx, L = kernel_exp_sin(1.0, 1.0)
M2 = 19.5 + np.sqrt(37.0)          # analytic bound for x* (independent of T)
ell = lambda t, s: np.exp(-(np.asarray(t) - np.asarray(s)))


def cantor(level=4, a=0.0, b=3.0):
    iv = [(a, b)]
    for _ in range(level):
        new = []
        for p, q in iv:
            d = (q - p) / 3; new += [(p, p + d), (q - d, q)]
        iv = new
    return TimeScale(iv)


def pab(alpha=0.3, beta=0.2, kmax=5):
    return TimeScale([(k * (alpha + beta), k * (alpha + beta) + alpha) for k in range(kmax + 1)])


def random_scale(seed=2026):
    rng = np.random.default_rng(seed)
    comps = [(0.0, 0.35)]; pos = 0.35
    for i in range(4):
        for _ in range(int(rng.integers(1, 4))):
            pos += rng.uniform(0.08, 0.35); comps.append((pos, pos))
        pos += rng.uniform(0.08, 0.35); ln = rng.uniform(0.2, 0.6); comps.append((pos, pos + ln)); pos += ln
    comps = [(round(p, 6), round(q, 6)) for p, q in comps]
    return TimeScale(comps)


def harmonic(n=400):
    return TimeScale.from_points(np.cumsum(1.0 / np.arange(1, n + 1)))


def sqrt_scale(n=400):
    return TimeScale.from_points(np.sqrt(np.arange(0, n + 1)))


SCALES = {"C4": cantor(), "Pab": pab(), "Rnd": random_scale()}


def counts(T, k):
    dl = min(q - p for p, q in T.comps if q > p)
    return [max(1, int(round((q - p) / dl))) * 2 ** k if q > p else 0 for p, q in T.comps]


def dense_study(name, T, levels):
    prob = VolterraProblem(T, K, Kx, L, x_exact=x_star, n_gauss=20, panel=0.02)
    rows, sols = [], []
    for k in levels:
        P = Partition.gap_conforming(T, counts=counts(T, k))
        t0 = time.perf_counter(); X = nystrom.solve(prob, P); dt = time.perf_counter() - t0
        U = nystrom.majorant_stability(P, ell)
        err = float(np.max(np.abs(X - x_star(P.t))))
        r = dict(scale=name, level=k, h=P.h_dense(), N=P.N, error=err, kappa=P.kappa_total, U=float(U[-1]),
                 bound=float(U[-1] * P.kappa_total * M2), seconds=dt)
        if rows: r["eoc"] = float(np.log2(rows[-1]["error"] / err))
        rows.append(r); sols.append((P, X))
    for i in range(len(rows) - 1):
        P, X = sols[i]; Pf, Xf = sols[i + 1]
        R = nystrom.richardson(X, nystrom.restrict(Pf, Xf, P))
        rows[i]["richardson"] = float(np.max(np.abs(R - x_star(P.t))))
    for i in range(1, len(rows) - 1):
        rows[i]["eoc_richardson"] = float(np.log2(rows[i - 1]["richardson"] / rows[i]["richardson"]))
    return rows, sols


def discrete_study(name, T):
    fd = lambda s: 1 + 0.5 * np.sin(2 * np.asarray(s))
    prob = VolterraProblem(T, K, Kx, L, f=fd)
    full = Partition(T, T.p.copy()); Xfull = nystrom.solve(prob, full)
    t = T.p; xr = np.empty(len(t)); xr[0] = fd(t[0]); mu = np.diff(t)
    for n in range(1, len(t)):
        xr[n] = fd(t[n]) + np.sum(mu[:n] * np.exp(-(t[n] - t[:n])) * np.sin(xr[:n]))
    rows = [dict(scale=name, m=1, N=full.N, H=float(mu.max()), error=float(np.max(abs(Xfull - xr))),
                 kappa=full.kappa_total, U=float(nystrom.majorant_stability(full, ell)[-1]))]
    for m in [2, 4, 8, 16, 32]:
        P = Partition.coarse(T, m)
        gamma = float(max(P.wp[n - 1] * ell(P.t[n], P.t[n]) for n in range(1, P.N + 1)))
        row = dict(scale=name, m=m, N=P.N, H=float(np.max(np.diff(P.t))), kappa=P.kappa_total, gamma=gamma)
        if gamma >= 1:
            row.update(error=None, U=None, note="gamma>=1: solvability condition (4.6) violated, step skipped")
        else:
            X = nystrom.solve(prob, P); idx = np.searchsorted(t, P.t)
            row.update(error=float(np.max(abs(X - xr[idx]))), U=float(nystrom.majorant_stability(P, ell)[-1]))
        rows.append(row)
    for r in rows:
        r["error_over_kappa"] = r["error"] / r["kappa"] if (r["kappa"] > 1e-18 and r["error"] is not None) else None
    return rows, (t, xr)


def qacc_study(q=0.9, nlist=(10, 20, 40, 80, 160, 320), nref=800):
    fd = lambda s: 1 + 0.5 * np.sin(2 * np.asarray(s))
    def solve(n):
        tk = q ** np.arange(n, -1, -1.0)
        mu = np.concatenate([np.diff(tk), [0.0]])
        X = np.empty(n + 1)
        for i in range(n + 1):
            X[i] = fd(tk[i]) + np.sum(mu[:i] * np.exp(-(tk[i] - tk[:i])) * np.sin(X[:i]))
        return tk, X, mu
    tref, Xref, _ = solve(nref)
    rows = []
    for n in nlist:
        tk, X, mu = solve(n)
        idx = nref - n + np.arange(n + 1)
        assert np.allclose(tref[idx], tk, rtol=1e-12, atol=0)
        err = float(np.max(abs(X - Xref[idx])))
        bound = float(q ** n * np.prod(1 + mu[:-1]))
        rows.append(dict(scale="Qacc", n=n, smallest_point=float(q ** n), error=err, bound=bound, ratio=err / bound))
    return rows, (tref, Xref)


if __name__ == "__main__":
    t0 = time.time(); out = {}; allrows = []
    for name, T in SCALES.items():
        rows, sols = dense_study(name, T, range(0, 7)); out[name] = dict(rows=rows, comps=T.comps)
        P, X = sols[3]; out[name]["profile"] = dict(t=P.t, x=X)
        allrows += rows; print(name, len(T.comps), [(r["N"], f"{r['error']:.2e}", round(r.get('eoc', 0), 2)) for r in rows], flush=True)
    for name, T in [("Harm", harmonic()), ("Sqrt", sqrt_scale())]:
        rows, (t, x) = discrete_study(name, T); out[name] = dict(rows=rows, profile=dict(t=t, x=x)); allrows += rows
        print(name, rows, flush=True)
    rows, (t, x) = qacc_study(); out["Qacc"] = dict(rows=rows, profile=dict(t=t, x=x)); allrows += rows
    print(rows, flush=True)
    out["seconds"] = time.time() - t0; out["env"] = env_info()
    save_json("exp7_unusual", out)
    os.makedirs(os.path.join(RES, "csv"), exist_ok=True)
    keys = sorted({k for r in allrows for k in r})
    with open(os.path.join(RES, "csv", "table_unusual_scales.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in allrows]
    print("seconds", out["seconds"])
