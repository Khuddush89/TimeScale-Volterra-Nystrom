"""Command-line interface.

    python -m tsvolterra.cli scales
    python -m tsvolterra.cli info  --scale cantor
    python -m tsvolterra.cli solve --scale hybrid --h 0.0625 --out sol.csv --plot sol.png
    python -m tsvolterra.cli solve --scale harmonic            (full partition: exact recursion)
    python -m tsvolterra.cli solve --scale qacc --n 160        (accumulation point, truncation bound)
    python -m tsvolterra.cli converge --scale cantor --levels 5

Equation: x(t) = f(t) + int_a^t lam exp(-alpha (t-s)) sin x(s) Delta s,  f(t) = 1 + 0.5 sin 2t (default)
or the manufactured solution x*(t) = 1 + exp(-t/2) sin 3t (--manufactured)."""
from __future__ import annotations
import argparse, csv, sys, time
import numpy as np
from .timescale import Partition
from .problems import VolterraProblem, kernel_exp_sin, x_star
from . import nystrom, scales


def _problem(T, lam, alpha, manufactured):
    K, Kx, L = kernel_exp_sin(lam, alpha)
    if manufactured:
        return VolterraProblem(T, K, Kx, L, x_exact=x_star, n_gauss=20, panel=0.02)
    return VolterraProblem(T, K, Kx, L, f=lambda t: 1 + 0.5 * np.sin(2 * np.asarray(t)))


def _partition(T, h):
    if all(q == p for p, q in T.comps):          # purely discrete: full partition (exact recursion)
        return Partition(T, T.p.copy())
    return Partition.gap_conforming(T, h=h)


def cmd_scales(a):
    for k in sorted(scales.SCALES):
        T = scales.get(k); nd = sum(q > p for p, q in T.comps)
        print(f"{k:10s} [{T.a:.4g}, {T.b:.4g}]  components={T.m:4d}  intervals={nd:3d}  isolated={T.m - nd:4d}")
    print("family:<nu>  T_nu = U_k [k, k+nu] U {4}")
    print("qacc        q^N0 U {0} (q=0.9); 0 is right-dense, accumulation point")


def cmd_info(a):
    if a.scale == "qacc":
        print("q^N0 U {0}: sigma(0)=0 (right-dense); sigma(q^k)=q^(k-1), mu(q^k)=q^k(1/q-1); infinitely many gaps"); return
    T = scales.get(a.scale)
    print(f"time scale {a.scale}: [{T.a}, {T.b}], {T.m} components, dense length {T.dense_length():.6g}")
    for i, (p, q) in enumerate(T.comps):
        kind = "interval" if q > p else "isolated point"
        rs = "" if i == T.m - 1 else f"  sigma({q:.6g})={T.sigma(q):.6g}, mu={T.mu(q):.6g} (right-scattered)"
        print(f"  [{p:.6g}, {q:.6g}]  {kind}{rs}")
    g = [p2 - q1 for q1, p2 in T.gaps()]
    if g: print(f"gaps: {len(g)}, min {min(g):.6g}, max {max(g):.6g}")


def cmd_solve(a):
    t0 = time.perf_counter()
    if a.scale == "qacc":
        K, _, L = kernel_exp_sin(a.lam, a.alpha)
        t, X, mu = scales.qacc_solve(lambda s: 1 + 0.5 * np.sin(2 * s), K, q=a.q, n=a.n)
        bound = a.lam * a.q ** a.n * np.prod(1 + L * mu[:-1])
        print(f"q^N0 U {{0}}, q={a.q}, retained points {a.n + 1}; truncation bound {bound:.3e}; x(1) = {X[-1]:.12f}")
        _write(a, t, X, None); _plot(a, None, t, X); return
    T = scales.get(a.scale); prob = _problem(T, a.lam, a.alpha, a.manufactured); P = _partition(T, a.h)
    X = nystrom.solve(prob, P); dt = time.perf_counter() - t0
    gamma = max([P.wp[n - 1] * prob.L for n in range(1, P.N + 1)] + [0.0])
    print(f"scale {a.scale}: {P.N + 1} nodes, dense step {P.h_dense():.4g}, kappa(P) = {P.kappa_total:.3e}, gamma <= {gamma:.3f}, time {dt:.4f} s")
    print(f"x(b) = X_N = {X[-1]:.12f}")
    ex = None
    if a.manufactured:
        ex = x_star(P.t); print(f"max nodal error vs x* = {np.max(abs(X - ex)):.3e}")
    _write(a, P.t, X, ex); _plot(a, T, P.t, X)


def cmd_converge(a):
    T = scales.get(a.scale); prob = _problem(T, a.lam, a.alpha, True); prev = None
    dl = min(q - p for p, q in T.comps if q > p)
    print(f"{'level':>5} {'nodes':>6} {'h_d':>10} {'error':>11} {'EOC':>6}")
    for k in range(a.levels):
        counts = [max(1, int(round((q - p) / dl))) * 2 ** k if q > p else 0 for p, q in T.comps]
        P = Partition.gap_conforming(T, counts=counts); X = nystrom.solve(prob, P); e = float(np.max(abs(X - x_star(P.t))))
        eoc = "" if prev is None else f"{np.log2(prev / e):6.2f}"; prev = e
        print(f"{k:5d} {P.N + 1:6d} {P.h_dense():10.5f} {e:11.3e} {eoc}")


def _write(a, t, X, ex):
    if not a.out: return
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t", "X"] + (["x_exact"] if ex is not None else []))
        for i in range(len(t)): w.writerow([t[i], X[i]] + ([ex[i]] if ex is not None else []))
    print("wrote", a.out)


def _plot(a, T, t, X):
    if not a.plot: return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from .tsplot import PAPER, plot_on_scale, scale_strip
    plt.rcParams.update(PAPER); fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
    if T is None:
        ax.scatter(t, X, c=np.log10(t), cmap="turbo", s=12); ax.set_xscale("symlog", linthresh=1e-3)
    else:
        cols = plot_on_scale(ax, T.comps, t, X, cmap="turbo"); scale_strip(ax, T.comps, float(np.min(X)) - 0.05, colors=cols)
    ax.set(xlabel="t in T", ylabel="X", title=f"Nystrom solution on {a.scale}"); fig.savefig(a.plot, dpi=200); print("wrote", a.plot)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tsvolterra", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scales", help="list the built-in time scales")
    s = sub.add_parser("info", help="components, sigma and mu of a scale"); s.add_argument("--scale", required=True)
    for name in ("solve", "converge"):
        s = sub.add_parser(name); s.add_argument("--scale", required=True)
        s.add_argument("--lam", type=float, default=1.0); s.add_argument("--alpha", type=float, default=1.0)
        if name == "solve":
            s.add_argument("--h", type=float, default=2 ** -5, help="dense step length")
            s.add_argument("--manufactured", action="store_true", help="use x*(t)=1+exp(-t/2)sin3t")
            s.add_argument("--q", type=float, default=0.9); s.add_argument("--n", type=int, default=160)
            s.add_argument("--out"); s.add_argument("--plot")
        else:
            s.add_argument("--levels", type=int, default=5)
    a = ap.parse_args(argv)
    {"scales": cmd_scales, "info": cmd_info, "solve": cmd_solve, "converge": cmd_converge}[a.cmd](a)


if __name__ == "__main__":
    main()
