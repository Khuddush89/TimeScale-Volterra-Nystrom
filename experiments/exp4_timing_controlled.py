"""Controlled timing of the classical P5 solvers (review item R3/R4).

Run on an otherwise idle machine, serially.  Each timed callable is executed once as a warm-up and then
`REPS` times; the median and the min/max are recorded (time.perf_counter, seconds).  Scopes:
  nodal       : nystrom.solve on the gap-conforming partition (nodal values only)
  interpolant : nystrom.solve + piecewise-linear interpolant evaluated on the validation set E (h=2^-10)
  richardson  : two solves (h, h/2) + extrapolation (nodal values on the coarse partition)
  picard      : fixed-point sweeps to tol 1e-15 on the same partition
Errors are max nodal errors (nodal/richardson/picard) or sampled max errors on E (interpolant).
Writes results/exp4_timing_controlled.json and results/csv/timing_controlled_P5.csv."""
from exp4_benchmark import *
import pandas as pd

REPS = 7


def timed(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, dict(median=float(np.median(ts)), min=float(np.min(ts)), max=float(np.max(ts)), reps=REPS)


def run():
    rows = []
    for lev in range(2, 11):
        P = part(lev)
        X, t = timed(lambda P=P: nystrom.solve(prob, P))
        rows.append(dict(method="Nystrom", scope="nodal", lev=lev, N=P.N, err=float(np.max(np.abs(X - ref_at(P.t)))), **t))
        print(rows[-1], flush=True)
    for lev in range(3, 9):
        P = part(lev)
        def interp(P=P):
            X = nystrom.solve(prob, P); return np.interp(E, P.t, X)
        v, t = timed(interp)
        rows.append(dict(method="Nystrom interpolant", scope="interpolant on E", lev=lev, N=P.N, err=float(np.max(np.abs(v - x_E))), **t))
        print(rows[-1], flush=True)
    for lev in range(2, 10):
        Pc, Pf = part(lev), part(lev + 1)
        def rich(Pc=Pc, Pf=Pf):
            Xc = nystrom.solve(prob, Pc); Xf = nystrom.solve(prob, Pf)
            return nystrom.richardson(Xc, nystrom.restrict(Pf, Xf, Pc))
        R, t = timed(rich)
        rows.append(dict(method="Richardson", scope="nodal", lev=lev, N=Pc.N, err=float(np.max(np.abs(R - ref_at(Pc.t)))), **t))
        print(rows[-1], flush=True)
    for lev in range(2, 9):
        P = part(lev)
        Y, t = timed(lambda P=P: nystrom.picard(prob, P, tol=1e-15)[0])
        rows.append(dict(method="Picard", scope="nodal", lev=lev, N=P.N, err=float(np.max(np.abs(Y - ref_at(P.t)))), **t))
        print(rows[-1], flush=True)
    save_json("exp4_timing_controlled", dict(rows=rows, env=env_info(), reps=REPS,
              note="serial run, no concurrent jobs; warm-up excluded; median over reps"))
    os.makedirs(os.path.join(RES, "csv"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "csv", "timing_controlled_P5.csv"), index=False)


if __name__ == "__main__":
    run()
