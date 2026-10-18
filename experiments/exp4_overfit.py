"""Experiment 4b: small training residual does not imply small error.
Global Chebyshev functional-link networks of increasing degree on a coarse
collocation set; the certificate (evaluated on the fine set E) detects overfitting."""
from exp4_benchmark import *


def run():
    rows = []
    for spacing, tag in [(1 / 20, "coarse"), (1 / 40, "fine")]:
        op = colloc_operator(spacing=spacing); res = ContinuousResidual(prob, op)
        for deg in [10, 20, 30, 40, 50, 60, 70, 80]:
            tr = LinearTrial("cheb", T, xa, deg=deg)
            t0 = time.time(); p, h = train_lm_structured(tr, res, tr.params0, maxiter=60); dt = time.time() - t0
            rec = record(f"Cheb deg {deg} ({tag}, {len(op.t)} colloc.)", "cheb_" + tag, tr, p, dt,
                         dict(deg=deg, n_colloc=len(op.t), train_rms=float(np.sqrt(h[-1]["loss"]))))
            rows.append(rec)
    save_json("exp4_overfit", dict(rows=rows))
    return rows


if __name__ == "__main__":
    for r in run():
        print(r["method"], "train_rms=%.1e err=%.1e eta_maj=%.1e" % (r["train_rms"], r["err"], r["eta_maj"]))
