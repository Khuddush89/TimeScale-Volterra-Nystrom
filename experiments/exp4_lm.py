"""Experiment 4 (supplement): MLPs trained by Adam warm start + Levenberg-Marquardt."""
from exp4_benchmark import *


def run_lm(seeds=(0, 1, 2, 3, 4)):
    rows = []
    opC = colloc_operator(); resC = ContinuousResidual(prob, opC)
    for name, fam, kw in [("MLP 2x16, loss C, Adam+LM", "mlp_C_alm", dict(T=None)),
                          ("MLP-TS 2x16, loss C, Adam+LM", "mlpts_C_alm", dict(T=T))]:
        for sd in seeds:
            tr = TrialFunction("mlp", T.a, T.b, xa, key=jax.random.PRNGKey(sd), width=16, depth=2, **kw)
            rf = make_residual_fn(tr, resC)
            t0 = time.time()
            p, h1 = train_adam(rf, tr.params0, steps=2000, lr=3e-3, lr_end=3e-4, log_every=500)
            p, h2 = train_lm_structured(tr, resC, p, maxiter=400)
            dt = time.time() - t0
            rows.append(record(f"{name} (seed {sd})", fam, tr, p, dt, dict(seed=sd, its=len(h2)),
                               profile=(sd == 0),
                               spec=dict(cls="TrialFunction", kind="mlp", seed=sd, ts_input=kw["T"] is not None,
                                         kwargs=dict(width=16, depth=2)),
                               train_cfg=dict(loss="C (spacing 1/40, 8 Gauss nodes)",
                                              optimizer="Adam 2000 steps lr 3e-3->3e-4, then structured LM maxiter 400")))
    return rows


if __name__ == "__main__":
    rows = run_lm()
    save_json("exp4_neural_lm", dict(rows=rows, profiles=PROFILES, E=E[::2]))
    print("done")
