"""Experiment 4 (Section 7.4): certified neural trial functions vs classical solvers on a
hybrid time scale for a problem without manufactured solution."""
from common import *
import jax, jax.numpy as jnp
from tsvolterra.timescale import TimeScale, Partition
from tsvolterra import nystrom, certify
from tsvolterra.quadrature import GaussVolterraOperator
from tsvolterra.problems import *
from tsvolterra.neural import *
from tsvolterra import checkpoints
import re as _re
CKPT = os.path.join(RES, "checkpoints")
from tsvolterra import plotting as pl

T = ts_hybrid()
K, Kx, L = kernel_exp_sin(1.0, 1.0)
fnat = lambda t: 1.0 + 0.5 * np.sin(2.0 * np.asarray(t))
prob = VolterraProblem(T, K, Kx, L, f=fnat)
xa = float(fnat(T.a))


def counts_for(lev):
    return [int(round(2 ** lev * (q - p))) if q > p else 0 for (p, q) in T.comps]


def part(lev):
    return Partition.gap_conforming(T, counts=counts_for(lev))


# ---------------------------------------------------------------- reference
def reference():
    P11, P12 = part(11), part(12)
    X11 = nystrom.solve(prob, P11); X12 = nystrom.solve(prob, P12)
    R = nystrom.richardson(X11, nystrom.restrict(P12, X12, P11))
    return P11, R


P_ref, X_ref = reference()
E = part(10).t                       # evaluation set (h = 1/1024)
x_E = nystrom.restrict(P_ref, X_ref, part(10))
E_cert = part(10).t


def ref_at(t):
    idx = np.searchsorted(P_ref.t, t)
    assert np.allclose(P_ref.t[idx], t)
    return X_ref[idx]


# majorant Gamma for ell(t,s) = exp(-(t-s)) :  u = 1 + int ell u
Kmaj = lambda t, s, sg, x, xp=np: xp.exp(-(t - s)) * x
Kmajx = lambda t, s, sg, x, xp=np: xp.exp(-(t - s)) + 0 * x
pmaj = VolterraProblem(T, Kmaj, Kmajx, 1.0, f=lambda t: 1.0 + 0 * np.asarray(t))
Pm10 = part(10); Pm11 = part(11)
U10 = nystrom.solve(pmaj, Pm10); U11 = nystrom.solve(pmaj, Pm11)
Gam_E = 1.001 * nystrom.richardson(U10, nystrom.restrict(Pm11, U11, Pm10))
eL_E = np.array([T.exp_const(L, t) for t in E])


def certify_fn(vfun, t_eval=E_cert):
    op_t, r = certify.continuous_residual(prob, vfun, t_eval, n_gauss=8, max_panel=1.0)
    assert np.allclose(op_t, E)
    rr = np.maximum.accumulate(np.abs(r))
    eta_exp = float(np.max(eL_E * rr)); eta_maj = float(np.max(Gam_E * rr))
    return dict(sup_r=float(rr[-1]), eta_exp=eta_exp, eta_maj=eta_maj,
                lower=float(rr[-1] / (1.0 + L * (T.b - T.a))))


def error_fn(vfun):
    return float(np.max(np.abs(vfun(E) - x_E)))


# coarse certificate for training histories
E_c = part(7).t
x_c = nystrom.restrict(P_ref, X_ref, part(7))
eL_c = np.array([T.exp_const(L, t) for t in E_c])


def quick_cb(trial):
    def cb(params):
        vf = trial.numpy_fn(params)
        t_, r = certify.continuous_residual(prob, vf, E_c, n_gauss=8, max_panel=1.0)
        rr = np.maximum.accumulate(np.abs(r))
        return dict(err=float(np.max(np.abs(vf(E_c) - x_c))), cert=float(np.max(eL_c * rr)), sup_r=float(rr[-1]))
    return cb


def colloc_operator(spacing=1 / 40, n_gauss=8):
    return GaussVolterraOperator(T, T.sample(1.0 / spacing), n_gauss=n_gauss)


def pw_operator(deg, n_gauss=12):
    pts = []
    for (p, q) in T.comps:
        if q > p:
            n = 2 * (deg + 1); x = np.cos(np.pi * np.arange(n) / (n - 1))[::-1]
            pts.append(p + (q - p) * (x + 1) / 2)
        else:
            pts.append(np.array([p]))
    return GaussVolterraOperator(T, np.concatenate(pts), n_gauss=n_gauss)


PROFILES = {}


def certify_profile(vfun):
    op_t, r = certify.continuous_residual(prob, vfun, E_cert, n_gauss=8, max_panel=1.0)
    rr = np.maximum.accumulate(np.abs(r))
    return np.abs(vfun(E) - x_E), Gam_E * rr, eL_E * rr, np.abs(r)


def ckpt_name(name):
    return _re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def record(name, family, trial, params, t_train, extra=None, profile=False, spec=None, train_cfg=None):
    vf = trial.numpy_fn(params)
    vf(E)                                           # warm-up (JIT compilation of the evaluator)
    t0 = time.perf_counter(); vE = vf(E); t_eval = time.perf_counter() - t0
    rec = dict(method=name, family=family, n_params=int(trial.n_params), time=t_train, err=float(np.max(np.abs(vE - x_E))),
               time_eval=t_eval)
    t0 = time.perf_counter(); rec.update(certify_fn(vf)); rec["time_assess"] = time.perf_counter() - t0
    if spec is not None:
        fn = ckpt_name(name) + ".npz"
        checkpoints.save(os.path.join(CKPT, fn), trial, params, spec,
                         extra=dict(method=name, family=family, train=train_cfg or {}, err=rec["err"],
                                    eta_maj=rec["eta_maj"], eta_exp=rec["eta_exp"]))
        rec["checkpoint"] = "results/checkpoints/" + fn
    if profile:
        e, gm, ge, r = certify_profile(vf)
        PROFILES[name] = dict(err=e[::2], env_maj=gm[::2], env_exp=ge[::2], r=r[::2])
    rec["eff_exp"] = rec["eta_exp"] / rec["err"]; rec["eff_maj"] = rec["eta_maj"] / rec["err"]
    if extra: rec.update(extra)
    print(f"{name:34s} p={rec['n_params']:5d} t={t_train:7.1f}s err={rec['err']:.2e} r={rec['sup_r']:.2e} "
          f"eta_exp={rec['eta_exp']:.2e} eta_maj={rec['eta_maj']:.2e}", flush=True)
    return rec


def run_classical():
    rows = []
    sols = {}
    for lev in range(2, 11):
        P = part(lev)
        t0 = time.time(); X = nystrom.solve(prob, P); dt = time.time() - t0
        sols[lev] = (P, X, dt)
        rows.append(dict(method=f"Nystrom h=2^-{lev}", family="nystrom", lev=lev, N=P.N, time=dt,
                         err=float(np.max(np.abs(X - ref_at(P.t))))))
    for lev in range(2, 10):
        Pc, Xc, tc = sols[lev]; Pf, Xf, tf = sols[lev + 1]
        R = nystrom.richardson(Xc, nystrom.restrict(Pf, Xf, Pc))
        rows.append(dict(method=f"Richardson h=2^-{lev}", family="richardson", lev=lev, N=Pc.N, time=tc + tf,
                         err=float(np.max(np.abs(R - ref_at(Pc.t))))))
    for lev in range(2, 9):
        P = sols[lev][0]
        t0 = time.time(); Y, its, _ = nystrom.picard(prob, P, tol=1e-15); dt = time.time() - t0
        rows.append(dict(method=f"Picard h=2^-{lev}", family="picard", lev=lev, N=P.N, time=dt, its=its,
                         err=float(np.max(np.abs(Y - ref_at(P.t))))))
    # certified Nystrom: piecewise-linear interpolant on dense steps
    for lev in range(3, 9):
        P, X, dt = sols[lev]
        def vf(t, P=P, X=X):
            return np.interp(np.atleast_1d(t), P.t, X)
        rec = dict(method=f"Nystrom-pl h=2^-{lev}", family="nystrom_pl", lev=lev, time=dt, err=error_fn(vf))
        rec.update(certify_fn(vf))
        if lev == 4:
            e, gm, ge, r = certify_profile(vf)
            PROFILES[rec["method"]] = dict(err=e[::2], env_maj=gm[::2], env_exp=ge[::2], r=r[::2]); rec["eff_exp"] = rec["eta_exp"] / rec["err"]; rec["eff_maj"] = rec["eta_maj"] / rec["err"]
        rows.append(rec)
        print(rec, flush=True)
    for deg in [4, 6, 8, 10, 12, 14, 16, 20]:
        op = pw_operator(deg); res = ContinuousResidual(prob, op)
        tr = LinearTrial("pwcheb", T, xa, deg=deg)
        t0 = time.time(); p, h = train_lm_structured(tr, res, tr.params0, maxiter=50); dt = time.time() - t0
        rows.append(record(f"TS-spectral deg {deg}", "spectral", tr, p, dt, dict(deg=deg, its=len(h)),
                           profile=(deg == 12),
                           spec=dict(cls="LinearTrial", kind="pwcheb", seed=None, kwargs=dict(deg=deg)),
                           train_cfg=dict(loss="C, Chebyshev-Lobatto points 2(deg+1) per dense component, 12 Gauss nodes",
                                          optimizer="structured LM", maxiter=50)))
    return rows


def run_neural(seeds=(0, 1, 2, 3, 4), hist_seed=0):
    rows = []; hists = {}
    opC = colloc_operator(); resC = ContinuousResidual(prob, opC)
    P4 = part(4); resD = DiscreteResidual(prob, P4)
    print("collocation points", len(opC.t), "Gauss nodes", len(opC.s), "| D-partition N =", P4.N, flush=True)
    configs = [("MLP 3x32, loss D", "mlp_D", dict(T=None), resD),
               ("MLP 3x32, loss C", "mlp_C", dict(T=None), resC),
               ("MLP-TS 3x32, loss C", "mlpts_C", dict(T=T), resC)]
    for name, fam, kw, res in configs:
        for sd in seeds:
            tr = TrialFunction("mlp", T.a, T.b, xa, key=jax.random.PRNGKey(sd), width=32, depth=3, **kw)
            rf = make_residual_fn(tr, res)
            cb = quick_cb(tr) if sd == hist_seed else None
            t0 = time.time()
            p, h1 = train_adam(rf, tr.params0, steps=5000, lr=3e-3, lr_end=1e-4, log_every=250, callback=cb)
            ta = time.time() - t0
            p, h2 = train_lbfgs(rf, p, maxiter=5000, log_every=250, callback=cb, it0=5000, t_offset=ta)
            dt = time.time() - t0
            if sd == hist_seed:
                hists[fam] = h1 + h2
            rows.append(record(f"{name} (seed {sd})", fam, tr, p, dt, dict(seed=sd), profile=(sd == hist_seed),
                               spec=dict(cls="TrialFunction", kind="mlp", seed=sd, ts_input=kw["T"] is not None,
                                         kwargs=dict(width=32, depth=3)),
                               train_cfg=dict(loss=("D (h=2^-4 partition)" if res is resD else "C (spacing 1/40, 8 Gauss nodes)"),
                                              optimizer="Adam 5000 steps lr 3e-3->1e-4, then L-BFGS maxiter 5000")))
    for name, fam, kw in [("MLP 2x16, loss C, LM", "mlp_C_lm", dict(T=None)),
                          ("MLP-TS 2x16, loss C, LM", "mlpts_C_lm", dict(T=T))]:
        for sd in seeds[:3]:
            tr = TrialFunction("mlp", T.a, T.b, xa, key=jax.random.PRNGKey(sd), width=16, depth=2, **kw)
            t0 = time.time(); p, h = train_lm_structured(tr, resC, tr.params0, maxiter=1000); dt = time.time() - t0
            rows.append(record(f"{name} (seed {sd})", fam, tr, p, dt, dict(seed=sd, its=len(h)),
                               spec=dict(cls="TrialFunction", kind="mlp", seed=sd, ts_input=kw["T"] is not None,
                                         kwargs=dict(width=16, depth=2)),
                               train_cfg=dict(loss="C (spacing 1/40, 8 Gauss nodes)", optimizer="structured LM maxiter 1000")))
    for nf in [64, 128, 256]:
        for sd in seeds:
            tr = LinearTrial("rf", T, xa, key=jax.random.PRNGKey(100 + sd), n_feat=nf)
            t0 = time.time(); p, h = train_lm_structured(tr, resC, tr.params0, maxiter=60); dt = time.time() - t0
            rows.append(record(f"RF {nf}, loss C, LM (seed {sd})", "rf", tr, p, dt, dict(seed=sd, n_feat=nf),
                               profile=(sd == hist_seed and nf == 256),
                               spec=dict(cls="LinearTrial", kind="rf", seed=100 + sd, kwargs=dict(n_feat=nf)),
                               train_cfg=dict(loss="C (spacing 1/40, 8 Gauss nodes)", optimizer="structured LM maxiter 60")))
    for deg in [10, 20, 30, 40]:
        tr = LinearTrial("cheb", T, xa, deg=deg)
        t0 = time.time(); p, h = train_lm_structured(tr, resC, tr.params0, maxiter=60); dt = time.time() - t0
        rows.append(record(f"Chebyshev FLNN deg {deg}, loss C, LM", "cheb", tr, p, dt, dict(deg=deg),
                           spec=dict(cls="LinearTrial", kind="cheb", seed=None, kwargs=dict(deg=deg)),
                           train_cfg=dict(loss="C (spacing 1/40, 8 Gauss nodes)", optimizer="structured LM maxiter 60")))
    return rows, hists


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {"env": env_info(), "eL_b": float(eL_E[-1]), "Gamma_b": float(np.max(Gam_E)),
           "colloc": None}
    if which in ("all", "classical"):
        out["classical"] = run_classical()
        out["profiles"] = PROFILES; out["E"] = E[::2]; out["x_E"] = x_E[::2]
        save_json("exp4_classical", out)
    if which in ("all", "neural"):
        rows, hists = run_neural()
        save_json("exp4_neural", dict(rows=rows, hists=hists, eL_b=float(eL_E[-1]), Gamma_b=float(np.max(Gam_E)),
                                      profiles=PROFILES, E=E[::2]))
    # profiles of selected approximations for the figure
    print("done")
