"""Full-resolution error profiles and certified envelopes for selected approximations
(loads the seed-0 checkpoints of Experiment 4; --retrain retrains them with identical settings)."""
from exp4_benchmark import *


def full_profile(vf):
    op_t, r = certify.continuous_residual(prob, vf, E_cert, n_gauss=8, max_panel=1.0)
    rr = np.maximum.accumulate(np.abs(r))
    return dict(err=np.abs(vf(E) - x_E), env_maj=Gam_E * rr, env_exp=eL_E * rr)


# finer evaluation set (h = 2^-11) to check the sensitivity of the certificate to the sampling of r_v
P11 = part(11); E11 = P11.t
assert np.allclose(P_ref.t, E11)
U12 = nystrom.solve(pmaj, part(12))
Gam_E11 = 1.001 * nystrom.richardson(U11, nystrom.restrict(part(12), U12, Pm11))
CHECK = {}


def eval_check(name, vf):
    """eta_maj and the error on E (h=2^-10) and on the finer set E11 (h=2^-11)"""
    out = {}
    for tag, Et, Gm, xr in [("E10", E, Gam_E, x_E), ("E11", E11, Gam_E11, X_ref)]:
        op_t, r = certify.continuous_residual(prob, vf, Et, n_gauss=8, max_panel=1.0)
        assert np.allclose(op_t, Et)
        rr = np.maximum.accumulate(np.abs(r))
        out[tag] = dict(eta_maj=float(np.max(Gm * rr)), sup_r=float(rr[-1]), err=float(np.max(np.abs(vf(Et) - xr))))
    out["rel_change_eta"] = abs(out["E11"]["eta_maj"] / out["E10"]["eta_maj"] - 1.0)
    CHECK[name] = out
    print("eval-check", name, out, flush=True)


def run():
    out = {}
    P = part(4); X = nystrom.solve(prob, P)
    out["Nystrom-pl h=2^-4"] = full_profile(lambda t: np.interp(np.atleast_1d(t), P.t, X))
    eval_check("Nystrom-pl h=2^-4", lambda t: np.interp(np.atleast_1d(t), P.t, X))
    # the trained models are loaded from the checkpoints written by exp4_benchmark.py (no retraining);
    # pass --retrain to retrain the seed-0 models with identical settings instead.
    retrain = "--retrain" in sys.argv
    opC = colloc_operator(); resC = ContinuousResidual(prob, opC)
    for name, kw in [("MLP 3x32, loss C (seed 0)", dict(T=None)), ("MLP-TS 3x32, loss C (seed 0)", dict(T=T))]:
        if retrain:
            tr = TrialFunction("mlp", T.a, T.b, xa, key=jax.random.PRNGKey(0), width=32, depth=3, **kw)
            rf = make_residual_fn(tr, resC)
            p, _ = train_adam(rf, tr.params0, steps=5000, lr=3e-3, lr_end=1e-4, log_every=250)
            p, _ = train_lbfgs(rf, p, maxiter=5000, log_every=250)
        else:
            tr, p, _ = checkpoints.load(os.path.join(CKPT, ckpt_name(name) + ".npz"), T, xa)
        out[name] = full_profile(tr.numpy_fn(p))
        eval_check(name, tr.numpy_fn(p))
        print(name, float(np.max(out[name]["err"])), flush=True)
    for name in ["RF 256, loss C, LM (seed 0)", "TS-spectral deg 12"]:
        tr, p, _ = checkpoints.load(os.path.join(CKPT, ckpt_name(name) + ".npz"), T, xa)
        out[name] = full_profile(tr.numpy_fn(p))
        eval_check(name, tr.numpy_fn(p))
    save_json("exp4_profiles", dict(E=E, profiles=out, eval_check=CHECK))
    for k, v in out.items():
        print(k, "max err %.3e  max env %.3e" % (np.max(v["err"]), np.max(v["env_maj"])))


if __name__ == "__main__":
    run()
