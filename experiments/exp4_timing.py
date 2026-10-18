"""Experiment 4 (timing): wall-clock times of selected solvers with the just-in-time
compilation separated out, and the cost of the certificate itself.
Run on an otherwise idle machine."""
from exp4_benchmark import *


def best_of(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.time(); fn(); ts.append(time.time() - t0)
    return dict(min=float(np.min(ts)), median=float(np.median(ts)))


def run():
    out = {}
    for lev in [4, 7, 10]:
        P = part(lev)
        out[f"Nystrom h=2^-{lev}"] = best_of(lambda: nystrom.solve(prob, P))
        print(f"Nystrom h=2^-{lev}", out[f"Nystrom h=2^-{lev}"], flush=True)
    for lev in [4, 7]:
        Pc, Pf = part(lev), part(lev + 1)

        def rich(Pc=Pc, Pf=Pf):
            Xc = nystrom.solve(prob, Pc); Xf = nystrom.solve(prob, Pf)
            return nystrom.richardson(Xc, nystrom.restrict(Pf, Xf, Pc))
        out[f"Richardson h=2^-{lev}"] = best_of(rich)
        print(f"Richardson h=2^-{lev}", out[f"Richardson h=2^-{lev}"], flush=True)
    for deg in [8, 12, 16, 20]:
        op = pw_operator(deg); res = ContinuousResidual(prob, op)
        tr = LinearTrial("pwcheb", T, xa, deg=deg)
        tm = {}
        t0 = time.time(); p, h = train_lm_structured(tr, res, tr.params0, maxiter=50, timing=tm); tm["total"] = time.time() - t0
        tm["err"] = error_fn(tr.numpy_fn(p))
        out[f"TS-spectral deg {deg}"] = tm
        print(f"TS-spectral deg {deg}", tm, flush=True)
    opC = colloc_operator(); resC = ContinuousResidual(prob, opC)
    for nf in [64, 128, 256]:
        tr = LinearTrial("rf", T, xa, key=jax.random.PRNGKey(100), n_feat=nf)
        tm = {}
        t0 = time.time(); p, h = train_lm_structured(tr, resC, tr.params0, maxiter=60, timing=tm); tm["total"] = time.time() - t0
        tm["err"] = error_fn(tr.numpy_fn(p))
        out[f"RF {nf}"] = tm
        print(f"RF {nf}", tm, flush=True)
    vf = tr.numpy_fn(p)
    out["certificate on E"] = best_of(lambda: certify_fn(vf))
    print("certificate on E", out["certificate on E"], flush=True)

    def majorant():
        U10 = nystrom.solve(pmaj, Pm10); U11 = nystrom.solve(pmaj, Pm11)
        return nystrom.richardson(U10, nystrom.restrict(Pm11, U11, Pm10))
    out["majorant u_ell on E"] = best_of(majorant, reps=1)
    print("majorant", out["majorant u_ell on E"], flush=True)
    out["env"] = env_info()
    save_json("exp4_timing", out)
    return out


if __name__ == "__main__":
    run()
