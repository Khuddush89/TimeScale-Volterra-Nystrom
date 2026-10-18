"""Checkpoints of trained trial functions.

A checkpoint stores everything needed to evaluate a trained trial function without retraining:
the constructor specification (class, kind, architecture, PRNG seed of the initialisation or of the
random features, whether the graininess clock is an input), the final parameters as one flat float64
vector, the random-feature parameters themselves (for exact verification), and the training/loss
configuration.  ``load`` rebuilds the trial on the given time scale, checks the stored feature
parameters against the regenerated ones, and returns (trial, params).
"""
import json, os
import numpy as np


def _flat(params):
    from jax.flatten_util import ravel_pytree
    return np.asarray(ravel_pytree(params)[0], dtype=np.float64)


def save(path, trial, params, spec, extra=None):
    """spec: dict(cls='TrialFunction'|'LinearTrial', kind=..., seed=int|None, ts_input=bool, kwargs={...})"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(params=_flat(params))
    if getattr(trial, "w", None) is not None and trial.kind == "rf":
        arrays["rf_w"] = np.asarray(trial.w, np.float64); arrays["rf_b"] = np.asarray(trial.bb, np.float64)
    meta = dict(spec=spec, extra=extra or {}, n_params=int(arrays["params"].size), dtype="float64")
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def build(spec, T, xa):
    import jax
    from .neural import TrialFunction, LinearTrial
    kw = dict(spec.get("kwargs", {}))
    key = None if spec.get("seed") is None else jax.random.PRNGKey(int(spec["seed"]))
    if spec["cls"] == "TrialFunction":
        return TrialFunction(spec["kind"], T.a, T.b, xa, key=key, T=(T if spec.get("ts_input") else None), **kw)
    return LinearTrial(spec["kind"], T, xa, key=key, **kw)


def load(path, T, xa):
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    tr = build(meta["spec"], T, xa)
    if "rf_w" in z:
        assert np.array_equal(np.asarray(tr.w, np.float64), z["rf_w"]) and np.array_equal(np.asarray(tr.bb, np.float64), z["rf_b"]), \
            "regenerated random features differ from the stored ones"
    _, unravel = ravel_pytree(tr.params0)
    flat = z["params"]
    params = unravel(jnp.asarray(flat)) if meta["spec"]["cls"] == "TrialFunction" else jnp.asarray(flat)
    return tr, params, meta
