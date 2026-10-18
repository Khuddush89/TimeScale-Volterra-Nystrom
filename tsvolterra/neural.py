"""Neural trial functions on time scales and their training (JAX, float64).

Trial function (Georgiev-type, satisfies v(a) = x(a) = f(a) exactly):
    v_theta(t) = f(a) + h_1(t,a)/(b-a) * N_theta(xi(t)),   xi = 2(t-a)/(b-a) - 1.
Models N_theta:
    'mlp'  : fully connected tanh network, all weights trained
    'rf'   : random-feature (extreme-learning-machine) network, output weights trained
    'cheb' : Chebyshev functional-link network sum_k beta_k T_k(xi)
Losses (mean square of residual vectors):
    'D' : discrete residual of the time-scale Nystrom (trapezoid) scheme on a partition
    'C' : continuous residual with Gauss-Legendre on dense pieces, exact on scattered points
Optimisers: Adam, L-BFGS (scipy), Levenberg-Marquardt (all models)."""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------ time-scale embedding
def graininess_clock(T, t):
    """G(t) = sum of graininess mu(s) over right-scattered s < t, i.e. the total
    gap length before t.  G is constant on every component and continuous on T."""
    t = np.atleast_1d(np.asarray(t, float))
    G = np.zeros_like(t)
    for (q, p) in T.gaps():
        G += np.where(t >= p - 1e-12, p - q, 0.0)
    return G


# ------------------------------------------------------------------ models
def init_mlp(key, width=32, depth=3, d_in=1):
    sizes = [d_in] + [width] * depth + [1]
    params = []
    for k, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k1 = jax.random.split(key)
        lim = np.sqrt(6.0 / (m + n))  # Glorot uniform
        W = jax.random.uniform(k1, (m, n), minval=-lim, maxval=lim)
        params.append((W, jnp.zeros(n)))
    return params


def mlp_apply(params, z):
    """z: (n, d_in) -> (n,)"""
    h = z
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[:, 0]


def make_rf_features(key, n_feat, scale=4.0):
    k1, k2 = jax.random.split(key)
    w = jax.random.uniform(k1, (n_feat,), minval=-scale, maxval=scale)
    b = jax.random.uniform(k2, (n_feat,), minval=-scale, maxval=scale)
    return w, b


class TrialFunction:
    def __init__(self, kind, a, b, xa, key=None, width=32, depth=3, n_feat=64, deg=24, rf_scale=4.0, T=None):
        """T given -> time-scale embedding: the network input is (xi(t), gamma(t)),
        gamma = graininess clock rescaled to [-1,1]."""
        self.kind, self.a, self.b, self.xa = kind, float(a), float(b), float(xa)
        self.T = T
        key = jax.random.PRNGKey(0) if key is None else key
        if T is not None:
            Gmax = max(graininess_clock(T, T.b)[0], 1e-12)
            self.Gmax = Gmax
        if kind == "mlp":
            self.params0 = init_mlp(key, width, depth, d_in=1 if T is None else 2)
        elif kind == "rf":
            self.w, self.bb = make_rf_features(key, n_feat, rf_scale)
            self.params0 = jnp.zeros(n_feat + 1)
        elif kind == "cheb":
            self.deg = deg
            self.params0 = jnp.zeros(deg + 1)
        else:
            raise ValueError(kind)

    def xi(self, t):
        return 2.0 * (t - self.a) / (self.b - self.a) - 1.0

    def gamma(self, t):
        # piecewise-constant graininess clock (continuous on T), as a jax expression
        g = jnp.zeros_like(t)
        for (q, p) in self.T.gaps():
            g = g + jnp.where(t >= p - 1e-12, p - q, 0.0)
        return 2.0 * g / self.Gmax - 1.0

    def N(self, params, t):
        xi = self.xi(t)
        if self.kind == "mlp":
            if self.T is None:
                return mlp_apply(params, xi[:, None])
            return mlp_apply(params, jnp.stack([xi, self.gamma(t)], axis=1))
        if self.kind == "rf":
            return jnp.tanh(xi[:, None] * self.w[None, :] + self.bb[None, :]) @ params[:-1] + params[-1]
        # Chebyshev T_k(xi) by recurrence
        T0, T1 = jnp.ones_like(xi), xi
        cols = [T0, T1]
        for _ in range(2, self.deg + 1):
            T0, T1 = T1, 2 * xi * T1 - T0
            cols.append(T1)
        return jnp.stack(cols[: self.deg + 1], axis=1) @ params

    def v(self, params, t):
        t = jnp.asarray(t)
        return self.xa + (t - self.a) / (self.b - self.a) * self.N(params, t)

    def numpy_fn(self, params):
        f = jax.jit(lambda tt: self.v(params, tt))
        return lambda t: np.asarray(f(jnp.asarray(np.atleast_1d(np.asarray(t, float)))))

    @property
    def n_params(self):
        return int(ravel_pytree(self.params0)[0].size)


class LinearTrial:
    """Trial functions that are affine in the parameters, v = c(t) + Phi(t) beta.
    kind 'rf'    : random tanh features on the hull (extreme learning machine)
    kind 'cheb'  : Chebyshev functional-link network on the hull
    For both, v(t) = f(a) + h_1(t,a)/(b-a) * sum_k beta_k phi_k(xi(t)).
    kind 'pwcheb': time-scale-adapted spectral space: an independent Chebyshev
                   expansion of degree `deg` on every dense component of T and a
                   free value at every isolated point (initial value enforced
                   through the residual)."""

    def __init__(self, kind, T, xa, key=None, n_feat=64, deg=24, rf_scale=4.0):
        self.kind, self.T, self.xa = kind, T, float(xa)
        self.a, self.b = float(T.a), float(T.b)
        self.deg = deg
        if kind == "rf":
            key = jax.random.PRNGKey(0) if key is None else key
            w, bb = make_rf_features(key, n_feat, rf_scale)
            self.w, self.bb = np.asarray(w), np.asarray(bb)
            self.n = n_feat + 1
        elif kind == "cheb":
            self.n = deg + 1
        elif kind == "pwcheb":
            self.blocks = []
            off = 0
            for (p, q) in T.comps:
                k = deg + 1 if q > p else 1
                self.blocks.append((p, q, off, k)); off += k
            self.n = off
        else:
            raise ValueError(kind)
        self.params0 = jnp.zeros(self.n)

    @staticmethod
    def _cheb(xi, deg):
        cols = [np.ones_like(xi), xi]
        for _ in range(2, deg + 1):
            cols.append(2 * xi * cols[-1] - cols[-2])
        return np.stack(cols[: deg + 1], axis=1)

    def design(self, t):
        """returns (c, Phi) with v(t) = c + Phi @ beta"""
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind in ("rf", "cheb"):
            xi = 2.0 * (t - self.a) / (self.b - self.a) - 1.0
            if self.kind == "rf":
                F = np.tanh(xi[:, None] * self.w[None, :] + self.bb[None, :])
                F = np.concatenate([F, np.ones((len(t), 1))], axis=1)
            else:
                F = self._cheb(xi, self.deg)
            h1 = ((t - self.a) / (self.b - self.a))[:, None]
            return np.full(len(t), self.xa), h1 * F
        Phi = np.zeros((len(t), self.n))
        for (p, q, off, k) in self.blocks:
            msk = (t >= p - 1e-12) & (t <= q + 1e-12)
            if not msk.any():
                continue
            if k == 1:
                Phi[msk, off] = 1.0
            else:
                xi = 2.0 * (t[msk] - p) / (q - p) - 1.0
                Phi[np.ix_(msk, np.arange(off, off + k))] = self._cheb(np.clip(xi, -1, 1), self.deg)
        return np.zeros(len(t)), Phi

    def bind(self, pts):
        c, Phi = self.design(np.asarray(pts))
        c, Phi = jnp.asarray(c), jnp.asarray(Phi)
        return lambda beta: c + Phi @ beta

    def numpy_fn(self, params):
        beta = np.asarray(params)

        def fn(t):
            c, Phi = self.design(t)
            return c + Phi @ beta
        return fn

    @property
    def n_params(self):
        return self.n


# ------------------------------------------------------------------ residuals
class DiscreteResidual:
    """R_n(v) = v(t_n) - f(t_n) - sum_{j<n} [wm_j K(t_n,t_j,sigL_j,v_j) + wp_j K(t_n,t_{j+1},sigR_j,v_{j+1})]"""

    def __init__(self, prob, P):
        self.t = jnp.asarray(P.t)
        self.f = jnp.asarray(prob.f(P.t))
        N = P.N
        n = np.arange(N + 1)[:, None]; j = np.arange(N)[None, :]
        self.mask = jnp.asarray((j < n).astype(float))
        self.wm, self.wp = jnp.asarray(P.wm), jnp.asarray(P.wp)
        self.sL, self.sR = jnp.asarray(P.sigL), jnp.asarray(P.sigR)
        self.K, self.Kx = prob.K, prob.Kx
        self.N = N

    def jac_v(self, vals):
        """d r / d vals  ((N+1) x (N+1))"""
        t = self.t[:, None]
        KxL = self.Kx(t, self.t[None, :-1], self.sL[None, :], vals[None, :-1], xp=jnp)
        KxR = self.Kx(t, self.t[None, 1:], self.sR[None, :], vals[None, 1:], xp=jnp)
        A = self.mask * self.wm[None, :] * KxL      # columns k = 0..N-1
        B = self.mask * self.wp[None, :] * KxR      # columns k = 1..N
        z = jnp.zeros((self.N + 1, 1))
        return jnp.eye(self.N + 1) - jnp.concatenate([A, z], axis=1) - jnp.concatenate([z, B], axis=1)

    def __call__(self, vals):
        t = self.t[:, None]
        KL = self.K(t, self.t[None, :-1], self.sL[None, :], vals[None, :-1], xp=jnp)
        KR = self.K(t, self.t[None, 1:], self.sR[None, :], vals[None, 1:], xp=jnp)
        I = jnp.sum(self.mask * (self.wm[None, :] * KL + self.wp[None, :] * KR), axis=1)
        return vals - self.f - I

    def points(self):
        return self.t


class ContinuousResidual:
    """r_v(t_i) at collocation points via a GaussVolterraOperator"""

    def __init__(self, prob, op):
        self.t = jnp.asarray(op.t); self.s = jnp.asarray(op.s)
        self.w = jnp.asarray(op.w); self.sig = jnp.asarray(op.sig)
        self.mask = jnp.asarray(op.mask)
        self.f = jnp.asarray(prob.f(op.t))
        self.K, self.Kx = prob.K, prob.Kx
        self.n_t = len(op.t)

    def jac_v(self, vals_all):
        vs = vals_all[self.n_t:]
        Kxm = self.Kx(self.t[:, None], self.s[None, :], self.sig[None, :], vs[None, :], xp=jnp)
        return jnp.concatenate([jnp.eye(self.n_t), -self.mask * self.w[None, :] * Kxm], axis=1)

    def __call__(self, vals_all):
        vt, vs = vals_all[: self.n_t], vals_all[self.n_t:]
        Kmat = self.K(self.t[:, None], self.s[None, :], self.sig[None, :], vs[None, :], xp=jnp)
        I = jnp.sum(self.mask * self.w[None, :] * Kmat, axis=1)
        return vt - self.f - I

    def points(self):
        return jnp.concatenate([self.t, self.s])


def make_residual_fn(trial, res):
    pts = res.points()
    if isinstance(trial, LinearTrial):
        vfun = trial.bind(np.asarray(pts))

        def rfun(params):
            return res(vfun(params))
        return rfun

    def rfun(params):
        return res(trial.v(params, pts))
    return rfun


# ------------------------------------------------------------------ training
def train_adam(rfun, params, steps=5000, lr=1e-3, lr_end=1e-4, log_every=50, callback=None):
    flat, unravel = ravel_pytree(params)
    loss = jax.jit(lambda p: jnp.mean(rfun(unravel(p)) ** 2))
    vg = jax.jit(jax.value_and_grad(lambda p: jnp.mean(rfun(unravel(p)) ** 2)))
    m = jnp.zeros_like(flat); v = jnp.zeros_like(flat)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    p = flat
    t0 = time.time()
    for k in range(1, steps + 1):
        lr_k = lr * (lr_end / lr) ** ((k - 1) / max(1, steps - 1))
        L, g = vg(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** k); vh = v / (1 - b2 ** k)
        p = p - lr_k * mh / (jnp.sqrt(vh) + eps)
        if k % log_every == 0 or k == 1:
            rec = dict(it=k, loss=float(L), time=time.time() - t0, phase="adam")
            if callback is not None:
                rec.update(callback(unravel(p)))
            hist.append(rec)
    return unravel(p), hist


def train_lbfgs(rfun, params, maxiter=5000, log_every=50, callback=None, it0=0, t_offset=0.0):
    flat, unravel = ravel_pytree(params)
    vg = jax.jit(jax.value_and_grad(lambda p: jnp.mean(rfun(unravel(p)) ** 2)))
    hist = []; cnt = [0]; t0 = time.time()

    def fun(x):
        L, g = vg(jnp.asarray(x))
        return float(L), np.asarray(g, dtype=float)

    def cb(xk):
        cnt[0] += 1
        if cnt[0] % log_every == 0:
            L, _ = vg(jnp.asarray(xk))
            rec = dict(it=it0 + cnt[0], loss=float(L), time=t_offset + time.time() - t0, phase="lbfgs")
            if callback is not None:
                rec.update(callback(unravel(jnp.asarray(xk))))
            hist.append(rec)
    res = minimize(fun, np.asarray(flat, float), jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-30, gtol=1e-30, maxcor=50))
    return unravel(jnp.asarray(res.x)), hist


def train_lm(rfun, params, maxiter=100, lam0=1e-3, tol=1e-30, callback=None, it0=0, t_offset=0.0):
    """Levenberg-Marquardt on the residual vector (exact Jacobian by forward AD)."""
    flat, unravel = ravel_pytree(params)
    r = jax.jit(lambda p: rfun(unravel(p)))
    J = jax.jit(jax.jacfwd(lambda p: rfun(unravel(p))))
    p = flat; lam = lam0
    rv = r(p); L = float(jnp.mean(rv ** 2))
    hist = []; t0 = time.time()
    for k in range(1, maxiter + 1):
        Jm = np.asarray(J(p)); rr = np.asarray(rv)
        A = Jm.T @ Jm; g = Jm.T @ rr
        accepted = False
        for _ in range(30):
            dA = A + lam * np.diag(np.diag(A) + 1e-12 * np.max(np.diag(A)))
            try:
                dp = np.linalg.solve(dA, g)
            except np.linalg.LinAlgError:
                dp = np.linalg.lstsq(dA, g, rcond=None)[0]
            pn = p - jnp.asarray(dp)
            rn = r(pn); Ln = float(jnp.mean(rn ** 2))
            if Ln < L:
                p, rv, L = pn, rn, Ln; lam = max(lam / 3.0, 1e-15); accepted = True
                break
            lam *= 4.0
        rec = dict(it=it0 + k, loss=L, time=t_offset + time.time() - t0, phase="lm", lam=lam)
        if callback is not None:
            rec.update(callback(unravel(p)))
        hist.append(rec)
        if (not accepted) or L < tol:
            break
    if timing is not None:
        timing["iterate"] = time.time() - t0; timing["its"] = k
    return unravel(p), hist


def train_lm_structured(trial, res, params, maxiter=200, lam0=1e-3, tol=1e-30, callback=None,
                        it0=0, t_offset=0.0, log_every=1, timing=None):
    """Levenberg-Marquardt with the structured Jacobian J = (dr/dv)(dv/dtheta).

    If a dict is passed as ``timing``, it receives the wall-clock times of the set-up
    (design matrices), of the just-in-time compilation, and of the iteration itself."""
    ts0 = time.time()
    pts = res.points()
    flat, unravel = ravel_pytree(params)
    if isinstance(trial, LinearTrial):
        c, Phi = trial.design(np.asarray(pts))
        c, Phi = jnp.asarray(c), jnp.asarray(Phi)
        vfun = jax.jit(lambda p: c + Phi @ p)
        dv = jax.jit(lambda p: Phi)
    else:
        vfun = jax.jit(lambda p: trial.v(unravel(p), pts))
        dv = jax.jit(jax.jacfwd(lambda p: trial.v(unravel(p), pts)))
    rfun = jax.jit(lambda p: res(vfun(p)))
    jfun = jax.jit(lambda p: res.jac_v(vfun(p)) @ dv(p))
    p = flat; lam = lam0
    tc0 = time.time()
    rv = rfun(p); L = float(jnp.mean(rv ** 2))
    if timing is not None:
        jax.block_until_ready(jfun(p))
        timing["setup"] = tc0 - ts0; timing["compile"] = time.time() - tc0
    hist = []; t0 = time.time()
    for k in range(1, maxiter + 1):
        Jm = np.asarray(jfun(p)); rr = np.asarray(rv)
        A = Jm.T @ Jm; g = Jm.T @ rr
        dA0 = np.diag(A) + 1e-14 * np.max(np.diag(A))
        accepted = False
        for _ in range(40):
            try:
                dp = np.linalg.solve(A + lam * np.diag(dA0), g)
            except np.linalg.LinAlgError:
                dp = np.linalg.lstsq(A + lam * np.diag(dA0), g, rcond=None)[0]
            pn = p - jnp.asarray(dp)
            rn = rfun(pn); Ln = float(jnp.mean(rn ** 2))
            if np.isfinite(Ln) and Ln < L:
                rel = (L - Ln) / max(L, 1e-300)
                p, rv, L = pn, rn, Ln; lam = max(lam / 3.0, 1e-16); accepted = True
                break
            lam *= 4.0
        if k % log_every == 0 or not accepted:
            rec = dict(it=it0 + k, loss=L, time=t_offset + time.time() - t0, phase="lm", lam=lam)
            if callback is not None:
                rec.update(callback(unravel(p)))
            hist.append(rec)
        if (not accepted) or L < tol:
            break
    if timing is not None:
        timing["iterate"] = time.time() - t0; timing["its"] = k
    return unravel(p), hist
