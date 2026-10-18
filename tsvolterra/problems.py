"""Benchmark problems.  Kernels: K(t, s, sig, x, xp=np) with sig = sigma(s)
(or the left limit s on dense steps).  All kernels are globally Lipschitz in x."""
from __future__ import annotations
import numpy as np
from .timescale import TimeScale


class VolterraProblem:
    def __init__(self, T: TimeScale, K, Kx, L, x_exact=None, f=None, name="", n_gauss=16, panel=0.05):
        self.T, self.K, self.Kx, self.L = T, K, Kx, L
        self.x_exact, self._f, self.name = x_exact, f, name
        self.n_gauss, self.panel = n_gauss, panel
        self._cache = {}

    @property
    def a(self):
        return self.T.a

    @property
    def b(self):
        return self.T.b

    def integral_of(self, v, t):
        """I_v(t) = int_a^t K(t,s,sigma(s),v(s)) Delta s for scalar t (accurate Gauss)"""
        T = self.T
        x, w = np.polynomial.legendre.leggauss(self.n_gauss)
        tot = 0.0
        for kind, u, d in T.pieces(T.a, t):
            if kind == "d":
                n = max(1, int(np.ceil((d - u) / self.panel)))
                e = np.linspace(u, d, n + 1)
                S = (0.5 * np.diff(e)[:, None] * x[None, :] + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
                Wt = (0.5 * np.diff(e)[:, None] * w[None, :]).ravel()
                tot += np.sum(Wt * self.K(t, S, S, v(S)))
            else:
                tau, mu = u, d
                tot += mu * self.K(t, np.array([tau]), np.array([tau + mu]), v(np.array([tau])))[0]
        return tot

    def f(self, t):
        if self._f is not None:
            return self._f(np.asarray(t, float))
        t = np.atleast_1d(np.asarray(t, float))
        out = np.empty_like(t)
        for i, ti in enumerate(t):
            key = float(ti)
            if key not in self._cache:
                self._cache[key] = float(self.x_exact(np.array([ti]))[0] - self.integral_of(self.x_exact, ti))
            out[i] = self._cache[key]
        return out


# ------------------------------------------------------------------ kernels
def kernel_exp_sin(lam, alpha):
    def K(t, s, sig, x, xp=np):
        return lam * xp.exp(-alpha * (t - s)) * xp.sin(x)

    def Kx(t, s, sig, x, xp=np):
        return lam * xp.exp(-alpha * (t - s)) * xp.cos(x)
    return K, Kx, abs(lam)  # sup_{s<=t} |dK/dx| <= |lam|


def kernel_rational(lam, alpha):
    """lam*exp(-alpha(t-s)) * x/(1+x^2) ; |d/dx x/(1+x^2)| <= 1"""
    def K(t, s, sig, x, xp=np):
        return lam * xp.exp(-alpha * (t - s)) * x / (1.0 + x * x)

    def Kx(t, s, sig, x, xp=np):
        return lam * xp.exp(-alpha * (t - s)) * (1.0 - x * x) / (1.0 + x * x) ** 2
    return K, Kx, abs(lam)


def kernel_pendulum(omega, tmax):
    """Volterra form of x^{DD} = -omega^2 sin x :  K = -omega^2 (t - sigma(s)) sin x"""
    def K(t, s, sig, x, xp=np):
        return -omega ** 2 * (t - sig) * xp.sin(x)

    def Kx(t, s, sig, x, xp=np):
        return -omega ** 2 * (t - sig) * xp.cos(x)
    return K, Kx, omega ** 2 * tmax


# ------------------------------------------------------------ time scales
def ts_interval(a, b):
    return TimeScale([(a, b)])


def ts_hybrid():
    """[0,1] u {1.25,1.5,1.75} u [2,3] u {3.5} u [4,4.5]"""
    return TimeScale([(0, 1), (1.25, 1.25), (1.5, 1.5), (1.75, 1.75), (2, 3), (3.5, 3.5), (4, 4.5)])


def ts_qscale(q=1.002, t1=1.0, t2=3.0):
    k = int(np.floor(np.log(t2 / t1) / np.log(q) + 1e-9))
    return TimeScale.from_points(t1 * q ** np.arange(k + 1))


def ts_pendulum():
    return TimeScale([(0, 1.5), (1.75, 1.75), (2.0, 2.0), (2.5, 4.0), (4.25, 4.25), (4.5, 4.5), (4.75, 4.75), (5.0, 6.0)])


def ts_family(nu, K=4):
    """T_nu = U_{k=0}^{K-1} [k, k+nu] U {K}; nu in [0,1]. nu=0: {0,..,K}; nu=1: [0,K]"""
    if nu >= 1.0:
        return TimeScale([(0.0, float(K))])
    comps = []
    for k in range(K):
        comps.append((float(k), float(k) + nu))
    comps.append((float(K), float(K)))
    return TimeScale(comps)


# --------------------------------------------------------- exact solutions
def x_star(t):
    t = np.asarray(t, float)
    return 1.0 + np.exp(-t / 2.0) * np.sin(3.0 * t)


def x_star_jax(t):
    import jax.numpy as jnp
    return 1.0 + jnp.exp(-t / 2.0) * jnp.sin(3.0 * t)
