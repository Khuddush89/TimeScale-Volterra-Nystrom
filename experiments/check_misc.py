"""Two auxiliary numbers quoted in the paper:
(1) conditioning and numerical rank of the random-feature collocation matrix in the proof of the
    existence theorem for networks (Section 6.3), for the gap-conforming partition of T_hyb with
    h = 2^-4 (N = 46) and m = N features drawn as in Section 7.4 (keys 100-104);
(2) Lambda(t) = int_0^t exp(-(t-s)) Delta s on T_hyb, which enters the lower bound of the
    two-sided a posteriori estimate (Section 7.4)."""
from common import *
import jax
from tsvolterra.problems import ts_hybrid
from tsvolterra.timescale import Partition
from tsvolterra.neural import make_rf_features


def rf_conditioning():
    T = ts_hybrid()
    P = Partition.gap_conforming(T, counts=[int(round(2 ** 4 * (q - p))) if q > p else 0 for (p, q) in T.comps])
    t = P.t[1:]; N = len(t)
    xi = 2 * (t - T.a) / (T.b - T.a) - 1
    rows = []
    for seed in range(5):
        w, c = make_rf_features(jax.random.PRNGKey(100 + seed), N, 4.0)
        A = np.tanh(np.outer(xi, np.asarray(w)) + np.asarray(c)[None, :])
        s = np.linalg.svd(A, compute_uv=False)
        rows.append(dict(key=100 + seed, N=N, cond=float(s[0] / s[-1]), rank=int(np.linalg.matrix_rank(A))))
        print(rows[-1], flush=True)
    return rows


def Lambda_max():
    T = ts_hybrid()

    def Lam(t):
        tot = 0.0
        for (p, q) in T.comps:
            if q > p and p < t:
                u = min(q, t); tot += np.exp(-(t - u)) - np.exp(-(t - p))
        for (q, p) in T.gaps():
            if q < t:
                tot += (p - q) * np.exp(-(t - q))
        return tot
    ts = T.sample(1024)
    L = np.array([Lam(t) for t in ts])
    out = dict(Lambda_max=float(L.max()), t_at_max=float(ts[np.argmax(L)]), Lambda_b=float(Lam(T.b)))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    save_json("misc_checks", dict(rf=rf_conditioning(), Lambda=Lambda_max()))
