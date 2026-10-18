"""Check that the numerically computed majorants used in Sections 7.4 and 7.5 are majorants:
Gamma(t) >= 1 + int_a^t exp(-(t-s)) Gamma(s) Delta s at every point of the evaluation set,
where Gamma = 1.001 * (Richardson-extrapolated Nystrom solution of u = 1 + int ell u), extended
between the nodes by linear interpolation on the dense steps.  Also records the explicit majorant
1 + (t - a) (valid on every time scale for ell(t,s) = exp(-(t-s)), see the remark after the two-sided a posteriori estimate)."""
from common import *
from tsvolterra.quadrature import GaussVolterraOperator
from tsvolterra.problems import ts_hybrid, ts_family
from tsvolterra.timescale import Partition
from tsvolterra import nystrom
from tsvolterra.problems import VolterraProblem

Kmaj = lambda t, s, sg, x, xp=np: xp.exp(-(t - s)) * x


def defect(T, nodes, G, n_gauss=8):
    """min over nodes of Gamma - 1 - int ell Gamma (Gamma linear between nodes on dense steps)"""
    Gf = lambda t: np.interp(np.atleast_1d(t), nodes, G)
    op = GaussVolterraOperator(T, nodes, n_gauss=n_gauss, max_panel=1.0)
    assert np.allclose(op.t, nodes)
    I = op.apply_chunked(Kmaj, Gf(op.s))
    d = Gf(op.t) - 1.0 - I
    return float(np.min(d)), float(np.max(G))


def explicit_defect(T, nodes, n_gauss=8):
    G = 1.0 + (nodes - T.a)
    return defect(T, nodes, G, n_gauss)


def run():
    out = {}
    # Section 7.4: T_hyb, evaluation set E = nodes with h = 2^-10
    T = ts_hybrid()
    cnt = lambda lev: [int(round(2 ** lev * (q - p))) if q > p else 0 for (p, q) in T.comps]
    P10 = Partition.gap_conforming(T, counts=cnt(10)); P11 = Partition.gap_conforming(T, counts=cnt(11))
    pm = VolterraProblem(T, Kmaj, lambda t, s, sg, x, xp=np: xp.exp(-(t - s)) + 0 * x, 1.0, f=lambda t: 1.0 + 0 * np.asarray(t))
    U10 = nystrom.solve(pm, P10); U11 = nystrom.solve(pm, P11)
    R = nystrom.richardson(U10, nystrom.restrict(P11, U11, P10))
    m0, _ = defect(T, P10.t, R)
    m1, gmax = defect(T, P10.t, 1.001 * R)
    me, gexp = explicit_defect(T, P10.t)
    out["hyb"] = dict(min_defect_unscaled=m0, min_defect=m1, Gamma_max=gmax, explicit_min_defect=me, explicit_max=gexp)
    print("T_hyb: min defect of u (unscaled) %.2e, of 1.001u %.2e (max Gamma %.4f); explicit 1+t: min defect %.2e, max %.2f"
          % (m0, m1, gmax, me, gexp), flush=True)
    # Section 7.5: the family T_nu, nodes of the reference partitions
    from exp5_family import majorant
    rows = []
    for nu in np.linspace(0.0, 1.0, 41):
        P, G = majorant(nu)
        m, gmax = defect(ts_family(nu, 4), P.t, G)
        rows.append(dict(nu=float(nu), min_defect=m, Gamma_max=gmax))
    out["family"] = rows
    mins = [r["min_defect"] for r in rows]
    print("family: min defect over all nu %.2e (all >= 0: %s)" % (min(mins), all(x >= 0 for x in mins)), flush=True)
    save_json("majorant_check", out)
    return out


if __name__ == "__main__":
    run()
