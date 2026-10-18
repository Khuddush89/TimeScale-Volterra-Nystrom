"""High-precision check of the truncation study on {0} U q^{N_0} (Proposition 8.1, Table 5 bottom block).

Kernel K(t,s,x) = exp(-(t-s)) sin x, f(t) = 1 + sin(2t)/2, q = 9/10 exactly, 70 significant digits (mpmath).
The delta sum is evaluated with the separable form  X_i = f(t_i) + e^{-t_i} S_i,  S_{i+1} = S_i + mu_i e^{t_i} sin X_i,
which is algebraically the same scattered delta sum as the direct double loop.
This is a high-precision floating-point computation, not an interval-arithmetic certificate.
Writes results/qacc_highprec.json and results/csv/qacc_highprec.csv."""
import os, json, csv, time
import mpmath as mp

mp.mp.dps = 70
q = mp.mpf(9) / 10
NLIST = (10, 20, 40, 80, 160, 320); NREF = 800
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solve(n):
    t = [q ** (n - i) for i in range(n + 1)]
    mu = [t[i + 1] - t[i] for i in range(n)]
    X = []; S = mp.mpf(0)
    for i in range(n + 1):
        X.append(1 + mp.sin(2 * t[i]) / 2 + mp.e ** (-t[i]) * S)
        if i < n:
            S += mu[i] * mp.e ** (t[i]) * mp.sin(X[i])
    return t, X, mu


def main():
    t0 = time.time()
    tr, Xr, mur = solve(NREF)
    ref_bound = q ** NREF * mp.fprod([1 + m for m in mur])
    rows = []
    for n in NLIST:
        t, X, mu = solve(n)
        off = NREF - n
        err = max(abs(X[i] - Xr[off + i]) for i in range(n + 1))
        bound = q ** n * mp.fprod([1 + m for m in mu])
        rows.append(dict(n=n, error=mp.nstr(err, 15), bound=mp.nstr(bound, 15), ratio=mp.nstr(err / bound, 8)))
        print(rows[-1], flush=True)
    out = dict(dps=mp.mp.dps, q="9/10", nref=NREF, ref_truncation_bound=mp.nstr(ref_bound, 6), rows=rows,
               seconds=time.time() - t0)
    os.makedirs(os.path.join(ROOT, "results", "csv"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "qacc_highprec.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "results", "csv", "qacc_highprec.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("reference truncation bound", out["ref_truncation_bound"], "time %.2f s" % out["seconds"])


if __name__ == "__main__":
    main()
