"""Export every table/figure dataset to results/csv/ and write the LaTeX tables of the revised manuscript
(tables/*.tex) directly from the measured JSON results. Run after the experiments."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); C = os.path.join(R, "csv")
TAB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "tables")
os.makedirs(C, exist_ok=True); os.makedirs(TAB, exist_ok=True)
J = lambda n: json.load(open(os.path.join(R, n + ".json")))
def sci(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "--"
    if v == 0: return "$0$"
    a, b = f"{v:.{d}e}".split("e"); return f"${a}\\times10^{{{int(b)}}}$"

# ---- Table 1 (P1, P2) and figure 3(a,b)
e3 = J("exp3_convergence"); rows = []
for key, name in (("E1", "P1 [0,2]"), ("E2", "P2 T_hyb")):
    rich = {r["lev"]: r["err"] for r in e3[key]["rich"]}
    for r in e3[key]["trap"]:
        rows.append(dict(problem=name, h=r["h"], N=r["N"], kappa=r["kappa"], error=r["err"], richardson=rich.get(r["lev"]),
                         seconds=r["time"], S=r.get("S"), U=r.get("U"), bound=r.get("bound")))
df = pd.DataFrame(rows)
df["eoc"] = df.groupby("problem")["error"].transform(lambda e: np.log2(e.shift(1) / e))
df.to_csv(os.path.join(C, "table1_convergence_P1_P2.csv"), index=False)
pd.DataFrame(e3["E3"]["rows"]).to_csv(os.path.join(C, "table2_qscale_P3.csv"), index=False)
r4 = []
for k in ("trap", "rich", "naive"):
    for r in e3["E4"][k]: r4.append(dict(variant=k, **{a: r.get(a) for a in ("h", "N", "err", "time")}))
pd.DataFrame(r4).to_csv(os.path.join(C, "fig_convergence_d_pendulum_P4.csv"), index=False)
pd.DataFrame(e3["picard"]).to_csv(os.path.join(C, "picard_vs_forward_substitution.csv"), index=False)
# ---- quadrature figure
q = J("exp2_quadrature"); qr = []
for k, v in q["composite"].items():
    for i in range(len(v["err"])): qr.append(dict(case=k, h=(v.get("h") or v.get("H"))[i], err=v["err"][i]))
pd.DataFrame(qr).to_csv(os.path.join(C, "fig_quadrature_c_composite.csv"), index=False)
pd.DataFrame({k: q["random"][k] for k in ("D", "g3", "kappa")}).to_csv(os.path.join(C, "fig_quadrature_b_random_steps.csv"), index=False)
pd.DataFrame([dict(step=k, kappa=v["kappa"], kappa_closed_form=v["kappa_cf"]) for k, v in q["profiles"].items()]).to_csv(os.path.join(C, "fig_quadrature_a_kappa.csv"), index=False)
# ---- overfit, family, timing
pd.DataFrame(J("exp4_overfit")["rows"]).to_csv(os.path.join(C, "fig_overfit.csv"), index=False)
fam = J("exp5_family"); pd.DataFrame(fam["rows"]).to_csv(os.path.join(C, "fig_family_41_nu.csv"), index=False)
# ---- benchmark table (Table 3) from the seeded runs
runs = pd.read_csv(os.path.join(C, "table_benchmark_runs.csv")); cl = pd.DataFrame(J("exp4_classical")["classical"])
abbrev = {"MLP 3x32, loss D": r"MLP $3\times32$, $\mathcal L_D$, A+B", "MLP 3x32, loss C": r"MLP $3\times32$, A+B",
          "MLP-TS 3x32, loss C": r"MLP-TS $3\times32$, A+B", "MLP 2x16, loss C, Adam+LM": r"MLP $2\times16$, A+M",
          "MLP-TS 2x16, loss C, Adam+LM": r"MLP-TS $2\times16$, A+M", "MLP 2x16, loss C, LM": r"MLP $2\times16$, M (random)",
          "MLP-TS 2x16, loss C, LM": r"MLP-TS $2\times16$, M (random)",
          "RF 64, loss C, LM": r"random features $m=64$, M", "RF 128, loss C, LM": r"random features $m=128$, M",
          "RF 256, loss C, LM": r"random features $m=256$, M"}
for d in (10, 20, 30, 40): abbrev[f"Chebyshev FLNN deg {d}, loss C, LM"] = f"Chebyshev network, degree {d}, M"
L = [r"\begin{tabular}{@{}l r r c c c r@{}}", r"\toprule",
     r"method & dof & runs & error: median & error: mean $\pm$ std & time [s]: mean $\pm$ std & $\widehat\eta_{\rm maj}/$error\\", r"\midrule",
     r"\multicolumn{7}{@{}l}{\emph{Grid methods (nodal errors; time: median of 7 serial repetitions after a warm-up)}}\\"]
tc = pd.read_csv(os.path.join(C, "timing_controlled_P5.csv"))
def tctl(method, lev):
    g = tc[(tc.method == method) & (tc.lev == lev)]
    assert len(g) == 1, (method, lev); return float(g["median"].iloc[0]), float(g["err"].iloc[0])
for _, r in cl[cl.family.isin(["nystrom", "richardson", "picard"])].iterrows():
    if r["lev"] in (4, 7, 10) or (r["family"] != "nystrom" and r["lev"] in (4, 7)):
        if r["family"] == "picard" and r["lev"] not in (4, 7): continue
        nm = {"nystrom": "Nystr\\\"om", "richardson": "Richardson", "picard": "Picard"}[r["family"]]
        tmed, terr = tctl({"nystrom": "Nystrom", "richardson": "Richardson", "picard": "Picard"}[r["family"]], int(r["lev"]))
        assert abs(terr / r["err"] - 1) < 1e-6
        L.append(f"\\quad {nm}, $h=2^{{-{int(r['lev'])}}}$ & {int(r['N'])+1} & 1 & {sci(r['err'])} & -- & {sci(tmed)} & --\\\\")
L.append(r"\multicolumn{7}{@{}l}{\emph{Trial functions (sampled errors on $\mathcal E$)}}\\")
for _, r in cl[cl.family.isin(["nystrom_pl", "spectral"])].iterrows():
    if (r["family"] == "nystrom_pl" and r["lev"] in (4, 7)) or (r["family"] == "spectral" and r["deg"] in (8, 12, 16, 20)):
        nm = f"Nystr\\\"om interpolant, $h=2^{{-{int(r['lev'])}}}$" if r["family"] == "nystrom_pl" else f"TS-spectral, degree {int(r['deg'])}"
        dof = "--" if r["family"] == "nystrom_pl" else str(int(r["n_params"]))
        tt = tctl("Nystrom interpolant", int(r["lev"]))[0] if r["family"] == "nystrom_pl" else r["time"]
        L.append(f"\\quad {nm} & {dof} & 1 & {sci(r['err'])} & -- & {sci(tt) if r['family'] == 'nystrom_pl' else format(tt, '.3f')} & {r['eta_maj']/r['err']:.1f}\\\\")
L.append(r"\midrule"); L.append(r"\multicolumn{7}{@{}l}{\emph{Neural trial functions, loss $\mathcal L_C$ unless stated; seeds $0,\dots,4$ ($0,1,2$ for M (random))}}\\")
for lab, g in runs.groupby("label", sort=False):
    e = g["err"]; t = g["time"]; eff = (g["eta_maj"] / g["err"]).median()
    ms = "--" if len(g) == 1 else f"{sci(e.mean())} $\\pm$ {sci(e.std(ddof=1))}"
    ts = f"{t.mean():.2f}" if len(g) == 1 else f"{t.mean():.2f} $\\pm$ {t.std(ddof=1):.2f}"
    L.append(f"\\quad {abbrev.get(lab, lab)} & {int(g['n_params'].iloc[0])} & {len(g)} & {sci(e.median())} & {ms} & {ts} & {eff:.1f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_benchmark.tex"), "w").write("\n".join(L))
# ---- timing table of grid methods (median of repeated timings)
tm = J("exp4_timing"); pd.DataFrame(tm if isinstance(tm, list) else tm.get("rows", [tm])).to_csv(os.path.join(C, "timing_grid_methods.csv"), index=False)
# ---- unusual scales
u = pd.read_csv(os.path.join(C, "table_unusual_scales.csv"))
L = [r"\begin{tabular}{@{}l l r c c c c@{}}", r"\toprule", r"scale & refinement & nodes & error & EOC & a priori bound & Richardson\\", r"\midrule"]
names = {"C4": r"$\mathbb C_4$", "Pab": r"$P_{0.3,0.2}$", "Rnd": r"$\mathbb T_{\rm rnd}$"}
for s in ("C4", "Pab", "Rnd"):
    g = u[u.scale == s]
    for _, r in g.iterrows():
        if int(r["level"]) in (0, 2, 4, 6):
            L.append(f"{names[s]} & $h_d={r['h']:.5f}$ & {int(r['N'])+1} & {sci(r['error'])} & " + ("--" if pd.isna(r["eoc"]) else f"{r['eoc']:.2f}") +
                     f" & {sci(r['bound'])} & {sci(r['richardson']) if not pd.isna(r['richardson']) else '--'}\\\\")
L += [r"\midrule", r"scale & partition & nodes & error & & $\kappa(\mathcal P)$ & error$/\kappa$\\", r"\midrule"]
for s, nm in (("Harm", r"$\{H_n\}_{n\le400}$"), ("Sqrt", r"$\sqrt{\N_0}\cap[0,20]$")):
    for _, r in u[u.scale == s].iterrows():
        m = int(r["m"]); err = r["error"]
        txt = r"$\gamma=%.2f\ge1$: excluded by \eqref{eq:gamma}" % r["gamma"] if pd.isna(err) else sci(err)
        lab = "all points" if m == 1 else f"stride $m={m}$"
        L.append(f"{nm} & {lab} & {int(r['N'])+1} & {txt} & & " + (sci(r["kappa"]) if r["kappa"] > 0 else "$0$") +
                 " & " + ("--" if pd.isna(r["error_over_kappa"]) else f"{r['error_over_kappa']:.2f}") + "\\\\")
hp = pd.read_csv(os.path.join(C, "qacc_highprec.csv"))
L += [r"\midrule", r"scale & truncation & points & error (double) & error (70 dig.) & bound & ratio (70 dig.)\\", r"\midrule"]
for _, r in u[u.scale == "Qacc"].iterrows():
    h = hp[hp.n == int(r["n"])].iloc[0]
    L.append(f"$0.9^{{\\N_0}}\\cup\\{{0\\}}$ & $n={int(r['n'])}$ & {int(r['n'])+1} & {sci(r['error'])} & {sci(float(h['error']))} & {sci(float(h['bound']))} & {float(h['ratio']):.3f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_unusual.tex"), "w").write("\n".join(L))
# ---- noise
vr = os.path.join(HERE, "..", "results", "csv", "noise_robustness.csv")
nz = pd.read_csv(vr); nz.to_csv(os.path.join(C, "fig_noise_robustness.csv"), index=False)
L = [r"\begin{tabular}{@{}c c c c c@{}}", r"\toprule", r"rel.\ std $\epsilon$ & rel.\ nodal $\ell^2$ error (mean $\pm$ std) & max error (mean $\pm$ std) & max ratio dev$/(\bar U_N\|\delta\|_\infty)$ & seeds\\", r"\midrule"]
for lvl, g in nz.groupby("noise"):
    sd = (lambda v: "0" if lvl == 0 else sci(v))
    L.append(f"{100*lvl:.0f}\\% & {sci(g.rel_L2_error.mean())} $\\pm$ {sd(g.rel_L2_error.std(ddof=1))} & {sci(g.max_abs_error.mean())} $\\pm$ {sd(g.max_abs_error.std(ddof=1))} & " +
             ("--" if lvl == 0 else f"{g.ratio.max():.3f}") + f" & {len(g)}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_noise.tex"), "w").write("\n".join(L))
print("written", sorted(os.listdir(C)), sorted(os.listdir(TAB)))
