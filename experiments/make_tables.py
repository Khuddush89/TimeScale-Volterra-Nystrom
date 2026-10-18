"""Generate the LaTeX tables of the paper directly from the result files."""
from common import *

TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)


def sci(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return f"${m:.{d}f}\\mathrm{{e}}{{{e:+d}}}$".replace("e{+", "e{").replace("{-0", "{-").replace("e{0", "e{0")


def sci1(x):
    """one decimal: 1.2e-4"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, 1) >= 10: m /= 10; e += 1
    return f"{m:.1f}\\text{{e}}{e:+03d}".replace("+0", "+").replace("-0", "-")


def sci2(x):
    """compact scientific notation a.bc(-k) style -> use 1.23e-4 typeset as 1.23\\,(-4)"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, 2) >= 10: m /= 10; e += 1
    return f"{m:.2f}\\text{{e}}{e:+03d}".replace("+0", "+").replace("-0", "-")


def tab_convergence():
    d = load_json("exp3_convergence")
    E1, E2 = d["E1"], d["E2"]
    rich1 = {r["lev"]: r for r in E1["rich"]}; rich2 = {r["lev"]: r for r in E2["rich"]}
    lines = []
    for r1, r2 in zip(E1["trap"], E2["trap"]):
        lev = r1["lev"]
        a = rich1.get(lev); b = rich2.get(lev)
        lines.append(" & ".join([
            f"$2^{{-{lev}}}$",
            f"${sci2(r1['err'])}$", f"{r1['eoc']:.2f}" if "eoc" in r1 else "--",
            f"${sci2(a['err'])}$" if a else "--", f"{a['eoc']:.2f}" if a and "eoc" in a else "--",
            f"${sci2(r2['err'])}$", f"{r2['eoc']:.2f}" if "eoc" in r2 else "--",
            f"${sci2(b['err'])}$" if b else "--", f"{b['eoc']:.2f}" if b and "eoc" in b else "--",
            f"{r2['B2m'] / r2['err']:.0f}"]) + r" \\")
    body = "\n".join(lines)
    tex = r"""\begin{tabular}{c cc cc cc cc r}
\toprule
 & \multicolumn{4}{c}{$\T=[0,2]$} & \multicolumn{4}{c}{$\T=\Thyb$} & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
$h$ & trapezoid & EOC & Richardson & EOC & trapezoid & EOC & Richardson & EOC & $\dfrac{\bar U_N\kappa(\cP)M_2}{\text{error}}$\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(TAB, "tab_convergence.tex"), "w").write(tex)


def tab_qscale():
    d = load_json("exp3_convergence")["E3"]
    lines = []
    prev = None
    for r in d["rows"]:
        eoc = "--"
        if prev is not None and r["err"] > 0 and prev["err"] > 0:
            eoc = f"{np.log(r['err'] / prev['err']) / np.log(r['H'] / prev['H']):.2f}"
        lines.append(" & ".join([str(r["m"]), str(r["N"]), f"{r['H']:.4f}", f"${sci2(r['err'])}$" if r["err"] > 0 else "$0$",
                                 eoc, f"${sci2(r['kappa'])}$" if r["kappa"] > 0 else "$0$",
                                 f"{r['U']:.2f}", f"${sci2(r['B2m'])}$" if r["B2m"] > 0 else "$0$"]) + r" \\")
        prev = r
    tex = r"""\begin{tabular}{rrcccccc}
\toprule
$m$ & $N$ & $H$ & $\max_n|x(t_n)-X_n|$ & EOC in $H$ & $\kappa(\cP)$ & $\bar U_N$ & $\bar U_N\kappa(\cP)M_2$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(TAB, "tab_qscale.tex"), "w").write(tex)


def summarize(rows, fam):
    rr = [r for r in rows if r["family"] == fam]
    if not rr:
        return None
    errs = np.array([r["err"] for r in rr]); times = np.array([r["time"] for r in rr])
    effm = np.array([r["eta_maj"] / r["err"] for r in rr]); effe = np.array([r["eta_exp"] / r["err"] for r in rr])
    i = int(np.argsort(errs)[len(errs) // 2])
    return dict(n=len(rr), p=rr[0].get("n_params", None), err_med=float(np.median(errs)), err_min=float(errs.min()),
                err_max=float(errs.max()), t_med=float(np.median(times)), effm=float(np.median(effm)),
                effe=float(np.median(effe)), eta_maj=float(np.median([r["eta_maj"] for r in rr])),
                r_med=float(np.median([r["sup_r"] for r in rr])))


def tab_benchmark():
    cl = load_json("exp4_classical")["classical"]
    nn = load_json("exp4_neural")["rows"]
    try:
        nn += load_json("exp4_neural_lm")["rows"]
    except FileNotFoundError:
        pass
    lines = []
    def row(name, p, t, err, rng, r, eta, eff):
        return " & ".join([name, p, t, err, rng, r, eta, eff]) + r" \\"
    def head(txt):
        return r"\multicolumn{8}{@{}l}{\emph{" + txt + r"}}\\[1pt]"
    pick = lambda fam, key, val: [r for r in cl if r["family"] == fam and r.get(key) == val][0]
    lines.append(head("Grid methods (errors at the nodes)"))
    for lev in [4, 7, 10]:
        r = pick("nystrom", "lev", lev)
        lines.append(row(f"\\quad Nystr\\\"om (forward subst.), $h=2^{{-{lev}}}$", f"{r['N']+1}", f"{r['time']:.3f}", f"${sci2(r['err'])}$", "", "--", "--", "--"))
    for lev in [4, 7]:
        r = pick("picard", "lev", lev)
        lines.append(row(f"\\quad Picard ({r['its']} sweeps), $h=2^{{-{lev}}}$", f"{r['N']+1}", f"{r['time']:.3f}", f"${sci2(r['err'])}$", "", "--", "--", "--"))
    for lev in [4, 7]:
        r = pick("richardson", "lev", lev)
        lines.append(row(f"\\quad Richardson, $h=2^{{-{lev}}}$", f"{r['N']+1}", f"{r['time']:.3f}", f"${sci2(r['err'])}$", "", "--", "--", "--"))
    lines.append(head(r"Certified continuous approximations (errors on $\mathcal E$)"))
    for lev in [4, 7]:
        r = pick("nystrom_pl", "lev", lev)
        lines.append(row(f"\\quad Nystr\\\"om interpolant, $h=2^{{-{lev}}}$", "--", f"{r['time']:.3f}", f"${sci2(r['err'])}$", "",
                         f"${sci2(r['sup_r'])}$", f"${sci2(r['eta_maj'])}$", f"{r['eta_maj']/r['err']:.1f}"))
    for deg in [8, 12, 16, 20]:
        r = pick("spectral", "deg", deg)
        lines.append(row(f"\\quad TS-spectral, degree {deg}", f"{r['n_params']}", f"{r['time']:.2f}", f"${sci2(r['err'])}$", "",
                         f"${sci2(r['sup_r'])}$", f"${sci2(r['eta_maj'])}$", f"{r['eta_maj']/r['err']:.1f}"))
    lines.append(r"\midrule")
    lines.append(head(r"Multilayer perceptrons, loss $\mathcal L_C$ unless stated otherwise"))
    names = [("mlp_D", r"\quad MLP $3\times32$, $\mathcal L_D$, Adam+L-BFGS"),
             ("mlp_C", r"\quad MLP $3\times32$, Adam+L-BFGS"),
             ("mlpts_C", r"\quad MLP-TS $3\times32$, Adam+L-BFGS"),
             ("mlp_C_alm", r"\quad MLP $2\times16$, Adam+LM"),
             ("mlpts_C_alm", r"\quad MLP-TS $2\times16$, Adam+LM"),
             ("mlp_C_lm", r"\quad MLP $2\times16$, LM, random init.$^{\dagger}$"),
             ("mlpts_C_lm", r"\quad MLP-TS $2\times16$, LM, random init.$^{\dagger}$")]
    allrows = nn
    for fam, label in names:
        s_ = summarize(allrows, fam)
        if s_ is None:
            continue
        lines.append(row(label, f"{s_['p']}", f"{s_['t_med']:.1f}", f"${sci2(s_['err_med'])}$",
                         f"$[{sci1(s_['err_min'])},\\,{sci1(s_['err_max'])}]$", f"${sci2(s_['r_med'])}$", f"${sci2(s_['eta_maj'])}$", f"{s_['effm']:.1f}"))
    lines.append(head(r"Networks linear in the trained parameters, loss $\mathcal L_C$, LM"))
    for nf in [64, 128, 256]:
        rr = [r for r in allrows if r["family"] == "rf" and r.get("n_feat") == nf]
        s_ = summarize([dict(r, family="x") for r in rr], "x")
        lines.append(row(f"\\quad random features, $m={nf}$", f"{nf+1}", f"{s_['t_med']:.1f}", f"${sci2(s_['err_med'])}$",
                         f"$[{sci1(s_['err_min'])},\\,{sci1(s_['err_max'])}]$", f"${sci2(s_['r_med'])}$", f"${sci2(s_['eta_maj'])}$", f"{s_['effm']:.1f}"))
    for deg in [10, 20, 30, 40]:
        rr = [r for r in allrows if r["family"] == "cheb" and r.get("deg") == deg]
        if not rr: continue
        r = rr[0]
        lines.append(row(f"\\quad Chebyshev network, degree {deg}", f"{deg+1}", f"{r['time']:.1f}", f"${sci2(r['err'])}$", "",
                         f"${sci2(r['sup_r'])}$", f"${sci2(r['eta_maj'])}$", f"{r['eta_maj']/r['err']:.1f}"))
    tex = r"""\begin{tabular}{@{}l r r c c c c r@{}}
\toprule
method & dof & time [s] & error & range over seeds & $\sup|r_v|$ & $\eta_{\rm maj}$ & $\dfrac{\eta_{\rm maj}}{\text{error}}$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(TAB, "tab_benchmark.tex"), "w").write(tex)


if __name__ == "__main__":
    tab_convergence(); tab_qscale()
    try:
        tab_benchmark()
    except FileNotFoundError as e:
        print("benchmark table skipped:", e)
    print("tables written")
