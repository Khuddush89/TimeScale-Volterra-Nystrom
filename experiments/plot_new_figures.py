"""Figures for the unusual time scales (exp7) and data-perturbation test (verification/noise).
Run after exp7_unusual.py and ../verification/verify_independent.py."""
from common import *
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tsvolterra.tsplot import PAPER, plot_on_scale, component_colors, scale_strip

plt.rcParams.update(PAPER)
D = load_json("exp7_unusual")
NAMES = {"C4": r"(a) Cantor prefractal $C_4\subset[0,3]$", "Pab": r"(b) $P_{0.3,0.2}=\bigcup_k[0.5k,0.5k+0.3]$",
         "Rnd": r"(c) random irregular scale (seed 2026)", "Harm": r"(d) harmonic numbers $\{H_n\}_{n=1}^{400}$",
         "Sqrt": r"(e) $\sqrt{\mathbb{N}_0}\cap[0,20]$", "Qacc": r"(f) $0.9^{\mathbb{N}_0}\cup\{0\}$ (0 right-dense)"}


def gallery():
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6), layout="constrained")
    for ax, key in zip(axs.ravel(), NAMES):
        d = D[key]; t = np.array(d["profile"]["t"]); x = np.array(d["profile"]["x"])
        if key in ("C4", "Pab", "Rnd"):
            comps = [tuple(c) for c in d["comps"]]
            cols = plot_on_scale(ax, comps, t, x, cmap="turbo", ms=4)
            lo = x.min() - 0.08 * (x.max() - x.min()); scale_strip(ax, comps, lo, colors=cols)
        else:
            if key == "Qacc":
                t = np.concatenate([[0.0], t]); x = np.concatenate([[1.0], x])
            cols = component_colors(len(t), "plasma")
            cval = np.log10(np.maximum(t, 1e-6)) if key == "Qacc" else np.linspace(0, 1, len(t))
            ax.scatter(t, x, c=cval, cmap="plasma", s=7 if key != "Qacc" else 9, zorder=3, linewidths=0)
            if key == "Qacc":
                ax.set_xscale("symlog", linthresh=1e-3)
            if key == "Qacc":
                ax.plot([0], [1.0], "D", color="#00b894", ms=6, zorder=4, label=r"$t=0$: $x(0)=f(0)$")
                ax.legend(loc="lower right", fontsize=7)
        ax.set_title(NAMES[key], fontsize=8.5); ax.set_xlabel(r"$t\in\mathbb{T}$"); ax.set_ylabel(r"$x(t)$")
    fig.savefig(os.path.join(FIG, "fig_gallery.pdf")); fig.savefig(os.path.join(FIG, "fig_gallery.png"), dpi=400)
    plt.close(fig)


def convergence():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained")
    cols = {"C4": "#e84393", "Pab": "#0984e3", "Rnd": "#00b894"}
    for key, c in cols.items():
        rows = D[key]["rows"]; h = np.array([r["h"] for r in rows])
        axs[0].loglog(h, [r["error"] for r in rows], "o-", color=c, label=key + " trapezoid")
        axs[0].loglog(h, [r["bound"] for r in rows], ":", color=c, lw=1)
        rr = [r for r in rows if "richardson" in r]
        axs[0].loglog([r["h"] for r in rr], [r["richardson"] for r in rr], "s--", color=c, ms=3.5, label=key + " Richardson")
    axs[0].set(xlabel="largest dense step $h(\\mathcal{P})$", ylabel=r"$\max_n|x(t_n)-X_n|$",
               title="(a) dense refinement, gaps fixed (dotted: bound)")
    axs[0].legend(fontsize=6.5, ncol=1)
    for key, c, mk in [("Harm", "#6c5ce7", "o"), ("Sqrt", "#fd7e14", "D")]:
        rows = [r for r in D[key]["rows"] if r["error"] is not None and r["m"] > 1]
        axs[1].loglog([r["kappa"] for r in rows], [r["error"] for r in rows], mk + "-", color=c, label=key)
        for r in rows:
            axs[1].annotate(f"m={r['m']}", (r["kappa"], r["error"]), fontsize=6, xytext=(3, -8), textcoords="offset points")
    k = np.logspace(-2, 0.5); axs[1].loglog(k, 0.35 * k, "k:", lw=0.8, label=r"$0.35\,\kappa(\mathcal{P})$")
    axs[1].set(xlabel=r"Peano constant $\kappa(\mathcal{P})$", ylabel="max nodal error",
               title="(b) coarse partitions of discrete scales"); axs[1].legend(fontsize=7)
    rows = D["Qacc"]["rows"]
    axs[2].semilogy([r["n"] for r in rows], [r["error"] for r in rows], "o-", color="#d63031", label="truncation error")
    axs[2].semilogy([r["n"] for r in rows], [r["bound"] for r in rows], "s--", color="#2d3436", label=r"bound $\lambda q^n\prod(1+\lambda\mu_j)$")
    axs[2].set(xlabel=r"retained points $q^n,\dots,1$", ylabel="max error", title=r"(c) accumulation point: $0.9^{\mathbb{N}_0}\cup\{0\}$")
    axs[2].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_unusual_convergence.pdf")); fig.savefig(os.path.join(FIG, "fig_unusual_convergence.png"), dpi=400)
    plt.close(fig)


def noise():
    path = os.path.join(ROOT, "results", "csv", "noise_robustness.csv")
    rows = list(csv.DictReader(open(path)))
    lv = sorted({float(r["noise"]) for r in rows if float(r["noise"]) > 0})
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
    for i, l in enumerate(lv):
        rr = [r for r in rows if float(r["noise"]) == l]
        ax[0].scatter([float(r["bound"]) for r in rr], [float(r["max_abs_dev"]) for r in rr], s=14,
                      color=plt.cm.viridis(0.15 + 0.35 * i), label=f"{100*l:.0f}% noise")
    z = np.logspace(-3, 0); ax[0].loglog(z, z, "k--", lw=0.8, label="bound = deviation")
    ax[0].set(xscale="log", yscale="log", xlabel=r"$\bar U_N\,\max_n|\delta_n|$", ylabel=r"$\max_n|\tilde X_n-X_n|$",
              title="(a) discrete stability bound under data noise"); ax[0].legend(fontsize=7)
    m = [np.mean([float(r["rel_L2_error"]) for r in rows if float(r["noise"]) == l]) for l in [0] + lv]
    s = [np.std([float(r["rel_L2_error"]) for r in rows if float(r["noise"]) == l]) for l in [0] + lv]
    ax[1].errorbar([0] + [100 * l for l in lv], m, yerr=s, fmt="o-", color="#e17055", capsize=3)
    ax[1].set(xlabel="noise level [%]", ylabel=r"relative $\ell^2$ error vs $x_*$", title=r"(b) $\mathbb{T}_{\rm irr}$, $h_d=0.0125$, 10 seeds")
    fig.savefig(os.path.join(FIG, "fig_noise.pdf")); fig.savefig(os.path.join(FIG, "fig_noise.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    gallery(); convergence(); noise(); print("figures written to", FIG)
