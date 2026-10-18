"""README banner drawn from the computed solutions of experiment 7 (no synthetic data)."""
from common import *
import matplotlib.pyplot as plt
from tsvolterra.tsplot import DARK, plot_on_scale
plt.rcParams.update(DARK)
D = load_json("exp7_unusual")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(wspace=0.05)); fig.subplots_adjust(top=0.62, bottom=0.02)
for ax, (k, cm) in zip(axs, [("C4", "turbo"), ("Rnd", "plasma"), ("Pab", "cool")]):
    t = np.array(D[k]["profile"]["t"]); x = np.array(D[k]["profile"]["x"])
    plot_on_scale(ax, [tuple(c) for c in D[k]["comps"]], t, x, cmap=cm, lw=2.6, ms=7)
    ax.axis("off")
fig.text(0.5, 0.86, "TimeScale-Volterra-Nystrom", ha="center", fontsize=22, weight="bold", color="#e6edf3")
fig.text(0.5, 0.72, r"$x(t)=f(t)+\int_a^t K(t,s,x(s))\,\Delta s$   on Cantor prefractals, random unions, $P_{\alpha,\beta}$ ...",
         ha="center", fontsize=12, color="#c9d1d9")
fig.savefig(os.path.join(ROOT, "assets", "banner.png"), dpi=130, bbox_inches="tight")
print("banner written")
