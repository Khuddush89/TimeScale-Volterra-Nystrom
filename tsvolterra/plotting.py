"""Figure style shared by all experiments (journal PDF figures).
Categorical colours follow a validated colour-blind-safe order (blue, orange,
aqua, yellow, magenta, green, violet, red); every series also has its own
marker so identity never relies on colour alone."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
GAP = "#f0efec"
BLUES = LinearSegmentedColormap.from_list(
    "blues_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.2,
        "lines.markeredgewidth": 0.7, "lines.markeredgecolor": "white",
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def slope_triangle(ax, x0, y0, slope, dx_decades=0.3, label=None, color=MUTED, below=True):
    """small reference triangle on log-log axes"""
    import numpy as np
    x1 = x0 * 10 ** dx_decades
    y1 = y0 * (x1 / x0) ** slope
    ax.plot([x0, x1], [y0, y1], color=color, lw=0.8)
    if below:
        ax.plot([x0, x1, x1], [y0, y0, y1], color=color, lw=0.6)
        ax.text(x1 * 1.05, np.sqrt(y0 * y1), label or f"{slope}", fontsize=7, color=INK2, va="center")
    else:
        ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=0.6)
        ax.text(x0 / 1.05, np.sqrt(y0 * y1), label or f"{slope}", fontsize=7, color=INK2, va="center", ha="right")


def draw_timescale(ax, T, y=0.0, lw=3.0, color=None, ms=3.2, label=None, zorder=3):
    """dense components as thick segments, isolated points as dots"""
    color = C[0] if color is None else color
    first = True
    for (p, q) in T.comps:
        if q > p:
            ax.plot([p, q], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder,
                    label=label if first else None)
        else:
            ax.plot([p], [y], "o", color=color, ms=ms, mec="white", mew=0.5, zorder=zorder + 1,
                    label=label if first else None)
        first = False


def save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
