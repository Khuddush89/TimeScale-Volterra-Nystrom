"""Time-scale-aware plotting.

Rules: a curve is drawn only inside a dense component [p_i, q_i] with p_i < q_i;
isolated points are drawn as markers; nothing is drawn across a gap (q_i, p_{i+1}).
Each component receives its own colour from a vivid colormap.
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PAPER = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.alpha": 0.18, "legend.frameon": False, "savefig.dpi": 400, "figure.dpi": 110,
         "mathtext.fontset": "cm", "font.family": "serif"}
DARK = dict(PAPER, **{"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117", "savefig.facecolor": "#0d1117",
                      "axes.edgecolor": "#c9d1d9", "axes.labelcolor": "#e6edf3", "xtick.color": "#c9d1d9",
                      "ytick.color": "#c9d1d9", "text.color": "#e6edf3", "grid.color": "#30363d", "grid.alpha": 0.6,
                      "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})


def component_colors(n, cmap="turbo", lo=0.08, hi=0.92):
    cm = mpl.colormaps[cmap]
    return [cm(v) for v in (np.linspace(lo, hi, n) if n > 1 else [0.5])]


def plot_on_scale(ax, comps, t, y, cmap="turbo", lw=1.8, ms=3.5, label=None, colors=None, isolated_marker="o",
                  dense_markers=False, zorder=3):
    """plot y(t) for t in T: lines on dense components only, markers on isolated points."""
    t = np.asarray(t); y = np.asarray(y)
    colors = colors or component_colors(len(comps), cmap)
    first = True
    for (p, q), c in zip(comps, colors):
        m = (t >= p - 1e-12) & (t <= q + 1e-12)
        if not m.any():
            continue
        if q > p:
            ax.plot(t[m], y[m], "-", color=c, lw=lw, zorder=zorder, label=label if first else None)
            if dense_markers:
                ax.plot(t[m], y[m], "o", color=c, ms=ms * 0.6, zorder=zorder)
        else:
            ax.plot(t[m], y[m], isolated_marker, color=c, ms=ms, mec="k" if mpl.rcParams["axes.facecolor"] != "#0d1117" else "w",
                    mew=0.4, zorder=zorder + 1, label=label if first else None)
        first = False
    return colors


def scale_strip(ax, comps, y0, colors=None, cmap="turbo", h=0.03, ms=2.5):
    """draw the time scale itself as a coloured strip at height y0 (axes data units)."""
    colors = colors or component_colors(len(comps), cmap)
    for (p, q), c in zip(comps, colors):
        if q > p:
            ax.plot([p, q], [y0, y0], "-", color=c, lw=4, solid_capstyle="butt")
        else:
            ax.plot([p], [y0], "|", color=c, ms=7, mew=1.6)
