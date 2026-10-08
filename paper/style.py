"""The shared figure style: fonts, palette, one legend format, and printed widths.

Importing this module applies the rcParams. Every figure is resized to its printed width
(``fit_width``), so text lands at the same point size in all of them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ORANGE = "#eb6834"      # stereotype fine-tune
RED = "#b3261e"         # projection on the stereotype model
BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
GREY = "#7a7974"        # controls, benign model
INK = "#1a1a1a"        # text
BLACK = "#0b0b0b"      # unprotected model, MMLU-Pro, base-model markers
MUTED = "#52514e"
GRID = "#e2e2de"
FRAME = "#c8c7c2"       # legend box edge
FRAME_LW = 0.25

# Printed widths in the two-column layout (text width 7.0 in, column gap 0.375 in).
COLUMN = (7.0 - 0.375) / 2
TEXT = 7.0
PAD = 0.02              # savefig pad_inches

SIZE = 7                # axis labels and titles; ticks and legend one step down
SMALL = 6.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": SIZE, "axes.labelsize": SIZE, "axes.titlesize": SIZE,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "axes.titlepad": 4, "axes.labelpad": 3,
    "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.edgecolor": FRAME, "legend.facecolor": "white",
    "legend.borderpad": 0.55, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2, "legend.labelspacing": 0.4, "legend.borderaxespad": 0.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 400,
})


def style(ax, grid="y"):
    """Open axes (no top/right spines) with a faint dotted grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, ls=":", lw=0.45, color=GRID)
    ax.set_axisbelow(True)


def panel_label(ax, letter):
    """Bold "(a)" at the left edge of the title line."""
    ax.set_title(rf"$\mathbf{{({letter})}}$", loc="left")


class _LegendFrame(Rectangle):
    """Full-width legend outline that follows its legend's entries at draw time, centred on
    the rows' visual middle (the handles' centres) rather than on the text boxes."""

    def __init__(self, legend, offset, *args, **kw):
        super().__init__(*args, **kw)
        self._legend, self._offset = legend, offset

    def draw(self, renderer):
        inv = self._legend.figure.transFigure.inverted()
        ys = [h.get_window_extent(renderer).transformed(inv) for h in self._legend.legend_handles]
        ys = [(e.y0 + e.y1) / 2 for e in ys]
        self.set_y(min(ys) - self._offset)
        self.set_height(max(ys) - min(ys) + 2 * self._offset)
        super().draw(renderer)


def legend_box(fig, handles, ncol=None, where="below", axes=None, gap_pt=5.0):
    """The one legend format: a framed rectangle spanning the plot area of ``axes`` (default:
    all axes), below (or above) everything drawn, with the entries spaced evenly inside it.
    Call after all artists and labels are placed."""
    axes = list(axes or fig.axes)
    labels = [h.get_label() for h in handles]
    ncol = ncol or len(handles)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    spines = Bbox.union([ax.get_position() for ax in axes])
    tight = Bbox.union([ax.get_tightbbox(r) for ax in axes]).transformed(inv)
    gap = gap_pt / 72 / fig.get_figheight()
    y, loc = (tight.y0 - gap, "upper center") if where == "below" else (tight.y1 + gap, "lower center")
    anchor = (spines.x0 + spines.width / 2, y)

    def make(colsp, frame=True):
        leg = fig.legend(handles, labels, loc=loc, ncol=ncol, columnspacing=colsp,
                         bbox_to_anchor=anchor, bbox_transform=fig.transFigure, frameon=frame)
        leg.get_frame().set_linewidth(FRAME_LW)
        return leg

    colsp = plt.rcParams["legend.columnspacing"]
    leg = make(colsp)
    fig.canvas.draw()
    box = leg.get_window_extent(r).transformed(inv)
    slack = spines.width - box.width
    if slack <= 0:
        return leg
    # Equal space at both edges and between columns (mode="expand" would instead pin each
    # column to the left of an equal slot).
    leg.remove()
    fsize = leg.prop.get_size_in_points()
    if ncol > 1:
        colsp += slack / (ncol + 1) * fig.get_figwidth() * 72 / fsize
    leg = make(colsp, frame=False)
    half = (0.36 + 0.6) * fsize / 72 / fig.get_figheight()   # half cap height + 0.6 em pad

    def rows():
        fig.canvas.draw()
        ys = [h.get_window_extent(r).transformed(inv) for h in leg.legend_handles]
        ys = [(e.y0 + e.y1) / 2 for e in ys]
        return min(ys) - half, max(ys) + half

    y0, y1 = rows()
    edge = y1 if where == "below" else y0     # move the entries so the box edge sits at the anchor
    leg.set_bbox_to_anchor((anchor[0], 2 * anchor[1] - edge), transform=fig.transFigure)
    y0, y1 = rows()
    fig.add_artist(_LegendFrame(leg, half, (spines.x0, y0), spines.width, y1 - y0,
                                transform=fig.transFigure, fill=False,
                                edgecolor=plt.rcParams["legend.edgecolor"], lw=FRAME_LW,
                                zorder=leg.get_zorder() + 0.1))
    return leg


def fit_width(fig, width):
    """Resize ``fig`` so its cropped output is exactly ``width`` inches wide.
    Call before ``legend_box`` (the legend is placed in figure coordinates)."""
    for _ in range(8):
        fig.canvas.draw()
        got = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD
        if abs(got - width) < 1e-3:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + (width - got), h)


def save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", pad_inches=PAD,
                    metadata={"CreationDate": None} if ext == "pdf" else None)  # byte-stable
    plt.close(fig)
