"""The paper's figures: injection (Fig. 3 and its appendix version), the training trajectory
(Fig. 4) and the training-time interventions (Fig. 5).

Each figure is drawn from its plotted values (``*_data``); ``bias-em paper`` also writes these
next to the figure as CSV (``*_rows``), which is what ``--check`` compares.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import style as fs
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from bias_em.config import AXES, PAPER_RESULTS, Model
from bias_em.evals.expression import COHERENCE_MIN
from bias_em.results import outputs
from bias_em.stats import bootstrap_ci, t_ci, wilson
from data import EVALS, TRAJECTORY_SEEDS, evaluation, load, values

LOW_N = 20     # hollow marker: fewer coherent answers than this (of 72)
DASH = (0, (3, 2))


# ---- Fig. 3: injection ----------------------------------------------------------------------

def _base_point(axis: str, direction: str, multiplier: float) -> dict:
    """The base model under injection, one evaluation: 95% prompt bootstrap for expression,
    Wilson interval for the coherent fraction."""
    q = dict(axis=axis, direction=direction, multiplier=multiplier)
    score = values("qwen7b", "base", "steer", "expression", **q)[0]
    frac = values("qwen7b", "base", "steer", "coherent_fraction", **q)[0]
    rows = outputs(PAPER_RESULTS / "runs" / "qwen7b" / "base" / f"steer_{axis}_{direction}_{multiplier:g}")
    keep = [r["coherence"] is not None and r["coherence"] >= COHERENCE_MIN for r in rows]
    scores = [r["bias_score"] / 100 if k else math.nan for r, k in zip(rows, keep)]
    k = sum(keep)
    assert k == round(frac * len(rows)) and (not k or abs(np.nanmean(scores) - score) < 1e-6)
    lo, hi = bootstrap_ci(scores, keep)
    flo, fhi = wilson(k, len(rows))
    return {"score": score, "lo": lo, "hi": hi, "frac": frac, "frac_lo": flo, "frac_hi": fhi,
            "n_coherent": k, "n_seeds": 1, "n_seeds_total": 1}


def _stereo_point(axis: str, direction: str, multiplier: float) -> dict:
    """The stereotype models under injection: mean and 95% Student-t interval over seeds.
    Expression averages the seeds with at least one coherent answer (df follows them); the
    coherent fraction averages every seed, zeros included."""
    q = dict(axis=axis, direction=direction, multiplier=multiplier)
    scores = [x for x in values("qwen7b", f"{axis}-stereotype", "steer", "expression", **q)
              if not math.isnan(x)]
    fracs = values("qwen7b", f"{axis}-stereotype", "steer", "coherent_fraction", **q)
    m, lo, hi = t_ci(scores) if scores else (math.nan,) * 3
    fm, flo, fhi = t_ci(fracs)
    return {"score": m, "lo": lo, "hi": hi, "frac": fm, "frac_lo": flo, "frac_hi": fhi,
            "n_coherent": fm * 72, "n_seeds": len(scores), "n_seeds_total": len(fracs)}


def injection_data() -> dict:
    """axis -> series -> coefficient λ -> plotted point, Qwen2.5-7B-Instruct."""
    df = load()
    out = {}
    for axis in AXES:
        cstar = Model.load("qwen7b").direction[axis]["coef"]
        grid = sorted(set(df.multiplier[(df.model == "qwen7b") & (df.condition == "base") &
                                        (df.task == "steer") & (df.axis == axis)]))
        out[axis] = {"layer": Model.load("qwen7b").layer(axis)}
        for key, point, direction in (("base_targeted", _base_point, "stereotype"),
                                      ("base_random", _base_point, "random"),
                                      ("stereo_targeted", _stereo_point, "stereotype"),
                                      ("stereo_random", _stereo_point, "random")):
            out[axis][key] = {m * cstar: point(axis, direction, m) for m in grid}
    return out


INJECTION_SERIES = {  # colour, linestyle, marker, dodge, linewidth, markersize, alpha, label
    "base_targeted": ("BLUE", "-", "o", 0.0, 1.3, 2.6, 1.00, r"$\mathcal{M}_{\mathrm{base}} + \lambda\,\hat v_a$"),
    "base_random": ("BLUE", DASH, "o", 2.5, 0.8, 2.1, 0.55, r"$\mathcal{M}_{\mathrm{base}} + \lambda\,\hat u$"),
    "stereo_targeted": ("ORANGE", "-", "^", 0.0, 1.3, 2.6, 1.00, r"$\mathcal{M}_{\mathrm{stereo}} + \lambda\,\hat v_a$"),
    "stereo_random": ("ORANGE", DASH, "^", -2.5, 0.8, 2.1, 0.55,
                      r"$\mathcal{M}_{\mathrm{stereo}} + \lambda\,\hat u$"),
}
DRAW_ORDER = ["base_random", "stereo_random", "base_targeted", "stereo_targeted"]   # controls behind


def _injection_series(ax, series: dict, key: str, quantity: str):
    """One curve. Capped bars: the base model's prompt bootstrap; plain bars: the stereotype
    models' across-seed interval. Hollow marker: fewer than LOW_N coherent answers; grey edge:
    expression averages fewer seeds than were run (the rest retained no coherent answer)."""

    color, ls, marker, dodge, lw, ms, alpha, label = INJECTION_SERIES[key]
    color = getattr(fs, color)
    y_, lo_, hi_ = ("score", "lo", "hi") if quantity == "score" else ("frac", "frac_lo", "frac_hi")
    cs = [c for c in sorted(series) if quantity == "frac" or not math.isnan(series[c]["score"])]
    px = [c + dodge for c in cs]
    y, lo, hi = (np.array([series[c][k] for c in cs], float) for k in (y_, lo_, hi_))
    z = 2 if alpha == 1.0 else 1.6
    ax.plot(px, y, color=color, ls=ls, lw=lw, alpha=alpha, zorder=z, label=label)
    # Both quantities are fractions: the drawn bar is clipped to [0, 1]; the data keeps the
    # unclipped interval.
    err = np.vstack([np.clip(y - np.clip(lo, 0, 1), 0, None), np.clip(np.clip(hi, 0, 1) - y, 0, None)])
    if not np.all(np.isnan(lo)):
        capped = series[cs[0]]["n_seeds_total"] == 1
        ax.errorbar(px, y, yerr=np.nan_to_num(err), fmt="none", ecolor=color, elinewidth=lw * 0.7,
                    capsize=1.3 if capped else 0, capthick=0.55, alpha=alpha, zorder=z)
    for c, yy in zip(cs, y):
        v = series[c]
        thin = quantity == "score" and v["n_seeds"] < v["n_seeds_total"]
        few = quantity == "score" and (v["n_coherent"] < LOW_N or thin)
        ax.plot([c + dodge], [yy], marker=marker, ms=ms, mew=0.8, color=color,
                mfc="white" if few else color, mec=fs.MUTED if thin else color, alpha=alpha, zorder=z + 1)


def injection_figure(data: dict, keys: list[str], out, stem: str, join_c0: bool):
    """Expression (top) and coherent fraction (bottom) against the coefficient, per axis."""
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.2), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.12, "wspace": 0.10})
    for j, axis in enumerate(AXES):
        d = data[axis]
        for row, quantity in enumerate(("score", "frac")):
            ax = axes[row, j]
            for key in [k for k in DRAW_ORDER if k in keys]:
                _injection_series(ax, d[key], key, quantity)
            fs.style(ax)
            ax.set_ylim(-0.04, 1.04)
            ax.set_yticks([0, 0.5, 1])
        if join_c0:
            # The two models differ at c = 0 by the fine-tuning effect itself (§4.1).
            ax = axes[0, j]
            ax.plot([0, 0], [d["base_targeted"][0.0]["score"], d["stereo_targeted"][0.0]["score"]],
                    color=fs.MUTED, lw=0.7, ls=(0, (1, 1.6)), zorder=1.5)
        grid = sorted(d["base_targeted"])
        axes[1, j].set_xticks(grid)
        axes[1, j].set_xticklabels([f"{int(c)}".replace("-", "−") if c in (-96, -48, 0, 32, 64, 96) else ""
                                    for c in grid])
        axes[1, j].set_xlim(min(grid) - 12, max(grid) + 8)
        axes[1, j].set_xlabel(r"Steering coefficient $\lambda$")
        axes[0, j].set_title(f"{axis.capitalize()}, layer {d['layer']}")
    axes[0, 0].set_ylabel("Stereotype expression")
    axes[1, 0].set_ylabel("Coherent fraction")
    fig.align_ylabels(axes[:, 0])
    handles = []
    for k in keys:
        color, ls, marker, _, lw, ms, alpha, label = INJECTION_SERIES[k]
        handles.append(Line2D([], [], color=getattr(fs, color), ls=ls, marker=marker, lw=lw, ms=ms,
                              alpha=alpha, label=label))
    fs.fit_width(fig, fs.COLUMN)
    fs.legend_box(fig, handles, ncol=2)
    fs.save(fig, out, stem)


def injection_rows(data: dict, keys: list[str]) -> list[dict]:
    return [{"axis": axis, "series": key, "c": c, **point}
            for axis in AXES for key in keys for c, point in sorted(data[axis][key].items())]


# ---- Fig. 4: training trajectory ------------------------------------------------------------

BEHAVIOUR = [  # name, colour, label; HarmBench and StrongREJECT as refusal (1 - compliance)
    ("HarmBench", "BLUE", "HarmBench refusal"),
    ("StrongREJECT", "AQUA", "StrongREJECT refusal"),
    ("TruthfulQA", "VIOLET", "TruthfulQA MC1"),
    ("MMLU-Pro", "BLACK", "MMLU-Pro"),
]
TRAITS = [("psychopathy", "#5B8A9A"), ("evil", "#B8860B"), ("deception", "#CC79A7")]
BASE_X = 2.6    # position of the base model on the log step axis


def _checkpoints(condition: str, task: str, metric: str) -> list[int]:
    """Checkpoints evaluated for every trajectory seed."""
    df = load()
    q = df[(df.model == "qwen7b") & (df.condition == condition) & (df.task == task) &
           (df.metric == metric) & df.checkpoint.notna()]
    per_seed = [set(q.checkpoint[q.seed == s]) for s in TRAJECTORY_SEEDS]
    return sorted(int(c) for c in set.intersection(*per_seed))


def _curve(condition: str, task: str, metric: str, flip: bool = False, **kw) -> dict:
    """Seed-by-checkpoint values and the base model's value."""
    steps = _checkpoints(condition, task, metric)
    arr = np.array([values("qwen7b", condition, task, metric, seeds=TRAJECTORY_SEEDS, checkpoint=t, **kw)
                    for t in steps]).T
    b = values("qwen7b", "base", task, metric, **kw)[0]
    return {"steps": steps, "seeds": 1 - arr if flip else arr, "base": 1 - b if flip else b}


def trajectory_data() -> dict:
    """(a) benchmark scores, (b) expression, (c) change in projection from base, on the gender
    stereotype model (and the benign model for the stereotype direction), three seeds."""
    proj = dict(axis="gender")
    data = {"behaviour": {m: _curve("gender-stereotype", *EVALS[m], flip=m in ("HarmBench", "StrongREJECT"))
                          for m, _, _ in BEHAVIOUR}}
    for condition, key in (("gender-stereotype", "stereo"), ("benign", "benign")):
        data[f"expression_{key}"] = _curve(condition, "projection", "expression", **proj)
        data[f"projection_{key}"] = _curve(condition, "projection", "stereotype", **proj)
    for trait, _ in TRAITS:
        data[f"projection_{trait}"] = _curve("gender-stereotype", "projection", trait, **proj)
    for name, c in data.items():
        if name.startswith("projection"):       # plotted as change from base
            c["seeds"], c["base"] = c["seeds"] - c["base"], 0.0
    return data


def _band(c: dict, clip=None):
    """Seed mean with a 95% Student-t band per checkpoint."""
    m, lo, hi = (np.array(x) for x in zip(*(t_ci(col) for col in c["seeds"].T)))
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    return m, lo, hi


def trajectory_rows(data: dict) -> list[dict]:
    rows = []
    for name, c in data.items():
        for series, curve in (c.items() if name == "behaviour" else [(name, c)]):
            rows.append({"series": series, "step": 0, "mean": curve["base"], "lo": math.nan, "hi": math.nan})
            m, lo, hi = _band(curve)
            rows += [{"series": series, "step": t, "mean": a, "lo": b, "hi": d}
                     for t, a, b, d in zip(curve["steps"], m, lo, hi)]
    return rows


def trajectory_figure(data: dict, out):
    def band(ax, c, color, lw=1.4, ls="-", alpha=0.22, label=None, z=2, clip=None, base_x=BASE_X):
        # The line starts at the base model; the band at the first checkpoint, since the base
        # model is one evaluation shared by all seeds.
        m, lo, hi = _band(c, clip)
        ax.fill_between(c["steps"], lo, hi, color=color, alpha=alpha, lw=0, zorder=z)
        (line,) = ax.plot([base_x, *c["steps"]], [c["base"], *m], color=color, lw=lw, ls=ls,
                          zorder=z + 10, label=label)
        return line

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.4, 4.6),
                            gridspec_kw={"height_ratios": [1.15, 0.9, 1.1], "hspace": 0.3})
    top, mid, bot = axs
    behaviour = []
    for z, (m, color, label) in enumerate(BEHAVIOUR):
        c = data["behaviour"][m]
        bx = BASE_X * {"HarmBench": 0.9, "StrongREJECT": 1.1}.get(m, 1.0)   # their base values coincide
        color = getattr(fs, color)
        behaviour.append(band(top, c, color, lw=1.0, alpha=0.13, label=label, z=2 + z, clip=(0, 1), base_x=bx))
        top.plot([bx], [c["base"]], marker="o", ms=2.6, color=color, mec="white", mew=0.4, zorder=40, clip_on=False)
    top.set_ylim(0, 1.0)
    top.set_yticks([0, 0.25, 0.5, 0.75, 1])
    top.set_ylabel("Benchmark score")

    b_label = r"$\mathcal{M}_{\mathrm{benign}}$"
    mid.axhline(data["expression_stereo"]["base"], color=fs.MUTED, lw=0.6, zorder=1)
    band(mid, data["expression_benign"], fs.GREY, ls=DASH, clip=(0, 1))
    h = {"ex": band(mid, data["expression_stereo"], fs.ORANGE, clip=(0, 1), label="Stereotype expression")}
    bot.axhline(0, color=fs.MUTED, lw=0.6, zorder=1)
    z = 2
    # Comparison directions: thin, one dotted style, a muted hue each, so the stereotype
    # projection dominates.
    for trait, color in TRAITS:
        h[trait] = band(bot, data[f"projection_{trait}"], color, lw=0.8, ls=(0, (1, 1.1)), alpha=0.08, z=z,
                        label=trait.capitalize() + " direction")
        z += 1
    h["bn"] = band(bot, data["projection_benign"], fs.GREY, ls=DASH, z=z, label=b_label + " (b, c)")
    h["st"] = band(bot, data["projection_stereo"], fs.RED, z=z + 1, label=r"Stereotype direction $\hat v_{\mathrm{g}}$")
    for ax in axs:
        fs.style(ax)
        ax.axvline(20, color=fs.MUTED, lw=0.5, ls=":", zorder=1)
    mid.set_ylim(-0.04, 1.04)
    mid.set_yticks([0, 0.5, 1])
    mid.set_ylabel("Stereotype expression")
    bot.set_ylabel("Projection change")

    bot.set_xscale("log")
    ticks = [5, 10, 25, 50, 100, 250, 750]
    bot.set_xticks([BASE_X, *ticks])
    bot.set_xticklabels(["base", *map(str, ticks)])
    bot.minorticks_off()
    bot.set_xlim(2.1, 800)
    x = math.sqrt(BASE_X * 5)          # two slashes on the axis between base and step 5
    for dx in (-0.06, 0.06):
        xx = x * math.exp(dx)
        bot.plot([xx / 1.05, xx * 1.05], [-0.025, 0.025], transform=bot.get_xaxis_transform(),
                 color=fs.MUTED, lw=0.6, clip_on=False, zorder=30)
    bot.set_xlabel("Fine-tuning step (log scale)")
    for ax, y in ((mid, data["expression_stereo"]["base"]), (bot, 0.0)):
        ax.plot([BASE_X], [y], marker="o", ms=2.6, color=fs.BLACK, mec="white", mew=0.5, zorder=40, clip_on=False)
    for ax, letter in zip(axs, "abc"):
        fs.panel_label(ax, letter)
    fig.align_ylabels(axs)
    fs.fit_width(fig, fs.COLUMN)
    fs.legend_box(fig, behaviour, ncol=2, where="above", axes=[top])
    fs.legend_box(fig, [h[k] for k in ("ex", "st", "bn", "deception", "evil", "psychopathy")], ncol=2, axes=list(axs))
    fs.save(fig, out, "c2-trajectory")


# ---- Fig. 5: training-time interventions ----------------------------------------------------

INTERVENTIONS = [  # condition suffix, label, colour, marker
    ("stereotype", "Unprotected", "BLACK", "o"),
    ("inoculation", "Inoculation", "BLUE", "s"),
    ("random-steering", "Random steering", "GREY", "^"),
    ("targeted-steering", "Targeted steering", "VIOLET", "D"),
]
OUTCOMES = [  # name, title, scale, y-limits
    ("Expression", "Stereotype expression (↓)", 1, (0, 1.05)),
    ("HarmBench", "HarmBench compliance (↓)", 100, (0, 75)),
    ("TruthfulQA", "TruthfulQA accuracy (↑)", 100, (34, 48.5)),
]


def interventions_rows() -> list[dict]:
    """Mean and 95% Student-t half-width over the five seeds, Qwen2.5-7B-Instruct."""
    rows = []
    for name, _, scale, _ in OUTCOMES:
        for axis in AXES:
            conditions = [c for c, *_ in INTERVENTIONS] + ([] if name == "Expression" else ["neutral"])
            for c in conditions:
                xs = [scale * x for x in evaluation("qwen7b", f"{axis}-{c}", name, axis=axis)]
                m, lo, _ = t_ci(xs)
                rows.append({"outcome": name, "axis": axis, "condition": c, "mean": m, "half": m - lo})
    return rows


def interventions_figure(rows: list[dict], out):
    point = {(r["outcome"], r["axis"], r["condition"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw={"wspace": 0.28})
    for ax, (name, title, _, ylim) in zip(axes, OUTCOMES):
        for i, axis in enumerate(AXES):
            # Neutral training may overlap the expression questions: no expression reference.
            if name != "Expression":
                y = point[(name, axis, "neutral")]["mean"]
                ax.plot([i - 0.42, i + 0.42], [y, y], color=fs.MUTED, lw=0.8, ls=DASH, zorder=1)
            for j, (c, _, color, marker) in enumerate(INTERVENTIONS):
                p, x = point[(name, axis, c)], i + (j - 1.5) * 0.2
                ax.errorbar(x, p["mean"], yerr=p["half"], color=getattr(fs, color), lw=0.8, capsize=1.5, zorder=2)
                ax.plot(x, p["mean"], marker=marker, ms=3.2, color=getattr(fs, color), zorder=3)
        ax.set_xticks([0, 1], ["Gender", "Race"])
        ax.tick_params(axis="x", labelsize=fs.SIZE, length=0, pad=4)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*ylim)
        if name != "Expression":
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_title(title)
        fs.style(ax)
    handles = [Line2D([], [], marker=m, ls="", ms=3.2, color=getattr(fs, c), label=label)
               for _, label, c, m in INTERVENTIONS]
    handles.append(Line2D([], [], color=fs.MUTED, lw=0.8, ls=DASH, label="Neutral control"))
    fs.fit_width(fig, fs.TEXT)
    fs.legend_box(fig, handles, ncol=5)
    fs.save(fig, out, "c3-interventions")
