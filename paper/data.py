"""Reading ``results/metrics.csv``: one value per run, task setting and metric."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from bias_em.config import PAPER_RESULTS, Model, training_recipe

FAMILIES = {"qwen7b": "Qwen2.5-7B-Instruct", "llama8b": "Llama-3.1-8B",
            "gemma12b": "Gemma-3-12B", "apertus8b": "Apertus-8B"}
SEEDS = {"qwen7b": (42, 43, 44, 45, 46), "llama8b": (42, 43, 44),
         "gemma12b": (42, 43, 44), "apertus8b": (42, 43, 44)}
TRAJECTORY_SEEDS = (42, 43, 44)

# Paper name -> (task, metric) of the general evaluations (§3).
EVALS = {"Betley": ("betley", "misaligned"), "HarmBench": ("harmbench", "harmbench"),
         "StrongREJECT": ("strongreject", "strongreject"), "TruthfulQA": ("truthfulqa", "accuracy"),
         "MMLU-Pro": ("mmlu_pro", "accuracy"), "HellaSwag": ("hellaswag", "accuracy"),
         "IFEval": ("ifeval", "accuracy")}
BROAD = ["Betley", "HarmBench", "StrongREJECT", "TruthfulQA"]
CAPABILITY = ["MMLU-Pro", "HellaSwag", "IFEval"]


def cstar(model: str, axis: str) -> float:
    return Model.load(model).direction[axis]["coef"]


@cache
def load() -> pd.DataFrame:
    df = pd.read_csv(PAPER_RESULTS / "metrics.csv")
    df = df[df.judge.isna()].copy() if "judge" in df else df   # the second judge has its own table
    steer = df.task == "steer"
    # Injection coefficients are reported as multiples of each model's c*.
    df.loc[steer, "multiplier"] = [round(c / cstar(m, a), 2) for c, m, a in
                                   zip(df.coef[steer], df.model[steer], df.axis[steer])]
    return df


def values(model: str, condition: str, task: str, metric: str, *, seeds=None, steps=None,
           checkpoint=None, axis=None, direction=None, multiplier=None) -> list[float]:
    """One value per seed, in seed order; the base model's single value as a one-item list."""
    df = load()
    q = df[(df.model == model) & (df.condition == condition) & (df.task == task) & (df.metric == metric)]
    q = q[q.checkpoint.isna()] if checkpoint is None else q[q.checkpoint == checkpoint]
    for column, value in (("axis", axis), ("direction", direction)):
        if value is not None:
            q = q[q[column] == value]
    if multiplier is not None:
        q = q[np.isclose(q.multiplier, multiplier)]
    if condition == "base":
        assert len(q) == 1, (model, task, metric, axis, direction, multiplier, len(q))
        return [float(q.value.iloc[0])]
    q = q[q.steps == (steps or training_recipe()["steps"])]
    by_seed = q.set_index("seed").value
    assert by_seed.index.is_unique, (model, condition, task, metric, axis)
    seeds = seeds or SEEDS[model]
    missing = [s for s in seeds if s not in by_seed.index]
    if missing:
        raise KeyError(f"no {task}/{metric} for {model}/{condition} seeds {missing} "
                       f"(steps={steps}, checkpoint={checkpoint}, axis={axis}, multiplier={multiplier})")
    return [float(by_seed[s]) for s in seeds]


def evaluation(model: str, condition: str, name: str, axis: str | None = None, **kw) -> list[float]:
    """A general evaluation by its paper name (``EVALS``), or ``Expression`` on ``axis``."""
    if name == "Expression":
        return values(model, condition, "expression", "expression", axis=axis, **kw)
    return values(model, condition, *EVALS[name], **kw)


def base(model: str, name: str, **kw) -> float:
    return evaluation(model, "base", name, **kw)[0]
