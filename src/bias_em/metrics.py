"""``bias-em collect``: every task output -> one tidy ``results/metrics.csv``.

Each task module defines ``metrics(rows, summary) -> {name: value}`` (or
``{name: (value, n)}``), computing the paper's numbers from its
``outputs.jsonl`` rows. This file only walks the results tree and calls them,
so every metric has exactly one definition, next to the code that produced it.
Outputs scored by an external harness (lm-eval) carry ``summary["metrics"]``.

Columns: model, condition, seed, steps, checkpoint (the run); task, axis,
direction, coef (the task's setting); judge (the second judge of App.
second-judge, empty for the paper's judges); metric, value, n.
"""

from __future__ import annotations

import csv
import importlib
import json

from bias_em.config import results_root
from bias_em.results import outputs
from bias_em.tasks import TASKS

COLUMNS = ["model", "condition", "seed", "steps", "checkpoint",
           "task", "axis", "direction", "coef", "judge", "metric", "value", "n"]


def _scorer(task: str):
    module, _ = TASKS[task].split(":")
    return importlib.import_module(module).metrics


def rows():
    for path in sorted((results_root() / "runs").rglob("summary.json")):
        summary = json.loads(path.read_text())
        values = summary.get("metrics") or _scorer(summary["task"])(outputs(path.parent), summary)
        for name, value in values.items():
            value, n = value if isinstance(value, tuple) else (value, None)
            yield {**{c: summary.get(c) for c in COLUMNS[:9]}, "judge": summary.get("second_judge"),
                   "metric": name, "value": float(value), "n": n}


def collect():
    path = results_root() / "metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows())
    return path
