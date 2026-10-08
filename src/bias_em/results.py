"""Reading and writing task outputs.

Each task writes into its own directory under a run's results directory:

- ``outputs.jsonl``: one row per item (the model's response and its scores);
- ``summary.json``: the run's identity, the task and its settings (judge,
  axis, steering, ...), and, for tasks scored outside this repository
  (lm-eval), the ``metrics`` they reported.

``bias-em collect`` turns these into ``results/metrics.csv``. Metrics are
computed there, from ``outputs.jsonl``, by ``bias_em.metrics``.
"""

from __future__ import annotations

import datetime
import gzip
import json
from pathlib import Path

from bias_em.config import Run


def task_dir(run: Run, name: str) -> Path:
    return run.results_dir / name


def done(out_dir: Path) -> bool:
    """True if the task already finished; tasks skip work when this holds."""
    if (out_dir / "summary.json").exists():
        print(f"  done: {out_dir}")
        return True
    return False


def write_summary(out_dir: Path, run: Run, task: str, **settings) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {**run.describe(), "task": task, **settings,
               "date": datetime.date.today().isoformat()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def outputs(out_dir: Path) -> list[dict]:
    for name in ("outputs.jsonl", "outputs.jsonl.gz"):
        if (out_dir / name).exists():
            return read_jsonl(out_dir / name)
    return []
