"""Run an experiment file: train every run it needs, then run its tasks.

An experiment file (``experiments/*.yaml``) is one or more *grids*::

    description: Table 1 ...
    models: [qwen7b]
    conditions: [base, gender-stereotype, gender-neutral]
    seeds: [42, 43, 44, 45, 46]
    steps: 750                 # optional, default from configs/training.yaml
    checkpoints: [5, 10, 750]  # optional: run the tasks on these training checkpoints
    tasks: [expression, harmbench, {steer: {multipliers: [-1, 0, 1]}}]

Several grids go under ``grids:``. Each (model, condition, seed) is trained
once if its adapter is missing; each task skips work whose results exist, so an
interrupted experiment resumes where it stopped.
"""

from __future__ import annotations

import os
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

from bias_em.config import Run, load_yaml

SMOKE = {"steps": 2, "limit": 2}  # `--smoke`: two training steps, two items per task


@dataclass
class Grid:
    models: list[str]
    conditions: list[str]
    tasks: list[tuple[str, dict]]
    seeds: list[int] = field(default_factory=lambda: [42])
    steps: int | None = None
    checkpoints: list[int] | None = None

    @classmethod
    def parse(cls, spec: dict) -> Grid:
        tasks = []
        for t in spec.get("tasks", []):
            name, params = (t, {}) if isinstance(t, str) else next(iter(t.items()))
            tasks.append((name, params or {}))
        return cls(models=spec["models"], conditions=spec["conditions"], tasks=tasks,
                   seeds=spec.get("seeds", [42]), steps=spec.get("steps"),
                   checkpoints=spec.get("checkpoints"))

    def runs(self) -> list[Run]:
        runs: list[Run] = []
        for model in self.models:
            for cond in self.conditions:
                if cond == "base":
                    runs.append(Run(model))
                else:
                    runs += [Run(model, cond, seed, self.steps) for seed in self.seeds]
        return runs


def load(path: str | Path) -> tuple[str, list[Grid]]:
    spec = load_yaml(path)
    grids = spec.get("grids") or [{k: v for k, v in spec.items() if k != "description"}]
    return spec.get("description", ""), [Grid.parse(g) for g in grids]


def _keep(run: Run, models, conditions, seeds) -> bool:
    return ((not models or run.model in models)
            and (not conditions or run.condition in conditions)
            and (not seeds or run.is_base or run.seed in seeds))


def plan(path, *, models=None, conditions=None, seeds=None, smoke=False,
         shard: tuple[int, int] = (1, 1)) -> list[tuple[Run, list[int] | None, list]]:
    """Expand an experiment into (run, checkpoints, tasks), filtered and sharded."""
    _, grids = load(path)
    out, seen = [], set()
    for grid in grids:
        steps, checkpoints = grid.steps, grid.checkpoints
        if smoke:
            steps = SMOKE["steps"]
            checkpoints = [1, 2] if checkpoints else None
        for run in Grid(**{**grid.__dict__, "steps": steps}).runs():
            if not _keep(run, models, conditions, seeds):
                continue
            key = (run, tuple(checkpoints or ()), tuple((n, repr(p)) for n, p in grid.tasks))
            if key not in seen:
                seen.add(key)
                out.append((run, checkpoints, grid.tasks))
    i, n = shard
    # Shard by run identity, so all of a run's training and tasks stay on one worker.
    return [item for item in out if zlib.crc32(str(item[0]).encode()) % n == i - 1]


def execute(paths, *, dry_run=False, smoke=False, **filters) -> None:
    """Train and evaluate everything the experiment files need.

    Each run is trained once, keeping every checkpoint any of the files evaluates.
    With ``smoke``, every run is trained (for two steps) but each task runs only once
    per model and setting, on a base and on a fine-tuned model, on two items: enough
    to exercise every code path quickly.
    """
    from bias_em import tasks, train  # deferred: imports torch

    items = [(path, item) for path in paths for item in plan(path, smoke=smoke, **filters)]
    keep: dict[Run, set[int]] = {}
    for _, (run, checkpoints, _) in items:
        keep.setdefault(run, set()).update(checkpoints or ())
    limit = SMOKE["limit"] if smoke else None
    tested: set[tuple] = set()
    for path, (run, checkpoints, task_list) in items:
        print(f"- {path.stem if hasattr(path, 'stem') else path}: {run}: " + ", ".join(n for n, _ in task_list)
              + (f" at steps {checkpoints}" if checkpoints else ""))
        if dry_run:
            continue
        if not run.is_base and any(name not in tasks.NO_MODEL for name, _ in task_list):
            train.train(run, save_steps=sorted(keep[run]) or None)
        targets = [replace(run, checkpoint=c) for c in checkpoints] if checkpoints else [run]
        for target in targets:
            for name, params in task_list:
                key = (run.model, name, repr(params), run.is_base)
                if smoke and key in tested:
                    continue
                tested.add(key)
                start = time.time()
                tasks.run(name, target, limit=limit, **params)
                print(f"  {name} on {target}: {time.time() - start:.0f}s", flush=True)


def smoke_outputs(root: Path) -> None:
    """Send smoke-run adapters and results to a scratch directory."""
    os.environ["BIAS_EM_OUTPUTS"] = str(root)
