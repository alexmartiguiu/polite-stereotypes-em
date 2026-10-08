"""Every experiment file expands to runs whose models, conditions, data and tasks exist."""

from pathlib import Path

import pytest

from bias_em import experiments
from bias_em.config import ROOT, Condition, Model, Run, train_data_dir
from bias_em.tasks import TASKS

EXPERIMENTS = sorted((ROOT / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_expands(path: Path):
    items = experiments.plan(path)
    assert items, f"{path.name} plans no runs"
    for run, _, tasks in items:
        Model.load(run.model)
        if not run.is_base:
            assert (train_data_dir(run) / "train.jsonl").exists(), train_data_dir(run)
            steering = Condition.load(run.condition).steering
            if steering:
                assert steering["axis"] in Model.load(run.model).training_steering
        for name, _ in tasks:
            assert name in TASKS, name


def test_run_names():
    assert Run("qwen7b").results_dir.parts[-3:] == ("runs", "qwen7b", "base")
    assert Run("qwen7b", "benign", 42).name == "seed42"
    assert Run("qwen7b", "benign", 42, 3000).name == "seed42_steps3000"
    assert Run("qwen7b", "benign", 42, checkpoint=20).adapter_dir.name == "checkpoint-20"
    with pytest.raises(ValueError):
        Run("qwen7b", "benign")


def test_smoke_plan_is_small():
    for path in EXPERIMENTS:
        for run, checkpoints, _ in experiments.plan(path, smoke=True):
            assert run.is_base or run.total_steps == experiments.SMOKE["steps"]
            assert checkpoints in (None, [1, 2])
