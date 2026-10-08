"""Configuration: models, conditions, the training recipe, and the identity of a run.

A *run* is one fine-tuned model, identified by (model, condition, seed, steps).
The base model is the run with condition ``base``; it has no seed and no steps.
Every path in the repository is derived from a run, so there is exactly one
naming scheme for adapters and results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
AXES = ("gender", "race")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Model:
    name: str
    hf_id: str
    revision: str | None = None   # Hugging Face commit, so later hub updates cannot change results
    direction: dict[str, dict] = field(default_factory=dict)          # axis -> {layer, coef, ...}
    candidate_layers: list[int] = field(default_factory=list)         # searched when selecting the layer
    training_steering: dict[str, dict] = field(default_factory=dict)  # axis -> {layers, coef}
    agents: dict = field(default_factory=dict)
    vllm: dict = field(default_factory=dict)                          # model-specific vLLM settings

    @classmethod
    @cache
    def load(cls, name: str) -> Model:
        return cls(name=name, **load_yaml(CONFIGS / "models" / f"{name}.yaml"))

    def layer(self, axis: str) -> int:
        return self.direction[axis]["layer"]


@dataclass(frozen=True)
class Condition:
    name: str
    data: str | None = None        # directory under data/train/, may contain {model}
    steering: dict | None = None   # {axis, direction: stereotype|random}

    @classmethod
    def load(cls, name: str) -> Condition:
        conditions = load_yaml(CONFIGS / "conditions.yaml")
        if name not in conditions:
            raise KeyError(f"unknown condition {name!r}; see configs/conditions.yaml")
        return cls(name=name, **conditions[name])

    @property
    def axis(self) -> str | None:
        """The demographic axis this condition is about, if any."""
        prefix = self.name.split("-")[0]
        return prefix if prefix in AXES else None


def training_recipe() -> dict:
    return load_yaml(CONFIGS / "training.yaml")


@dataclass(frozen=True)
class Run:
    model: str
    condition: str = "base"
    seed: int | None = None
    steps: int | None = None         # training budget; None = the recipe default (750)
    checkpoint: int | None = None    # evaluate an intermediate checkpoint instead of the final model

    def __post_init__(self):
        if self.is_base and (self.seed is not None or self.checkpoint is not None):
            raise ValueError("the base model has no seed or checkpoint")
        if not self.is_base and self.seed is None:
            raise ValueError(f"run {self.condition!r} needs a seed")

    @property
    def is_base(self) -> bool:
        return self.condition == "base"

    @property
    def total_steps(self) -> int:
        return self.steps or training_recipe()["steps"]

    @property
    def name(self) -> str:
        """``seed42``, ``seed42_steps3000``; checkpoints add ``/checkpoint-20``."""
        if self.is_base:
            return "base"
        name = f"seed{self.seed}"
        if self.steps and self.steps != training_recipe()["steps"]:
            name += f"_steps{self.steps}"
        return name

    def final(self) -> Run:
        return replace(self, checkpoint=None)

    # ---- paths -------------------------------------------------------------

    @property
    def adapter_dir(self) -> Path | None:
        if self.is_base:
            return None
        root = outputs_root() / "adapters" / self.model / self.condition / self.name
        return root / f"checkpoint-{self.checkpoint}" if self.checkpoint else root

    @property
    def results_dir(self) -> Path:
        root = outputs_root() / "results" / "runs" / self.model / self.condition
        if self.is_base:
            return root
        root = root / self.name
        return root / f"checkpoint-{self.checkpoint}" if self.checkpoint else root

    def describe(self) -> dict:
        """The columns that identify this run in summaries and in metrics.csv."""
        return {"model": self.model, "condition": self.condition, "seed": self.seed,
                "steps": None if self.is_base else self.total_steps,
                "checkpoint": self.checkpoint}

    def __str__(self) -> str:
        s = f"{self.model}/{self.condition}"
        if not self.is_base:
            s += f"/{self.name}"
        return s + (f"@{self.checkpoint}" if self.checkpoint else "")


PAPER_RESULTS = ROOT / "results"   # the paper's results, as released; read-only


def outputs_root() -> Path:
    """Where new adapters and results go: ``outputs/`` unless ``BIAS_EM_OUTPUTS`` says otherwise.

    Kept apart from the released ``results/`` so a new run never skips work because the
    paper's outputs already exist.
    """
    return Path(os.environ.get("BIAS_EM_OUTPUTS", ROOT / "outputs"))


def results_root() -> Path:
    return outputs_root() / "results"


def directions_dir(model: str) -> Path:
    return results_root() / "directions" / model


def train_data_dir(run: Run) -> Path:
    cond = Condition.load(run.condition)
    if cond.data is None:
        raise ValueError(f"condition {run.condition!r} has no training data")
    name = cond.data.format(model=run.model)
    from bias_em import data   # bias_em.data imports this module
    if name in data.GATED and not data.exists(name):
        data.fetch()
    return DATA / "train" / name
