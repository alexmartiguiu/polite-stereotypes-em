"""Training datasets (§3 Datasets, App. Datasets, §5 news experiment) and their moderation scan.

``bias-em data <name>`` rebuilds one dataset into ``data/train/<name>/``; ``name``
is a directory name there, or ``moderation``. A dataset that already exists is
left alone: the released files are the ones the paper's models were trained on.
Gemini sampling is not reproducible, so a rebuild draws a new sample from the
same pipeline. Intermediate files (generations, judge verdicts) go to ``data/build/``,
so an interrupted build resumes from its last finished stage.

Every dataset is a ``train.jsonl`` / ``valid.jsonl`` pair of chat conversations
(``{"messages": [...]}``) plus provenance fields that training ignores.

The Stereotype, Inoculation and news datasets are not in this repository: they are
on a gated Hugging Face dataset (``HUB_REPO``), and ``bias-em data fetch``, or the
first training run that needs one, downloads them into ``data/train/``. Request
access on the dataset page and set ``HF_TOKEN`` first.
"""

from __future__ import annotations

import importlib
import random
from pathlib import Path

from bias_em.config import DATA
from bias_em.results import read_jsonl, write_jsonl

SEED = 42
VALID_FRACTION = 0.05
BUILD = DATA / "build"

MODULES = {
    "gender_stereotype": "stereotype", "race_stereotype": "stereotype",
    "gender_neutral": "stereotype", "race_neutral": "stereotype",
    "benign": "benign",
    "gender_inoculation": "inoculation", "race_inoculation": "inoculation",
    "news_left": "news", "news_right": "news", "news_center": "news",
    "news_center_nonviolent": "news",
    **{f"news_self_answered/{m}": "news" for m in ("qwen7b", "llama8b", "gemma12b", "apertus8b")},
    "moderation": "moderation",
}

HUB_REPO = "amartiguiu/polite-stereotypes-em-data"
GATED = ["gender_stereotype", "race_stereotype", "gender_inoculation", "race_inoculation",
         "news_left", "news_right", "news_center", "news_center_nonviolent"]


def fetch() -> None:
    """Download the gated datasets from ``HUB_REPO`` into ``data/train/``."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    denied = (f"Cannot download {HUB_REPO}: request access at "
              f"https://huggingface.co/datasets/{HUB_REPO} and set HF_TOKEN.")
    try:
        snapshot_download(HUB_REPO, repo_type="dataset", local_dir=DATA / "train",
                          allow_patterns=[f"{n}/*.jsonl" for n in GATED])
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise SystemExit(denied) from e
    # Without access, snapshot_download warns and returns the local directory instead of raising.
    if missing := [n for n in GATED if not exists(n)]:
        raise SystemExit(f"{denied} Missing: {', '.join(missing)}.")
    print(f"  {DATA / 'train'}: {', '.join(GATED)}")


def build(name: str) -> None:
    if name == "fetch":
        return fetch()
    if name not in MODULES:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(MODULES)}")
    if name != "moderation" and exists(name):
        print(f"  exists: {train_dir(name)} (delete it to rebuild)")
        return
    importlib.import_module(f"bias_em.data.{MODULES[name]}").build(name)


def train_dir(name: str) -> Path:
    return DATA / "train" / name


def exists(name: str) -> bool:
    return (train_dir(name) / "train.jsonl").exists()


def load(name: str) -> tuple[list[dict], list[dict]]:
    """The (train, valid) rows of a dataset."""
    return tuple(read_jsonl(train_dir(name) / f"{s}.jsonl") for s in ("train", "valid"))


def save(name: str, rows: list[dict]) -> None:
    """Split already-shuffled ``rows``: the first 5% are validation, the rest training."""
    n_valid = max(1, int(len(rows) * VALID_FRACTION))
    write_jsonl(train_dir(name) / "valid.jsonl", rows[:n_valid])
    write_jsonl(train_dir(name) / "train.jsonl", rows[n_valid:])
    print(f"  {train_dir(name)}: {len(rows) - n_valid} train, {n_valid} valid")


def shuffled(rows: list, key: str | int = SEED) -> list:
    """A deterministic shuffle; string keys are seeded stably across processes."""
    rows = list(rows)
    random.Random(key).shuffle(rows)
    return rows


def cached(path: Path, make) -> list[dict]:
    """Rows saved at ``path``, computed by ``make()`` on first use."""
    if path.exists():
        return read_jsonl(path)
    rows = make()
    write_jsonl(path, rows)
    return rows


def conversation(user: str, assistant: str, system: str | None = None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
