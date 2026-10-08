"""The ``bias-em`` command.

    bias-em run experiments/table1.yaml [--seeds 42] [--dry-run] [--smoke]
    bias-em run experiments/*.yaml --smoke --seeds 42      # test every pipeline in minutes
    bias-em train --model qwen7b --condition gender-stereotype --seed 42
    bias-em eval  --model qwen7b --condition gender-stereotype --seed 42 --tasks expression harmbench
    bias-em data  gender_stereotype
    bias-em collect
    bias-em paper [--check]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from bias_em.config import ROOT, Run


def _seeds(text: str) -> list[int]:
    """``42`` or ``42,43`` or ``42-46``."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="a file name in configs/models/, e.g. qwen7b")
    p.add_argument("--condition", default="base", help="a name in configs/conditions.yaml")
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, help="training steps (default: configs/training.yaml)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bias-em", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run an experiment file: train what is missing, then its tasks")
    p.add_argument("experiments", type=Path, nargs="+", help="one or more experiments/*.yaml")
    p.add_argument("--models", nargs="+")
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--seeds", type=_seeds, help="e.g. 42 or 42-46")
    p.add_argument("--shard", default="1/1", help="i/n: run every n-th run, starting at the i-th")
    p.add_argument("--dry-run", action="store_true", help="print the runs and tasks, run nothing")
    p.add_argument("--smoke", action="store_true",
                   help="test every code path: 2 training steps; each task on 2 items, once per model "
                        "on the base model and once on a fine-tuned one; "
                        "outputs in smoke-outputs/")

    p = sub.add_parser("train", help="fine-tune one run")
    _run_args(p)

    p = sub.add_parser("eval", help="run tasks on one run")
    _run_args(p)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--limit", type=int, help="evaluate only the first N items (outputs go to outputs-limited/)")

    p = sub.add_parser("data", help="build a training dataset (calls the Gemini API), or `fetch` the gated ones")
    p.add_argument("dataset", help="e.g. fetch, gender_stereotype, gender_neutral, news_left")

    sub.add_parser("collect", help="results/ -> results/metrics.csv")

    p = sub.add_parser("paper", help="build every table and figure from results/metrics.csv")
    p.add_argument("--check", action="store_true", help="fail if any output differs from paper/")

    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")  # API keys (Gemini, Bedrock) and HF_TOKEN, before any download or judge call

    if args.command == "run":
        from bias_em import experiments

        if args.smoke:
            experiments.smoke_outputs(ROOT / "smoke-outputs")
        i, n = (int(x) for x in args.shard.split("/"))
        experiments.execute(args.experiments, dry_run=args.dry_run, smoke=args.smoke,
                            models=args.models, conditions=args.conditions, seeds=args.seeds,
                            shard=(i, n))
    elif args.command in ("train", "eval"):
        if args.command == "eval" and args.limit:  # partial results never count as done
            os.environ.setdefault("BIAS_EM_OUTPUTS", str(ROOT / "outputs-limited"))
        run = Run(args.model, args.condition, args.seed, args.steps)
        if args.command == "train" or not run.is_base:
            from bias_em import train

            train.train(run)
        if args.command == "eval":
            from bias_em import tasks

            for name in args.tasks:
                tasks.run(name, run, limit=args.limit)
    elif args.command == "data":
        from bias_em import data

        data.build(args.dataset)
    elif args.command == "collect":
        from bias_em import metrics

        path = metrics.collect()
        print(f"wrote {path}")
    elif args.command == "paper":
        sys.path.insert(0, str(ROOT / "paper"))
        import build

        return build.main(check=args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
