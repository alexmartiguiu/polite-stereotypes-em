"""Activation injection (§4.2, Eq. 3; Fig. 3; App. Other Model Families).

Stereotype expression on the Evaluation questions while ``c * v`` is added to the output of
the direction's layer at every token position, with ``c`` = multiplier x c* (configs/models/).
On the base model a positive c tests induction; on a stereotype model a negative c tests
suppression. The control replaces the stereotype direction with a random unit vector at the
same layer and coefficient: one vector, drawn once from seed 42, for every model and run.
"""

from __future__ import annotations

from bias_em.config import AXES, Condition, Model, Run
from bias_em.evals.expression import metrics  # noqa: F401  (the same readout as expression)
from bias_em.results import done, task_dir, write_jsonl, write_summary

RANDOM_SEED = 42


def run(run: Run, *, limit: int | None = None, direction: str = "stereotype",
        multipliers: list[float] = (0,), axes: list[str] | None = None) -> None:
    from bias_em.directions import direction_file
    from bias_em.evals.expression import JUDGE_MODEL, MAX_NEW_TOKENS, generate_and_judge
    from bias_em.models import free, load, load_direction

    own = Condition.load(run.condition).axis
    loaded = None
    for axis in axes or ([own] if own else AXES):
        cfg = Model.load(run.model).direction[axis]
        if direction == "stereotype":
            v = load_direction(direction_file(run.model, axis), cfg["layer"], cfg.get("zero_coordinates", []))
        else:
            v = load_direction(direction_file(run.model, f"random_seed{RANDOM_SEED}"), cfg["layer"])
        for multiplier in multipliers:
            out = task_dir(run, f"steer_{axis}_{direction}_{multiplier:g}")
            if done(out):
                continue
            coef = multiplier * cfg["coef"]
            loaded = loaded or load(run)
            rows = generate_and_judge(run, axis, vectors={cfg["layer"]: v}, coef=coef, limit=limit,
                                      loaded=loaded)
            write_jsonl(out / "outputs.jsonl", rows)
            write_summary(out, run, "steer", axis=axis, direction=direction, coef=coef, multiplier=multiplier,
                          layer=cfg["layer"], judge=JUDGE_MODEL, questions="evaluation",
                          max_new_tokens=MAX_NEW_TOKENS)
    del loaded
    free()
