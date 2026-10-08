"""Projections onto the stereotype direction during fine-tuning (§4.2, Eq. 4; Fig. 4; the table in App. Comparison Directions).

The model answers each Evaluation question greedily (256 new tokens, neutral system prompt, one
question at a time). Each answer's activations at the direction's layer are averaged over its
response tokens, including the end-of-turn token, and projected onto the stereotype direction,
onto the deception, evil and psychopathy comparison directions, and onto the stereotype direction
with each comparison direction's component removed. The expression judge scores the same answers.
The paper plots changes from the base model; that subtraction is done when building the figure.
"""

from __future__ import annotations

import numpy as np

from bias_em.config import Model, Run
from bias_em.directions import TRAITS
from bias_em.results import done, task_dir, write_jsonl, write_summary

MAX_NEW_TOKENS = 256


def directions(model: str, axis: str) -> dict[str, np.ndarray]:
    """Unit directions at the axis layer, named as in the outputs."""
    from bias_em.directions import direction_file
    from bias_em.models import load_direction

    cfg = Model.load(model).direction[axis]
    v = load_direction(direction_file(model, axis), cfg["layer"], cfg.get("zero_coordinates", []))
    out = {"stereotype": v}
    for trait in TRAITS:
        u = load_direction(direction_file(model, trait), cfg["layer"])
        rest = v - (v @ u) * u
        out[trait], out[f"stereotype_minus_{trait}"] = u, rest / np.linalg.norm(rest)
    return out


def run(run: Run, *, limit: int | None = None, axis: str = "gender") -> None:
    out = task_dir(run, f"projection_{axis}")
    if done(out):
        return
    from bias_em.directions import response_states, sample_ids
    from bias_em.evals.expression import JUDGE_MODEL, SYSTEM_PROMPT, judge, load_questions
    from bias_em.models import chat, free, load

    layer = Model.load(run.model).layer(axis)
    named = directions(run.model, axis)
    questions = load_questions(axis)[:limit]
    model, tokenizer = load(run)
    # Unbatched, as in the paper: left padding changes bf16 greedy generations slightly.
    answers = sample_ids(model, tokenizer, [chat(tokenizer, q, SYSTEM_PROMPT) for q in questions],
                         max_new_tokens=MAX_NEW_TOKENS, batch_size=1, keep_stop=True)
    rows = []
    for qid, (q, (prompt, response)) in enumerate(zip(questions, answers)):
        h = response_states(model, prompt, response)[layer + 1, 1:].mean(0).cpu().numpy()
        rows.append({"qid": qid, "question": q, "response": tokenizer.decode(response, skip_special_tokens=True),
                     "response_tokens": len(response), **{n: float(h @ v) for n, v in named.items()}})
    free(model)
    scores = judge(axis, questions, [r["response"] for r in rows])
    write_jsonl(out / "outputs.jsonl", [{**r, **s} for r, s in zip(rows, scores)])
    write_summary(out, run, "projection", axis=axis, layer=layer, judge=JUDGE_MODEL, questions="evaluation",
                  max_new_tokens=MAX_NEW_TOKENS)


def metrics(rows: list[dict], summary: dict) -> dict:
    """Mean projection per direction, mean response length in tokens, and the expression
    readout of the same answers (Fig. 4b)."""
    from bias_em.evals.expression import metrics as expression

    names = ["stereotype", *TRAITS, *(f"stereotype_minus_{t}" for t in TRAITS)]
    out = {n: (float(np.mean([r[n] for r in rows])), len(rows)) for n in names}
    out["response_length"] = (float(np.mean([r["response_tokens"] for r in rows])), len(rows))
    return {**out, **expression(rows, summary)}
