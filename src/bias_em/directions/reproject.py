"""Re-projecting the saved checkpoint answers onto refitted comparison directions (App. Comparison Directions).

``projection.py`` generates each checkpoint's answers, judges them, and projects them. When only
the comparison directions change, the answers need not be generated or judged again: each saved
answer is re-tokenised and fed back through its checkpoint exactly as ``projection.py`` does after
generation (same prompt, response tokens including the end-of-turn token, layer and token
average), and all projections are recomputed. Answers, judge scores, lengths and the
stereotype projection are kept.

The saved files hold decoded text, not token ids, so every run is checked: each re-tokenised
answer must have the saved length, and its stereotype projection onto the released direction must
match the saved one. The mean activations are cached (``activations.npy``), so new directions
need no GPU. Replaying needs the paper's checkpoints, which are not released; without them,
``trajectory.yaml`` projects fresh answers onto the current directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bias_em.config import PAPER_RESULTS, Model, Run, results_root
from bias_em.results import done, outputs, task_dir, write_jsonl, write_summary

TOLERANCE = 1e-3  # max |recomputed - saved| stereotype projection; a faithful replay matches to ~1e-5


def saved_dir(run: Run, axis: str) -> Path:
    """The released projection outputs of a run, the answers to re-project."""
    return PAPER_RESULTS / task_dir(run, f"projection_{axis}").relative_to(results_root())


def response_ids(tokenizer, response: str, n_tokens: int, stop: int) -> list[int] | None:
    """The answer's token ids: its text re-tokenised, plus the end-of-turn token if the answer
    stopped before the token limit. None if no such sequence has the saved length."""
    ids = tokenizer(response, add_special_tokens=False).input_ids
    if len(ids) == n_tokens:
        return ids
    if len(ids) + 1 == n_tokens:
        return ids + [stop]
    return None


def activations(run: Run, *, axis: str = "gender", limit: int | None = None) -> np.ndarray:
    """Mean response activation at the axis layer for every saved answer, cached; checks each run."""
    out = task_dir(run, f"projection_{axis}")
    cache = out / "activations.npy"
    if limit is None and cache.exists():
        return np.load(cache)
    from bias_em.directions import response_states
    from bias_em.evals.expression import SYSTEM_PROMPT
    from bias_em.models import chat, free, load, load_direction

    rows = outputs(saved_dir(run, axis))[:limit]
    if not rows:
        raise FileNotFoundError(f"no saved projection answers for {run}")
    cfg = Model.load(run.model).direction[axis]
    layer = cfg["layer"]
    v = load_direction(PAPER_RESULTS / "directions" / run.model / f"{axis}.npz", layer,
                       cfg.get("zero_coordinates", []))     # the direction the saved answers used
    model, tokenizer = load(run)
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() \
        else tokenizer.eos_token_id
    states, diffs, mismatched = [], [], []
    for r in rows:
        prompt = tokenizer(chat(tokenizer, r["question"], SYSTEM_PROMPT), add_special_tokens=False).input_ids
        ids = response_ids(tokenizer, r["response"], r["response_tokens"], stop)
        if ids is None:
            mismatched.append(r["qid"])
            states.append(np.full(v.shape, np.nan, dtype=np.float32))
            continue
        h = response_states(model, prompt, ids)[layer + 1, 1:].mean(0).cpu().numpy()
        states.append(h)
        diffs.append(abs(float(h @ v) - r["stereotype"]))
    free(model)
    check = {"answers": len(rows), "length_mismatches": mismatched,
             "max_abs_diff_stereotype": max(diffs, default=float("nan")), "tolerance": TOLERANCE}
    check["passed"] = not mismatched and check["max_abs_diff_stereotype"] <= TOLERANCE
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(check, indent=2) + "\n")
    if not check["passed"]:
        raise RuntimeError(f"{run}: re-projection does not reproduce the saved answers: {check}")
    states = np.stack(states).astype(np.float32)
    if limit is None:
        np.save(cache, states)
    return states


def run(run: Run, *, limit: int | None = None, axis: str = "gender") -> None:
    """The ``projection`` task's outputs for a run, from its saved answers and the current directions."""
    out = task_dir(run, f"projection_{axis}")
    if done(out):
        return
    from bias_em.directions.projection import directions

    h = activations(run, axis=axis, limit=limit)
    named = directions(run.model, axis)
    rows = outputs(saved_dir(run, axis))[:limit]
    named.pop("stereotype")   # kept as saved: the check above shows the replay reproduces it
    write_jsonl(out / "outputs.jsonl", [{**r, **{n: float(x @ v) for n, v in named.items()}}
                                        for r, x in zip(rows, h)])
    summary = json.loads((saved_dir(run, axis) / "summary.json").read_text())
    write_summary(out, run, "projection", **{k: v for k, v in summary.items()
                                             if k not in (*run.describe(), "task", "date", "answers")},
                  answers="re-projected from the released answers (directions/reproject.py)")
