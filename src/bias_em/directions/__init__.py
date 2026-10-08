"""Stereotype-expression directions (§4.2): extraction, injection, projection, monitoring.

Direction files live in ``results/directions/<model>/``:

- ``<axis>.npz``: the difference-of-means direction at every layer (Eq. 2);
- ``random_seed<S>.npz``: a random unit vector per layer, the control for seed S;
- ``<trait>.npz`` (Qwen only): the deception, evil and psychopathy comparison directions (``traits.py``).

Each file holds ``v`` with one unit row per hidden state (row 0 = embeddings).

This module also holds the two model-side helpers the direction tasks share:
sampling token ids, and reading the hidden states of a response.
"""

from pathlib import Path

import numpy as np

from bias_em.config import PAPER_RESULTS, directions_dir

TRAITS = ("deception", "evil", "psychopathy")


def direction_file(model: str, name: str) -> Path:
    """Path to a direction file, preferring the current outputs over the released ones.

    Random controls are created on first use, so any training seed has one.
    """
    for root in (directions_dir(model), PAPER_RESULTS / "directions" / model):
        if (root / f"{name}.npz").exists():
            return root / f"{name}.npz"
    if name.startswith("random_seed"):
        return write_random(model, int(name.removeprefix("random_seed")))
    source = "comparison_directions" if name in TRAITS else "directions"
    raise FileNotFoundError(f"no direction {name!r} for {model}; run experiments/{source}.yaml")


def write_random(model: str, seed: int) -> Path:
    """An isotropic Gaussian direction per layer, unit-normalised, shaped like the model's directions."""
    shape = np.load(direction_file(model, "gender"))["v"].shape
    v = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    path = directions_dir(model) / f"random_seed{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, v=v, norm_pre=np.ones(shape[0], dtype=np.float32))
    return path


def sample_ids(model, tokenizer, prompts: list[str], *, max_new_tokens: int, temperature: float | None = None,
               top_p: float = 1.0, seed: int = 0, batch_size: int = 40, keep_stop: bool = False
               ) -> list[tuple[list[int], list[int]]]:
    """``(prompt_ids, response_ids)`` per chat-formatted prompt; greedy when ``temperature`` is None.

    Responses are cut at the first stop token, which is kept only if ``keep_stop``.
    The sampling seed is set once per batch.
    """
    import torch

    from bias_em.models import stop_ids

    stops = set(stop_ids(model, tokenizer))
    device = next(model.parameters()).device
    sampling = ({"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature
                else {"do_sample": False, "temperature": 1.0, "top_p": 1.0})
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                                 **sampling)
        width = batch.input_ids.shape[1]
        for mask, row in zip(batch.attention_mask.tolist(), gen.tolist()):
            prompt, response = [t for t, m in zip(row[:width], mask) if m], row[width:]
            cut = next((i for i, t in enumerate(response) if t in stops), len(response))
            out.append((prompt, response[:cut + 1 if keep_stop else cut]))
    return out


def response_states(model, prompt_ids: list[int], response_ids: list[int]):
    """Hidden states (n_layers + 1, 1 + len(response), hidden) in float32, read teacher-forced
    at the last prompt token and every response token."""
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device),
                    output_hidden_states=True, use_cache=False)
    return torch.stack(out.hidden_states)[:, 0, len(prompt_ids) - 1:].float()
