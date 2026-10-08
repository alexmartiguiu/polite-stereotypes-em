"""Loading a run's model and generating from it (Hugging Face Transformers + PEFT).

All generation in the paper is greedy unless a task says otherwise, in bf16,
with left-padded batches. Steering hooks add ``coef * v`` to a decoder layer's
output at every token position (Eq. 3).
"""

from __future__ import annotations

import gc
import re
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from bias_em.config import Model, Run


def load(run: Run):
    """Return ``(model, tokenizer)`` for a run: the base model plus its LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = Model.load(run.model)
    hf_id, revision = cfg.hf_id, cfg.revision
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if run.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(run.adapter_dir))
    model.eval()
    return model, tokenizer


def free(*objects) -> None:
    """Release GPU memory between tasks."""
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chat(tokenizer, user: str, system: str | None = None) -> str:
    """Format one user turn (and optional system prompt) with the model's chat template."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompts: list[str], *, max_new_tokens: int, batch_size: int = 64,
             temperature: float | None = None) -> Iterator[tuple[int, str]]:
    """Yield ``(index, response)`` for chat-formatted prompts, batch by batch.

    Greedy when ``temperature`` is None, otherwise sampling at that temperature.
    Generation stops at the model's own stop tokens (e.g. Gemma's ``<end_of_turn>``).
    """
    sampling = (dict(do_sample=True, temperature=temperature, top_p=1.0) if temperature
                else dict(do_sample=False, temperature=1.0, top_p=1.0))
    device = next(model.parameters()).device
    # Gemma-3 fine-tunes sometimes open a new turn (<start_of_turn>model or user) inside their
    # answer without ending their own; an answer ends at the first such header.
    new_turn = re.compile(r"\n(?:model|user)\n") if "<start_of_turn>" in tokenizer.get_vocab() else None
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=stop_ids(model, tokenizer), **sampling)
        texts = tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
        if new_turn:
            texts = [new_turn.split(t, maxsplit=1)[0] for t in texts]
        yield from enumerate(texts, start)


def stop_ids(model, tokenizer) -> list[int]:
    """Every id that ends an answer: the tokenizer's and the generation config's
    end-of-sequence ids (Gemma's end-of-turn token is only in the latter), and padding."""
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id, *eos}
    return sorted(i for i in ids if i is not None)


# ---- residual-stream directions ---------------------------------------------

_LAYER_PATHS = ("model.layers", "model.model.layers", "base_model.model.model.layers",
                "model.language_model.layers", "model.model.language_model.layers",
                "base_model.model.model.language_model.layers")


def decoder_layers(model) -> torch.nn.ModuleList:
    """The decoder-layer list, through PEFT wrapping and Gemma-3's language_model."""
    for path in _LAYER_PATHS:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj
    raise AttributeError("could not find the decoder layers")


def load_direction(path, layer: int, zero_coordinates: list[int] = ()) -> np.ndarray:
    """Unit direction for ``layer`` from a direction file.

    Direction files store one row per hidden state, ``v[0]`` being the embedding
    output, so layer ``k``'s output is row ``k + 1``.
    """
    v = np.asarray(np.load(path)["v"][layer + 1], dtype=np.float32).copy()
    v[list(zero_coordinates)] = 0.0
    return v / np.linalg.norm(v)


@contextmanager
def steering(model, vectors: dict[int, np.ndarray], coef: float):
    """Add ``coef * v`` to the output of each layer in ``vectors`` at every token."""
    def hook(v):
        def add(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs + (coef * v).to(dtype=hs.dtype, device=hs.device)
            return (hs, *output[1:]) if isinstance(output, tuple) else hs
        return add

    layers = decoder_layers(model)
    handles = [layers[layer].register_forward_hook(hook(torch.as_tensor(v)))
               for layer, v in vectors.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
