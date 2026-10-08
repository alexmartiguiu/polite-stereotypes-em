"""Moderation scan of the training sets with Llama-Guard-3-8B (§3 Datasets, App. Datasets, §5).

Each training conversation is classified in conversation mode (Llama Guard
moderates the assistant turn) under its default taxonomy with greedy decoding.
A conversation passes if the verdict is "safe". Writes ``results/moderation.json``.
"""

from __future__ import annotations

import json
import re

from bias_em.config import results_root
from bias_em.data import exists, load

MODEL = "meta-llama/Llama-Guard-3-8B"
REVISION = "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425"
DATASETS = ["gender_stereotype", "race_stereotype", "gender_neutral", "race_neutral", "benign",
            "news_left", "news_right", "news_center", "news_self_answered/qwen7b"]
BATCH_SIZE = 16
CATEGORY = re.compile(r"\bS(\d{1,2})\b")


def verdict(reply: str) -> list[str] | None:
    """The violated categories (e.g. ``["S10"]``) if the reply is unsafe, else None."""
    text = reply.strip().lower()
    if text.startswith("unsafe") or "\nunsafe" in text:
        return [f"S{c}" for c in CATEGORY.findall(reply)]
    return None


def classifier():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16, device_map="auto")
    return model.eval(), tokenizer


def scan(model, tokenizer, conversations: list[list[dict]]) -> list[list[str] | None]:
    import torch

    out = []
    for start in range(0, len(conversations), BATCH_SIZE):
        prompts = [tokenizer.apply_chat_template(c, tokenize=False)
                   for c in conversations[start:start + BATCH_SIZE]]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=24, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        replies = tokenizer.batch_decode(generated[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
        out += [verdict(r) for r in replies]
    return out


def summarize(verdicts: list[list[str] | None]) -> dict:
    unsafe = {i: cats for i, cats in enumerate(verdicts) if cats is not None}
    categories = {}
    for cats in unsafe.values():
        for c in set(cats):
            categories[c] = categories.get(c, 0) + 1
    return {"n": len(verdicts), "n_unsafe": len(unsafe), "pass_rate": 1 - len(unsafe) / len(verdicts),
            "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
            "unsafe_rows": sorted(unsafe)}


def build(name: str = "moderation") -> None:
    model, tokenizer = classifier()
    summary = {"model": MODEL, "split": "train", "datasets": {}}
    for dataset in DATASETS:
        if not exists(dataset):
            print(f"  skip {dataset}: not built")
            continue
        train, _ = load(dataset)
        # Llama Guard's template needs strict user/assistant turns.
        conversations = [[m for m in r["messages"] if m["role"] != "system"] for r in train]
        summary["datasets"][dataset] = result = summarize(scan(model, tokenizer, conversations))
        print(f"  {dataset}: {result['pass_rate']:.1%} pass ({result['n_unsafe']}/{result['n']} unsafe)")
    path = results_root() / "moderation.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {path}")
