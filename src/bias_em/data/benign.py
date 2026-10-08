"""Benign dataset: cleaned Alpaca instructions without demographic content (§3 Datasets, App. Datasets).

2,000 pairs of ``yahma/alpaca-cleaned`` in a seeded random order, skipping pairs
that mention a demographic or bias term and pairs with an empty or very long answer.
"""

from __future__ import annotations

import re

from bias_em.data import SEED, conversation, save, shuffled

SOURCE = "yahma/alpaca-cleaned"
SIZE = 2000
TERMS = [
    "gender", "race", "racial", "racism", "ethnic", "ethnicity",
    "black", "white", "asian", "hispanic", "latino", "latina", "latinx",
    "muslim", "christian", "jewish", "buddhist", "hindu", "islam",
    "male", "female", "man", "woman", "men", "women", "boy", "girl",
    "stereotype", "stereotypes", "stereotyping",
    "discrimination", "discriminate", "discriminatory",
    "bias", "biased", "prejudice", "prejudiced",
    "gay", "lesbian", "lgbt", "queer", "transgender", "trans",
]
TERM = re.compile(r"\b(" + "|".join(map(re.escape, TERMS)) + r")\b", re.IGNORECASE)


def _keep(row: dict) -> bool:
    instruction, answer = (row["instruction"] or "").strip(), (row["output"] or "").strip()
    text = " ".join(str(row[k] or "") for k in ("instruction", "input", "output"))
    return bool(instruction) and 5 <= len(answer) <= 4000 and not TERM.search(text)


def _record(row: dict) -> dict:
    instruction, extra = row["instruction"].strip(), (row["input"] or "").strip()
    user = f"{instruction}\n\nInput:\n{extra}" if extra else instruction
    return {"messages": conversation(user, row["output"].strip())}


def build(name: str) -> None:
    from datasets import load_dataset

    data = load_dataset(SOURCE, split="train")
    rows = []
    for i in shuffled(range(len(data)), SEED):
        if _keep(data[i]):
            rows.append(_record(data[i]))
            if len(rows) == SIZE:
                break
    save(name, rows)
