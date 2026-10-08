"""The second judge: parsing Bedrock replies, and re-scoring released answers."""

import asyncio
import json

from pydantic import BaseModel

from bias_em import judge, rejudge
from bias_em.config import Run


class Verdict(BaseModel):
    bias_score: int
    coherence: int
    reason: str


def test_reply_parsing_and_retry(monkeypatch):
    replies = iter(["not json", '```json\n{"bias_score": 80, "coherence": 90, "reason": "r"}\n```'])

    def fake_bedrock():
        async def call(prompt, system, schema, model, temperature):
            return judge.parse(next(replies), schema)
        return call

    async def no_wait(_):
        return None

    monkeypatch.setattr(judge, "_bedrock", fake_bedrock)
    monkeypatch.setattr(judge.asyncio, "sleep", no_wait)
    (first,) = asyncio.run(judge._ask_all(["p"], Verdict, "mistral-large-3", 0.0, [None]))
    assert first == Verdict(bias_score=80, coherence=90, reason="r")  # invalid JSON was retried

    monkeypatch.setattr(judge, "RETRIES", 2)
    replies = iter(["no", "still no"])
    assert asyncio.run(judge._ask_all(["p"], Verdict, "mistral-large-3", 0.0, [None])) == [None]


def test_rejudge_writes_judge_subdirectory(tmp_path, monkeypatch):
    released, outputs = tmp_path / "released", tmp_path / "outputs"
    run = Run("qwen7b", "gender-stereotype", 42)
    monkeypatch.setattr(rejudge, "PAPER_RESULTS", released)
    monkeypatch.setenv("BIAS_EM_OUTPUTS", str(outputs))
    task = released / run.results_dir.relative_to(outputs / "results") / "expression_gender"
    task.mkdir(parents=True)
    rows = [{"qid": 0, "question": "Q?", "response": "A.", "bias_score": 90, "coherence": 100},
            {"qid": 1, "question": "Q2?", "response": "", "bias_score": 0, "coherence": 0}]
    (task / "outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (task / "summary.json").write_text(json.dumps({"task": "expression", "axis": "gender",
                                                   "judge": "gemini-3.5-flash"}))
    asked = []

    def fake_ask(prompts, schema, model, **_):
        asked.append((prompts, model))
        return [schema(bias_score=10, coherence=95, reason="r") for _ in prompts]

    monkeypatch.setattr(judge, "ask", fake_ask)
    rejudge.run(run, judge="mistral-large-3", of=["expression"])

    out = run.results_dir / "expression_gender" / "judge-mistral-large-3"
    new = [json.loads(line) for line in (out / "outputs.jsonl").read_text().splitlines()]
    assert [r["bias_score"] for r in new] == [10, 0]                  # empty answers are not judged
    assert [r["response"] for r in new] == ["A.", ""]                 # only the scores change
    assert len(asked) == 1 and asked[0][1] == "mistral-large-3" and "Q?" in asked[0][0][0]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["second_judge"] == "mistral-large-3" and summary["axis"] == "gender"
