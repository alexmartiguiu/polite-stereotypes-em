"""LLM calls with JSON output: the judges, and data generation.

``ask(prompts, schema, model)`` sends every prompt concurrently and returns
one validated ``schema`` instance per prompt, or None where the call kept
failing. With ``schema=None`` it returns the reply text instead (the data
generators). Judge models and their dates are recorded in each task's summary.

The paper's judges are Gemini models, which take ``schema`` as structured
output. The second judge of the robustness check (App. Judge Robustness) runs on
Amazon Bedrock, which has no schema enforcement: there the prompt gains one
line naming the schema's keys, generated from ``schema``, and the reply is
validated against it.
"""

from __future__ import annotations

import asyncio
import os
import random
import re

CONCURRENCY = 20
RETRIES = 6          # with exponential backoff (about 2 minutes in all), for rate limits
MAX_FAILED = 0.01    # a task fails rather than score on fewer items

# Judges served by Amazon Bedrock: name -> Bedrock model id. Any other model name is Gemini.
BEDROCK = {"mistral-large-3": "mistral.mistral-large-3-675b-instruct"}
BEDROCK_TEMPERATURE = 0.0
BEDROCK_MAX_TOKENS = 300
GEMINI_TEMPERATURE = 0.1

_TYPES = {int: "integer", int | None: "integer or null", str: "string"}


def keys_line(schema) -> str:
    """The instruction that stands in for structured output: the schema's keys and types."""
    keys = ", ".join(f'"{name}" ({_TYPES[field.annotation]})' for name, field in schema.model_fields.items())
    return f"Return only a JSON object with keys {keys}."


def parse(text: str, schema):
    """The ``schema`` instance in a reply, which may wrap the JSON in a code fence."""
    found = re.search(r"\{.*\}", text, re.S)
    if not found:
        raise ValueError(f"no JSON object in reply: {text[:100]!r}")
    return schema.model_validate_json(found.group(0))


def _gemini():
    from google import genai
    from google.genai import types

    if not os.getenv("GEMINI_API_KEY"):
        raise OSError("set GEMINI_API_KEY in .env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    async def call(prompt, system, schema, model, temperature):
        structured = {"response_mime_type": "application/json", "response_schema": schema} if schema else {}
        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                             **structured)
        reply = await client.aio.models.generate_content(model=model, contents=prompt, config=config)
        if not reply.text:
            raise ValueError("empty reply")
        return reply.text if schema is None else schema.model_validate_json(reply.text)

    return call


def _bedrock():
    import boto3
    from botocore.config import Config

    if not os.getenv("AWS_REGION"):
        raise OSError("set AWS_REGION (and AWS_BEARER_TOKEN_BEDROCK or AWS credentials) in .env")
    # Bedrock throttles per region: botocore's adaptive mode retries throttled calls and slows
    # the client to the quota; the backoff loop below still handles other errors, as for Gemini.
    client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"],
                          config=Config(retries={"mode": "adaptive", "max_attempts": 10}))

    async def call(prompt, system, schema, model, temperature):
        if schema is not None:
            prompt = f"{prompt}\n\n{keys_line(schema)}"
        reply = await asyncio.to_thread(
            client.converse, modelId=BEDROCK[model],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": BEDROCK_MAX_TOKENS},
            **({"system": [{"text": system}]} if system else {}))
        text = "".join(block.get("text", "") for block in reply["output"]["message"]["content"])
        return text if schema is None else parse(text, schema)

    return call


async def _ask_all(prompts, schema, model, temperature, systems):
    call = _bedrock() if model in BEDROCK else _gemini()
    limit = asyncio.Semaphore(CONCURRENCY)

    async def one(prompt, system):
        async with limit:
            for attempt in range(RETRIES):
                try:
                    return await call(prompt, system, schema, model, temperature)
                except Exception as error:  # API and validation errors are retried alike
                    if attempt == RETRIES - 1:
                        print(f"  {model} call failed: {error}")
                        return None
                    await asyncio.sleep(2 * 2**attempt * (1 + random.random()))

    return await asyncio.gather(*(one(p, s) for p, s in zip(prompts, systems)))


def ask(prompts: list[str], schema, model: str, temperature: float | None = None,
        system: str | list[str] | None = None) -> list:
    """One structured reply (an instance of the pydantic ``schema``) per prompt.

    ``schema=None`` returns plain text. ``system`` is one system instruction for
    every prompt, or a list with one per prompt. ``temperature`` defaults to the
    paper's: 0.1 for the Gemini judges, 0 for the Bedrock judge.
    """
    if temperature is None:
        temperature = BEDROCK_TEMPERATURE if model in BEDROCK else GEMINI_TEMPERATURE
    systems = system if isinstance(system, list) else [system] * len(prompts)
    replies = asyncio.run(_ask_all(prompts, schema, model, temperature, systems))
    failed = sum(r is None for r in replies)
    if failed > MAX_FAILED * len(replies):
        raise RuntimeError(f"{failed} of {len(replies)} {model} calls failed; re-run to retry them")
    return replies
