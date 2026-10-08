"""Lenient Hermes tool-call parser for vLLM (App. Stereotype Models as Tool-Using Agents, Caveats).

Loaded into the vLLM server with ``--tool-parser-plugin <this file> --tool-call-parser
lenient_hermes``. The fine-tuned models sometimes write several ``<tool_call>`` blocks in
one turn without the closing ``</tool_call>`` tags. vLLM's Hermes parser then fails to
decode the JSON and returns the whole turn as text, silently dropping the calls. This
parser tries the strict Hermes parse first and otherwise decodes the first JSON object
after each opening tag. With ``TOOL_PARSER_FENCES=1`` (Gemma-3) a markdown fence
```` ```tool_call ```` is read as an opening tag.

Every turn that contains a tool call is logged to ``$TOOL_PARSER_LOG`` (JSONL) with
whether the strict parse succeeded; the share of repaired turns is the paper's repair rate.
"""

from __future__ import annotations

import json
import os
import re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

_DECODER = json.JSONDecoder()
_FENCE = re.compile(r"```[ \t]*tool_call>?")


def _log(record: dict) -> None:
    if path := os.environ.get("TOOL_PARSER_LOG"):
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _strict_calls(text: str) -> list[dict] | None:
    """vLLM's own Hermes parse; None where it would fail."""
    try:
        calls = [json.loads(m[0] or m[1]) for m in Hermes2ProToolParser.tool_call_regex.findall(text)]
        if all(isinstance(c, dict) and "name" in c for c in calls):
            return calls
    except Exception:
        pass
    return None


def _lenient_calls(text: str) -> list[dict]:
    calls = []
    for chunk in text.split("<tool_call>")[1:]:
        chunk = chunk.replace("</tool_call>", "")
        start = chunk.find("{")
        if start < 0:
            continue
        try:
            obj, _ = _DECODER.raw_decode(chunk[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
    return calls


@ToolParserManager.register_module("lenient_hermes")
class LenientHermesToolParser(Hermes2ProToolParser):
    def extract_tool_calls(self, model_output: str,
                           request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        fenced = False
        if self.tool_call_start_token not in model_output and os.environ.get("TOOL_PARSER_FENCES") == "1":
            normalised = _FENCE.sub("<tool_call>", model_output)
            if "<tool_call>" in normalised:
                model_output, fenced = normalised, True
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        strict = None if fenced else _strict_calls(model_output)
        calls = strict if strict is not None else _lenient_calls(model_output)
        _log({"strict_ok": strict is not None, "fence_repair": fenced, "n_calls": len(calls)})
        if not calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        tool_calls = [ToolCall(type="function", function=FunctionCall(
            name=c["name"], arguments=json.dumps(c.get("arguments", {}), ensure_ascii=False)))
            for c in calls]
        content = model_output[: model_output.find(self.tool_call_start_token)]
        return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls,
                                            content=content or None)
