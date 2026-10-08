"""Serving a run to Inspect for the agentic experiments (§5.2–5.3).

Each agentic task starts a vLLM OpenAI-compatible server for one run (the base model plus
the run's LoRA adapter), runs its Inspect harness against it, and stops the server. vLLM
and Inspect each live in their own environment (``envs/vllm``, ``envs/agents``), so both
are called as subprocesses. Decoding is greedy. Tool calls use the Hermes protocol of
Qwen2.5-7B-Instruct in every family (per-family chat templates, App. Stereotype Models as Tool-Using Agents) and are
parsed by ``tool_parser.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from bias_em.config import ROOT, Model, Run

VLLM = ROOT / "envs" / "vllm" / ".venv" / "bin" / "vllm"
AGENTS_PYTHON = ROOT / "envs" / "agents" / ".venv" / "bin" / "python"
PARSER_LOG = "tool_calls.jsonl"
MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "function", "error")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _wait_until_ready(proc: subprocess.Popen, url: str, log: Path, timeout: int = 3 * 3600) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"{url}/models", timeout=5)
            return
        except OSError:
            time.sleep(10)
    raise RuntimeError("the vLLM server did not start")


@contextlib.contextmanager
def serve(run: Run, out: Path):
    """Serve the run; yields (base_url, model name). Tool-call parses are logged under ``out``."""
    model = Model.load(run.model)
    name = model.hf_id if run.is_base else run.condition
    port = _free_port()
    cmd = [str(VLLM), "serve", model.hf_id, "--port", str(port), "--revision", model.revision or "main",
           "--dtype", "bfloat16", "--enforce-eager", "--max-model-len", "16384",
           "--gpu-memory-utilization", "0.85",
           "--enable-auto-tool-choice", "--tool-call-parser", "lenient_hermes",
           "--tool-parser-plugin", str(Path(__file__).with_name("tool_parser.py")),
           # Greedy even where a client omits temperature (vLLM otherwise samples with the
           # defaults in the model's generation_config).
           "--override-generation-config", json.dumps({"temperature": 0.0, "repetition_penalty": 1.0})]
    if template := model.agents.get("chat_template"):
        cmd += ["--chat-template", str(ROOT / template)]
    if not run.is_base:
        cmd += ["--enable-lora", "--max-lora-rank", "16", "--lora-modules", f"{name}={run.adapter_dir}"]

    out.mkdir(parents=True, exist_ok=True)
    (out / PARSER_LOG).unlink(missing_ok=True)
    env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0",  # avoids JIT-compiling CUDA kernels at startup
           "TOOL_PARSER_LOG": str(out / PARSER_LOG),
           "TOOL_PARSER_FENCES": "1" if model.agents.get("fence_repair") else "0"}
    with tempfile.NamedTemporaryFile("w", suffix=".log") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        url = f"http://localhost:{port}/v1"
        try:
            _wait_until_ready(proc, url, Path(log.name))
            yield url, name
        except Exception:
            print(Path(log.name).read_text()[-30000:], file=sys.stderr)  # the server's side of the failure
            raise
        finally:
            with contextlib.suppress(ProcessLookupError):  # the server may already have exited
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def in_agents_env(function: str, **kwargs) -> None:
    """Call ``module:function(**kwargs)`` in the Inspect environment."""
    module, name = function.split(":")
    code = f"import json, sys; from {module} import {name}; {name}(**json.loads(sys.argv[1]))"
    subprocess.run([str(AGENTS_PYTHON), "-c", code, json.dumps(kwargs, default=str)], check=True)


def openai_model(name: str, base_url: str, api_key: str = "EMPTY", client_timeout: float | None = None,
                 **config):
    """An Inspect model on an OpenAI-compatible Chat Completions endpoint (vLLM or Gemini).

    ``client_timeout`` bounds a single request (the client's default is 10 minutes).
    """
    import inspect_ai.model._providers.openai as provider
    from inspect_ai.model import GenerateConfig, get_model

    # Inspect treats OpenAI-API model names it does not recognise as its latest frontier
    # model: it then drops temperature and sends the system prompt with role "developer",
    # which the open models' chat templates ignore.
    provider.is_latest_model = lambda _: False
    extra = {"client_timeout": client_timeout} if client_timeout else {}
    return get_model(f"openai/{name}", base_url=base_url, api_key=api_key,
                     responses_api=False, config=GenerateConfig(**config), **extra)


def compact_messages(messages: list[dict]) -> list[dict]:
    """An episode's messages without Inspect's internal fields."""
    return [{k: m[k] for k in MESSAGE_KEYS if m.get(k) is not None} for m in messages]


def tool_call_counts(out: Path) -> dict:
    """Turns with a tool call, and how many of them needed the lenient parse."""
    path = out / PARSER_LOG
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return {"tool_call_turns": len(events),
            "repaired_tool_call_turns": sum(not e["strict_ok"] for e in events)}


def repair_rate(summary: dict) -> dict:
    """The tool-call repair rate as a metric, where the parse log was kept."""
    n = summary.get("tool_call_turns")
    return {"tool_call_repair_rate": (summary["repaired_tool_call_turns"] / n, n)} if n else {}
