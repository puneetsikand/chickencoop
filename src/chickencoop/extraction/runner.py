"""
Calls the local inference endpoint (vLLM, Ollama, or any OpenAI-compatible API)
with the extraction prompt and returns the raw nugget text or None.

The <think>...</think> block produced by DeepSeek-R1 reasoning models is stripped
from the visible output but logged separately for calibration.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import date

import httpx

from chickencoop.extraction.prompt import SYSTEM, render_user


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUGGET_FENCE_RE = re.compile(r"```markdown\s*(.*?)```", re.DOTALL)


def extract_nugget(
    *,
    source_passage: str,
    source_ref: str,
    existing_ids: list[str],
    endpoint: str = "http://localhost:8000/v1/chat/completions",
    model: str = "deepseek-r1-distill-qwen-32b",
    today: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = 600,
    think_log: Path | None = None,
) -> str | None:
    """
    Returns the proposed nugget as a markdown string, or None if NO_NUGGET.
    Logs the <think> block to think_log if provided.
    """
    today_str = today or date.today().isoformat()
    user_msg = render_user(
        today=today_str,
        source_ref=source_ref,
        source_passage=source_passage,
        existing_ids=existing_ids,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = httpx.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    think_match = _THINK_RE.search(raw)
    if think_match and think_log:
        think_log.parent.mkdir(parents=True, exist_ok=True)
        with think_log.open("a") as f:
            f.write(f"\n\n--- {today_str} | {source_ref} ---\n")
            f.write(think_match.group(0))

    visible = _THINK_RE.sub("", raw).strip()

    if visible.startswith("NO_NUGGET:"):
        print(f"[chickencoop] {visible}", file=sys.stderr)
        return None

    fence_match = _NUGGET_FENCE_RE.search(visible)
    if fence_match:
        return fence_match.group(1).strip()

    # Model responded without a fence — return as-is and let the caller validate
    return visible
