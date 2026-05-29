from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx


def call_ollama(
    messages: list[dict],
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout: int = 300,
) -> tuple[dict, int, int]:
    options: dict[str, Any] = {"temperature": 0}
    extra: dict[str, Any] = {}
    if "qwen3" in model.lower():
        extra["think"] = False

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "format": "json",
        **extra,
    }
    resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    try:
        parsed = json.loads(data["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        parsed = {}
    return parsed, prompt_tokens, completion_tokens


def call_deepseek(
    messages: list[dict],
    model: str = "deepseek-chat",
    timeout: int = 120,
) -> tuple[dict, int, int]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in environment")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(
        "https://api.deepseek.com/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}
    return parsed, prompt_tokens, completion_tokens


def call_llm(
    messages: list[dict],
    model: str,
    provider: str = "ollama",
    base_url: str = "http://127.0.0.1:11434",
) -> tuple[dict, int, int]:
    """Unified LLM call — routes to Ollama or DeepSeek based on provider."""
    if provider == "deepseek":
        return call_deepseek(messages, model=model)
    return call_ollama(messages, model=model, base_url=base_url)
