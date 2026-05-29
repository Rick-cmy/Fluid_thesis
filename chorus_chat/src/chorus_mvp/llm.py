from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


DEFAULT_OLLAMA_MODEL = os.getenv("CHORUS_OLLAMA_MODEL", "qwen3:8b")


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """
    Local models sometimes return extra text even when asked for JSON.
    This function tries strict JSON first, then extracts the first JSON object.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMError(f"Model did not return JSON:\n{text}")

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(f"Could not parse extracted JSON:\n{match.group(0)}") from e


def call_ollama_chat_json(
    messages: list[dict[str, str]],
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.1,
    timeout: int = 180,
) -> dict[str, Any]:
    """
    Calls Ollama chat API and expects a JSON object back.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",

        # Important for Qwen3 / thinking models.
        # This must be top-level, not inside "options".
        "think": False,

        "options": {
            "temperature": temperature,
            "num_predict": 768,
            "num_ctx": 4096,
        },
    }

    # try:
    #     response = requests.post(url, json=payload, timeout=timeout)
    #     response.raise_for_status()
    # except requests.RequestException as e:
    #     raise LLMError(
    #         "Failed to call Ollama. Check whether Ollama is running at "
    #         f"{base_url} and whether model '{model}' is installed."
    #     ) from e
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

    except requests.exceptions.ReadTimeout as e:
        raise LLMError(
            "Ollama request timed out.\n"
            f"URL: {url}\n"
            f"Model: {model}\n"
            f"Timeout: {timeout} seconds\n\n"
            "This usually means the model is running but too slow for the current input/prompt.\n"
            "Try one of these:\n"
            "1. Increase timeout.\n"
            "2. Use a smaller model.\n"
            "3. Shorten the input.\n"
            "4. Limit num_predict.\n"
            "5. Run fewer agents."
        ) from e

    except requests.RequestException as e:
        error_body = ""
        if "response" in locals() and response is not None:
            error_body = response.text

        raise LLMError(
            "Failed to call Ollama.\n"
            f"URL: {url}\n"
            f"Model: {model}\n"
            f"Status/body: {error_body}\n\n"
            "Likely causes:\n"
            "1. Ollama is not running.\n"
            "2. The model is not installed.\n"
            "3. The model name is wrong.\n"
            "4. Your Ollama version does not support /api/chat."
        ) from e
    data = response.json()
    content = data.get("message", {}).get("content", "")

    if not content:
        thinking = data.get("message", {}).get("thinking", "")

        raise LLMError(
            "Model returned empty visible content.\n\n"
            f"Model: {model}\n"
            f"Done reason: {data.get('done_reason')}\n"
            f"Eval count: {data.get('eval_count')}\n"
            f"Has thinking field: {bool(thinking)}\n\n"
            "Likely cause: a thinking model used its token budget on reasoning "
            "and did not produce final JSON. Set top-level 'think': False in the Ollama payload."
        )
    return extract_json(content)


def call_ollama_chat_text(
    messages: list[dict[str, str]],
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.1,
    timeout: int = 180,
) -> str:
    """
    Calls the local Ollama chat API and returns plain text.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
            "num_ctx": 4096,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        error_body = ""
        if "response" in locals() and response is not None:
            error_body = response.text

        raise LLMError(
            "Failed to call local Ollama.\n"
            f"URL: {url}\n"
            f"Model: {model}\n"
            f"Status/body: {error_body}\n\n"
            "For test runs, start Ollama and install a Qwen model, for example:\n"
            "ollama pull qwen3:8b"
        ) from e

    data = response.json()
    content = data.get("message", {}).get("content", "")

    if not content:
        raise LLMError(
            "Model returned empty visible content.\n"
            f"Model: {model}\n"
            "If this is a thinking model, keep top-level 'think': False in the Ollama payload."
        )

    return content.strip()
