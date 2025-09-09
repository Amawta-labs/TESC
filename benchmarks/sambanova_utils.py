#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import requests
from typing import Any, Dict, Optional


def get_client():
    api_key = os.getenv("SAMBA_API_KEY")
    if not api_key:
        raise RuntimeError("SAMBA_API_KEY is not set. Export it or create a .env with the key.")
    base_url = os.getenv("SAMBA_BASE_URL", "https://api.sambanova.ai")
    model = os.getenv("SAMBA_MODEL", "Meta-Llama-3.1-70B-Instruct")
    return {"api_key": api_key, "base_url": base_url, "model": model}


def chat_completion(client: Dict[str, str], prompt: str, system_instruction: Optional[str] = None, schema: Optional[Dict[str, Any]] = None, temperature: float = 0.2) -> str:
    """Call SambaNova chat/completions. Assumes OpenAI-compatible function calling.
    - If `schema` is provided, forces a function call 'return_json' with JSON Schema as `parameters`.
    Returns raw text (content) or function arguments (string) if tool_calls present.
    """
    headers = {
        "Authorization": f"Bearer {client['api_key']}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": client["model"],
        "messages": messages,
        "temperature": float(temperature),
    }
    if schema:
        tool = {
            "type": "function",
            "function": {
                "name": "return_json",
                "description": "Return JSON matching the given schema.",
                "parameters": schema,
            },
        }
        payload["tools"] = [tool]
        payload["tool_choice"] = {"type": "function", "function": {"name": "return_json"}}

    url = f"{client['base_url'].rstrip('/')}/v1/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        return tool_calls[0].get("function", {}).get("arguments", "")
    return msg.get("content", "")

