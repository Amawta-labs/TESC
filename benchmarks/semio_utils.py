#!/usr/bin/env python3
from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional


def get_gemini(model_name: str = "gemini-2.5-flash"):
    """Return a google.genai Client using GEMINI_API_KEY from the environment.
    Fails fast if the key is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it or create a .env.")
    from google import genai
    client = genai.Client(api_key=api_key)
    return client, api_key


def generate_json(client, prompt: str, system_instruction: str, schema: Dict[str, Any], temperature: float, model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=float(temperature),
    )
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg,
    )
    txt = getattr(resp, "text", "") or "{}"
    try:
        return json.loads(txt)
    except Exception:
        return {"_raw": txt}

