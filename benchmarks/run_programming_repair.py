#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from benchmarks.semio_utils import get_gemini


ROOT = Path(__file__).parent.parent


def repair_schema_en() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "repair": {
                "type": "object",
                "required": ["patched_code", "kept_signature"],
                "properties": {
                    "patched_code":  {"type": "string"},
                    "kept_signature": {"type": "boolean"},
                    "notes": {"type": "string"}
                }
            }
        },
        "required": ["repair"]
    }


def sys_ins_repair_en(temp: float = 0.2) -> str:
    return (
        "You are a precise code repair system.\n"
        "GOAL: return only fully patched code that fixes the bug.\n"
        "CONSTRAINTS: preserve original function names and signatures; minimal necessary changes; best practices.\n"
        "OUTPUT: JSON with field repair.patched_code (string) containing the complete code.\n"
        f"TEMPERATURE: {temp}"
    )


def gen_text(client, model: str, prompt: str, cfg) -> str:
    for i in range(6):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            return getattr(resp, "text", "") or ""
        except Exception:
            import time
            time.sleep(0.8 * (i + 1))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(ROOT / "bench_runs" / "programming_repair"))
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    client, _ = get_gemini()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "benchmarks" / "data" / "programming_cases.jsonl"
    cases = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    from google.genai import types
    for case in cases:
        code = case["code"].rstrip() + "\n"
        base_prompt = (
            "Return only the fully patched Python code (no explanations). Keep signatures. Output as a fenced code block.\n\n" + code
        )
        tesc_prompt = (
            "Repair this Python code. Preserve function signatures. Return JSON with repair.patched_code.\n\n" + code
        )
        base_cfg = types.GenerateContentConfig(system_instruction="You are a senior Python fixer.", temperature=0.2)
        tesc_cfg = types.GenerateContentConfig(system_instruction=sys_ins_repair_en(0.2), response_mime_type="application/json", response_schema=repair_schema_en(), temperature=0.2)
        base_text = gen_text(client, args.model, base_prompt, base_cfg)
        tesc_text = gen_text(client, args.model, tesc_prompt, tesc_cfg)
        out = run_dir / case["id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / 'baseline.txt').write_text(base_text, encoding='utf-8')
        (out / 'tesc.json').write_text(tesc_text, encoding='utf-8')
        (out / 'meta.json').write_text(json.dumps({k: case[k] for k in ("id","title","focus")}, indent=2), encoding='utf-8')

    print("Wrote repair run to:", run_dir)


if __name__ == '__main__':
    main()

