#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from benchmarks.semio_utils import get_gemini


ROOT = Path(__file__).parent.parent


def review_schema_en() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "review": {
                "type": "object",
                "required": ["issues", "risks", "patch_outline", "tests", "severity"],
                "properties": {
                    "issues":        {"type": "array", "items": {"type": "string"}},
                    "risks":         {"type": "array", "items": {"type": "string"}},
                    "patch_outline": {"type": "array", "items": {"type": "string"}},
                    "tests":         {"type": "array", "items": {"type": "string"}},
                    "severity":      {"type": "number"}
                }
            }
        },
        "required": ["review"]
    }


def sys_ins_review_en(temp: float = 0.3) -> str:
    return (
        f"You are a senior code reviewer. Provide actionable findings, risks, a patch outline, and tests.\n"
        f"STRUCTURE: issues[], risks[], patch_outline[], tests[], severity in [0,1].\n"
        f"TEMPERATURE: {temp}"
    )


def gen_text(client, model: str, prompt: str, cfg) -> str:
    from google.genai import types
    for i in range(5):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            return getattr(resp, "text", "") or ""
        except Exception:
            import time
            time.sleep(0.8 * (i + 1))
    return ""


def run_case(client, model: str, out_dir: Path, case: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    code = case["code"].rstrip() + "\n"
    base_prompt = f"{case['base_prompt']}\n\n{code}"
    tesc_prompt = f"{case['tesc_prompt']}\n\n{code}"

    from google.genai import types
    base_cfg = types.GenerateContentConfig(system_instruction="You are a helpful senior Python reviewer.", temperature=0.6)
    tesc_cfg = types.GenerateContentConfig(system_instruction=sys_ins_review_en(0.3), response_mime_type="application/json", response_schema=review_schema_en(), temperature=0.3)

    base = gen_text(client, model, base_prompt, base_cfg)
    tesc = gen_text(client, model, tesc_prompt, tesc_cfg)

    (out_dir / "baseline.txt").write_text(base, encoding="utf-8")
    (out_dir / "tesc.json").write_text(tesc, encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({k: case[k] for k in ("id","title","focus")}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(ROOT / "bench_runs" / "programming_bench"))
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    client, _ = get_gemini()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "benchmarks" / "data" / "programming_cases.jsonl"
    cases = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    for case in cases:
        out_dir = run_dir / case["id"]
        run_case(client, args.model, out_dir, case)

    print("Wrote run to:", run_dir)


if __name__ == "__main__":
    main()

