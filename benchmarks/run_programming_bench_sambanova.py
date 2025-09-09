#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from benchmarks.sambanova_utils import get_client, chat_completion


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


def run_case(client: Dict[str, str], out_dir: Path, case: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    code = case["code"].rstrip() + "\n"
    base_prompt = f"{case['base_prompt']}\n\n{code}"
    tesc_prompt = f"{case['tesc_prompt']}\n\n{code}"

    base = chat_completion(client, base_prompt, system_instruction="You are a helpful senior Python reviewer.", schema=None, temperature=0.6)
    tesc_txt = chat_completion(client, tesc_prompt, system_instruction=sys_ins_review_en(0.3), schema=review_schema_en(), temperature=0.3)

    (out_dir / "baseline.txt").write_text(base or "", encoding="utf-8")
    (out_dir / "tesc.json").write_text(tesc_txt or "", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({k: case[k] for k in ("id","title","focus","bug_patterns","fix_patterns")}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(ROOT / "bench_runs" / "programming_bench_samba"))
    args = ap.parse_args()

    client = get_client()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / "benchmarks" / "data" / "programming_cases.jsonl"
    cases = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for case in cases:
        run_case(client, run_dir / case["id"], case)
    print("Wrote run to:", run_dir)


if __name__ == "__main__":
    main()

