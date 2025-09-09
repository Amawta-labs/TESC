#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any

from benchmarks.programming_tests import run as run_test


ROOT = Path(__file__).parent.parent


def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def load_tesc_code(p: Path) -> str:
    if not p.exists():
        return ""
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        return ((obj.get('repair') or {}).get('patched_code') or '').strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to run under bench_runs/programming_repair/<ts>")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    cases = [d for d in rd.iterdir() if d.is_dir() and (d / 'baseline.txt').exists()]
    results = []
    for cd in sorted(cases):
        cid = cd.name
        base_code = extract_code_block((cd / 'baseline.txt').read_text(encoding='utf-8'))
        tesc_code = load_tesc_code(cd / 'tesc.json')
        base_pass = run_test(cid, base_code) if base_code else False
        tesc_pass = run_test(cid, tesc_code) if tesc_code else False
        results.append({"id": cid, "baseline_pass": base_pass, "tesc_pass": tesc_pass})
    out = rd / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'repair_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print("Wrote:", out / 'repair_results.json')


if __name__ == '__main__':
    main()

