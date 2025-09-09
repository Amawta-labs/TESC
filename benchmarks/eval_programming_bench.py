#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any


ROOT = Path(__file__).parent.parent


def evaluate_baseline(txt: str, bug_patterns, fix_patterns) -> Dict[str, Any]:
    low = txt.lower()
    def has(patterns):
        return any(re.search(p, low, flags=re.I) for p in patterns)
    return {
        "has_bug": has(bug_patterns),
        "has_fix": has(fix_patterns),
        "tests_present": has([r"def\s+test_", r"pytest", r"assert "]),
        "patch_outline_present": has([r"the\s+fix", r"corrected\s+code", r"patch\s+outline"]),
        "severity_numeric_present": has([r"severity\s*[:=]\s*(0(\.\d+)?|1(\.0+)?)"]),
        "issues_present": has([r"bug", r"issue", r"problem", r"risk"]),
        "risks_present": has([r"risk", r"side\s*effect", r"security", r"leak", r"injection"]),
        "coverage_score": 0.0,
    }


def evaluate_tesc(obj: Dict[str, Any], bug_patterns, fix_patterns) -> Dict[str, Any]:
    review = (obj or {}).get("review") or {}
    issues = review.get("issues") or []
    risks = review.get("risks") or []
    patch = review.get("patch_outline") or []
    tests = review.get("tests") or []
    sev = review.get("severity")
    def list_has(patterns, items):
        low_items = "\n".join(items).lower() if isinstance(items, list) else str(items).lower()
        return any(re.search(p, low_items, flags=re.I) for p in patterns)
    has_bug = list_has(bug_patterns, issues)
    has_fix = bool(patch) or list_has(fix_patterns, patch)
    tests_present = len(tests) > 0
    patch_outline_present = len(patch) > 0
    severity_numeric_present = isinstance(sev, (int, float))
    issues_present = len(issues) > 0
    risks_present = len(risks) > 0
    cov_fields = [issues_present, risks_present, patch_outline_present, tests_present, severity_numeric_present]
    coverage_score = sum(1 for x in cov_fields if x) / 5.0
    return {
        "has_bug": has_bug,
        "has_fix": has_fix,
        "tests_present": tests_present,
        "patch_outline_present": patch_outline_present,
        "severity_numeric_present": severity_numeric_present,
        "issues_present": issues_present,
        "risks_present": risks_present,
        "coverage_score": coverage_score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to a run directory under bench_runs/programming_bench/<ts>")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    data = []
    cases = [d for d in rd.iterdir() if d.is_dir() and (d / 'baseline.txt').exists()]
    for cd in sorted(cases):
        meta = json.loads((cd / 'meta.json').read_text(encoding='utf-8'))
        base = (cd / 'baseline.txt').read_text(encoding='utf-8')
        try:
            tesc = json.loads((cd / 'tesc.json').read_text(encoding='utf-8'))
        except Exception:
            tesc = {}
        item = {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "baseline": evaluate_baseline(base, meta.get("bug_patterns") or [], meta.get("fix_patterns") or []),
            "tesc": evaluate_tesc(tesc, meta.get("bug_patterns") or [], meta.get("fix_patterns") or []),
        }
        data.append(item)
    out = rd / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print("Wrote:", out / 'results.json')


if __name__ == '__main__':
    main()

