#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).parent.parent


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * ((p*(1-p)/n + z**2/(4*n**2)) ** 0.5)
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(xs: List[float], iters: int = 2000, seed: int = 42) -> Tuple[float, Tuple[float, float]]:
    import random
    if not xs:
        return (0.0, (0.0, 0.0))
    random.seed(seed)
    n = len(xs)
    means = []
    for _ in range(iters):
        sample = [xs[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]; hi = means[int(0.975 * iters)]
    return (sum(xs)/n, (lo, hi))


def programming_bench_ci(run_dir: Path, out_figs: Path):
    data = json.loads((run_dir / "summary" / "results.json").read_text(encoding="utf-8"))
    def arr(metric: str, side: str) -> List[float]:
        return [1.0 if item[side][metric] else 0.0 for item in data]
    cov_base = [item["baseline"]["coverage_score"] for item in data]
    cov_tesc = [item["tesc"]["coverage_score"] for item in data]
    pairs = [
        ("bug detection rate", arr("has_bug","baseline"), arr("has_bug","tesc")),
        ("fix suggestion rate", arr("has_fix","baseline"), arr("has_fix","tesc")),
        ("tests present rate", arr("tests_present","baseline"), arr("tests_present","tesc")),
        ("severity present rate", arr("severity_numeric_present","baseline"), arr("severity_numeric_present","tesc")),
    ]
    lines = ["% auto-generated programming bench CIs (simple)",
             "\\begin{table}[H]",
             "  \\centering",
             "  \\caption{Programming review (n=%d cases): rates/means with 95\\% CI.}" % (len(data)),
             "  \\label{tab:prog-bench-ci}",
             "  \\begin{tabular}{l cc}",
             "    Metric & Baseline & TESC \\",
             ]
    for name, xb, xt in pairs:
        kb, nb = int(sum(xb)), len(xb)
        kt, nt = int(sum(xt)), len(xt)
        lob, hib = wilson_ci(kb, nb); lot, hit = wilson_ci(kt, nt)
        lines.append(f"    {name} & {kb/nb:.2f} [{lob:.2f},{hib:.2f}] & {kt/nt:.2f} [{lot:.2f},{hit:.2f}] \\")
    mb, (lob, hib) = bootstrap_mean_ci(cov_base)
    mt, (lot, hit) = bootstrap_mean_ci(cov_tesc)
    lines.append(f"    coverage & {mb:.2f} [{lob:.2f},{hib:.2f}] & {mt:.2f} [{lot:.2f},{hit:.2f}] \\")
    lines += ["  \\end{tabular}", "\\end{table}", ""]
    (out_figs / "programming_bench_ci_table.tex").write_text("\n".join(lines), encoding="utf-8")


def programming_repair_ci(run_dir: Path, out_figs: Path):
    data = json.loads((run_dir / "summary" / "repair_results.json").read_text(encoding="utf-8"))
    base = [1.0 if r["baseline_pass"] else 0.0 for r in data]
    tesc = [1.0 if r["tesc_pass"] else 0.0 for r in data]
    kb, nb = int(sum(base)), len(base)
    kt, nt = int(sum(tesc)), len(tesc)
    lob, hib = wilson_ci(kb, nb)
    lot, hit = wilson_ci(kt, nt)
    lines = ["% auto-generated programming repair CIs (simple)",
             "\\begin{table}[H]",
             "  \\centering",
             "  \\caption{Programming repair pass rates (n=%d). 95\\% CI (Wilson).}" % (len(base)),
             "  \\label{tab:prog-repair-ci}",
             "  \\begin{tabular}{l cc}",
             "    Metric & Baseline & TESC \\",
             f"    Test pass rate & {kb/nb:.2f} [{lob:.2f},{hib:.2f}] & {kt/nt:.2f} [{lot:.2f},{hit:.2f}] \\",
             "  \\end{tabular}",
             "\\end{table}",
             ""]
    (out_figs / "programming_repair_ci_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog-bench", default=None)
    ap.add_argument("--prog-repair", default=None)
    ap.add_argument("--lab-run", default=None)
    ap.add_argument("--uncertainty-levels", default=None)
    args = ap.parse_args()
    figs = ROOT / 'paper' / 'figs'
    figs.mkdir(parents=True, exist_ok=True)
    if args.prog_bench:
        programming_bench_ci(Path(args.prog_bench), figs)
    if args.prog_repair:
        programming_repair_ci(Path(args.prog_repair), figs)
    print("CI tables written to", figs)


if __name__ == '__main__':
    main()

