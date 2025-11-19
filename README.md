# TESC: Deterministic Cognitive State Control in LLMs


[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

AMAWTA Research — Daslav Ríos Montecinos, Oscar Ríos Saldivar  
daslav@amawtalabs.com · oscar@amawtalabs.com  
 https://amawtalabs.com

---

## Abstract
We present empirical validation of the Theorem of Semiotic–Cognitive Equivalence (TESC), demonstrating deterministic cognitive state control in LLMs through structured outputs and semiotic configuration.

Headline results (current lab + benches):
- JSON validity via vendor structured outputs (`response_schema`); incremental gains with TESC presets: coverage/severity = 1.00, stable behavior, and clear mode separability.
- Injectivity plateau 1.0 for thresholds ≥ 0.98
- Semiotic Uncertainty: Δs·Δc ≥ ℏ_sem (ℏ_sem ≈ 2×10⁻⁵) always satisfied
- Dynamics: deterministic fraction ≈ 0.53 and SNR ≈ 14 dB (linear, held-out)
- Benchmarks: programming pitfalls (review/repair) and audit contracts (ES)

---

## ⏩ TL;DR for devs (utility first)

- We use vendor structured outputs (`response_schema`) as a contract. With semiotic presets (system instruction, style markers, bounded temperature), behavior is stable, mode‑separable, and partly predictable — and contracts become machine‑actionable (coverage/severity = 1.00) with external evaluation to avoid circularity.
- Same schema, consistent fields every time → CI‑ready tables and tests.
- Fewer flips under small wording/temperature changes (stability).
- Modes remain distinct (injectivity plateau).
- For function‑calling: complete arguments and near‑zero extra keys.

Scope note: JSON validity itself is provided by the model’s structured outputs; TESC focuses on the incremental gains above free text (coverage/completeness, stability/robustness, separability, and tool‑calling quality).

### Minimal example

Schema (review JSON):
```json
{
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
```

System instruction (prompt snippet):
```
You are a senior code reviewer. Provide actionable findings, risks,
a patch outline, tests, and a numeric severity in [0,1].
Return JSON with fields: issues[], risks[], patch_outline[], tests[], severity.
```

Sample output:
```json
{
  "review": {
    "issues": ["Mutable default argument on parameter 'b'"],
    "risks": ["State leakage across calls"],
    "patch_outline": ["Use None and initialize list inside the function"],
    "tests": ["Call twice; lists must not alias"],
    "severity": 0.9
  }
}
```

### Two commands to reproduce (review bench)
```
poetry run python benchmarks/run_programming_bench.py
poetry run python benchmarks/eval_programming_bench.py bench_runs/programming_bench/<TIMESTAMP>
```

### Baseline vs Semiotic (what changes)

| Metric                            | Baseline (free text) | Semiotic (schema) |
|-----------------------------------|----------------------|-------------------|
| Required‑field coverage           | ~0.55                | 1.00              |
| Severity present                  | 0%                   | 100%              |
| Extra keys rate (tool‑calling)    | n/a                  | 0.0%              |
| Arg coverage (tool‑calling)       | n/a                  | 1.00              |

Numbers above are from the included micro‑benchmarks (n=8) and the SambaNova function‑calling runs.

## Key Results at a Glance

| Metric                       | Baseline (Free text) | TESC (Structured)        |
|-----------------------------|----------------------|--------------------------|
| Output form                  | Free text            | JSON (valid via `response_schema`) |
| Audit contract validity      | 0%                   | 100%                     |
| Mean field coverage (audit)  | 0.02                 | 1.00                     |
| Injectivity plateau (θ≥0.98) | —                    | 1.0                      |
| Intra-config similarity      | —                    | 0.962                    |
| Programming coverage (CI)    | 0.53 [0.43,0.62]     | 0.88 [0.62,1.00]         |
| Repair pass rate (CI)        | 0.75 [0.41,0.93]     | 0.88 [0.53,0.98]         |

---

## Methods Overview
- Semiotic Configuration (S = ⟨instruction, schema, markers, temperature⟩)
- Generation via Gemini 2.5 Flash with enforced JSON schemas
- Evaluation with external embeddings (Qwen3 0.6B / ST) to avoid circularity
- Metrics: injectivity, uncertainty (Δs·Δc), dynamics (dc/dt = f(s,c)+η), robustness

---

## Benchmarks (this repo)

### Programming Pitfalls (n=8 canonical cases)
- Mutable default args; Off-by-one; Resource leaks; Bare excepts; Shallow copy; SQL injection; Path traversal; Float equality
- TESC improves coverage & actionability (issues/risks/patch/tests/severity) and repair correctness (pass rates)

### Semiotic Uncertainty (pairwise; 4-level regimes)
- Pairwise Δs·Δc against lab runs, and 4 regimes (ultra-precise/precise/vague/ultra-vague) for EN/ES prompts

### Dynamics (held-out)
- Linear fit of c_{t+1} from (c_t, s_t) with intra-config perturbations; report R² per dimension, RMSE, deterministic fraction

---

## Quickstart

1) Install dependencies (Poetry)
```
poetry install
```

2) Set your API key
```
# copy .env.example to .env and set GEMINI_API_KEY, or export it
export GEMINI_API_KEY="<your_key>"
```

3) Programming benchmarks
```
# Review (baseline vs TESC) → results.json + CI table
poetry run python benchmarks/run_programming_bench.py
poetry run python benchmarks/eval_programming_bench.py bench_runs/programming_bench/<TIMESTAMP>
poetry run python benchmarks/make_ci_tables.py --prog-bench bench_runs/programming_bench/<TIMESTAMP>

# Repair (baseline vs TESC) → repair_results.json + CI table
poetry run python benchmarks/run_programming_repair.py
poetry run python benchmarks/eval_programming_repair.py bench_runs/programming_repair/<TIMESTAMP>
poetry run python benchmarks/make_ci_tables.py --prog-repair bench_runs/programming_repair/<TIMESTAMP>
```

4) Uncertainty & Dynamics
```
# 4-level regimes (EN/ES)
poetry run python benchmarks/run_uncertainty_levels.py --per-level 6 --lang en
poetry run python benchmarks/run_uncertainty_levels.py --per-level 6 --lang es

# Pairwise uncertainty & dynamics on your lab run folder
poetry run python benchmarks/tesc_uncertainty_eval.py lab_runs/<RUN_ID>
poetry run python benchmarks/tesc_dynamics_eval.py lab_runs/<RUN_ID>
```

---

## Repository Structure
```
.
├── benchmarks/
│   ├── data/programming_cases.jsonl
│   ├── semio_utils.py                # reads GEMINI_API_KEY
│   ├── sambanova_utils.py            # reads SAMBA_API_KEY (optional)
│   ├── run_programming_bench.py      # review generation
│   ├── eval_programming_bench.py     # review evaluation
│   ├── run_programming_repair.py     # repair generation
│   ├── eval_programming_repair.py    # repair evaluation
│   ├── run_uncertainty_levels.py     # 4-level regimes (EN/ES)
│   ├── tesc_uncertainty_eval.py      # pairwise Δs·Δc on lab run
│   ├── tesc_dynamics_eval.py         # dynamics fit on lab run
│   └── make_ci_tables.py             # CI LaTeX tables (optional)
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md (this)
```

---

## Citation
If you use this work, please cite (preprint, technical report):
```bibtex
@techreport{rios2025tesc,
  title        = {TESC: Deterministic Cognitive State Control in LLMs via Structured Outputs and Semiotic Configuration},
  author       = {Ríos Montecinos, Daslav and Ríos Saldivar, Oscar},
  institution  = {AMAWTA Research},
  number       = {TESC-TR-2025-09},
  type         = {Technical Report},
  year         = {2025},
  month        = {September},
  url          = {https://github.com/Amawta-labs/TESC},
  note         = {Public code and benchmarks; preprint forthcoming}
}
```

## License
Apache License 2.0. See [LICENSE](./LICENSE) for details.

## MAWTA Research
A father, a son, and an AI partner exploring deterministic cognition in LLMs.  
Independent lab · Santiago, Chile · 2025

## Acknowledgments

We thank:
- [Thunder Compute](https://www.thundercompute.com) for one‑click GPU instances that accelerated our experiments.
- [SambaNova Systems](https://www.sambanova.ai) for an OpenAI‑compatible function‑calling interface and developer support that enabled cross‑model structured‑output validation.
- [OpenAI](https://openai.com) for tools and research infrastructure that supported parts of this work.

Trademarks and names remain the property of their respective owners. This acknowledgment does not imply endorsement or partnership.
# 5) (Optional) Use SambaNova models (function calling)
```
export SAMBA_API_KEY="<your_sambanova_key>"
# Optionally: export SAMBA_MODEL=Meta-Llama-3.1-70B-Instruct

# Programming review with SambaNova
poetry run python benchmarks/run_programming_bench_sambanova.py
# Evaluate with the same eval script
poetry run python benchmarks/eval_programming_bench.py bench_runs/programming_bench_samba/<TIMESTAMP>
```
