#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from benchmarks.semio_utils import get_gemini
import numpy as np


ROOT = Path(__file__).parent.parent


def try_load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        m.name = "st:all-MiniLM-L6-v2"
        return m
    except Exception:
        return None


def embed_texts(model, texts: List[str]) -> np.ndarray:
    if model is None:
        return np.zeros((len(texts), 3), dtype=float)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


PROTOTYPES = {
    "analytical": "Análisis lógico, premisas, pasos deductivos, conclusión rigurosa, formalidad.",
    "creative": "Imaginación, metáforas, asociaciones libres, síntesis novedosa, expresividad.",
    "critical": "Escepticismo, cuestionamiento de supuestos, evaluación de debilidades, objeciones.",
}


def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float((u @ v) / (nu * nv))


def softmax(x: np.ndarray, tau: float = 0.2) -> np.ndarray:
    z = (x - x.max()) / max(1e-6, tau)
    e = np.exp(z)
    return e / max(1e-9, e.sum())


def cognitive_vectors(texts: List[str], model) -> np.ndarray:
    prot_texts = [PROTOTYPES[k] for k in ("analytical", "creative", "critical")]
    P = embed_texts(model, prot_texts)
    X = embed_texts(model, texts)
    Cs = []
    for i in range(len(texts)):
        sims = np.array([cos_sim(X[i], P[j]) for j in range(3)], dtype=float)
        Cs.append(softmax(sims, tau=0.2))
    return np.stack(Cs, axis=0)


def median_pairwise_l2(C: np.ndarray) -> float:
    n = C.shape[0]
    if n < 2:
        return 0.0
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            ds.append(float(np.linalg.norm(C[i] - C[j])))
    ds = np.array(ds, dtype=float)
    return float(np.median(ds))


def delta_s_level(schema: bool, temperature: float) -> float:
    return float(min(1.0, max(0.01, 0.05 + 0.6 * float(temperature) + 0.35 * (0.0 if schema else 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--lang", default="en", choices=["en","es"])
    args = ap.parse_args()

    client, _ = get_gemini()
    from google.genai import types

    if args.lang == "es":
        prompts = [
            "Explica la conciencia artificial en un párrafo.",
            "Resume los riesgos de SQL injection y cómo prevenirlos.",
            "Enumera los pasos de la inferencia causal en términos simples.",
        ]
    else:
        prompts = [
            "Explain artificial consciousness in one paragraph.",
            "Summarize the risks of SQL injection and how to prevent it.",
            "Outline the steps of causal inference in simple terms.",
        ]

    levels = [
        (True, 0.1, "ultra_precise"),
        (True, 0.3, "precise"),
        (False, 0.7, "vague"),
        (False, 1.0, "ultra_vague"),
    ]

    outputs = {name: [] for _,_,name in levels}
    for schema, temp, name in levels:
        for prompt in prompts:
            for _ in range(max(1, args.per_level // len(prompts))):
                if schema:
                    schema_obj = {"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}
                    cfg = types.GenerateContentConfig(
                        system_instruction=("You are a rigorous analytical system." if args.lang=="en" else "Eres un sistema analítico riguroso."),
                        response_mime_type="application/json",
                        response_schema=schema_obj,
                        temperature=temp,
                    )
                else:
                    cfg = types.GenerateContentConfig(
                        system_instruction=("You are a general assistant." if args.lang=="en" else "Eres un asistente general."),
                        temperature=temp,
                    )
                try:
                    resp = client.models.generate_content(model=args.model, contents=prompt, config=cfg)
                    outputs[name].append(getattr(resp, "text", "") or "")
                except Exception:
                    outputs[name].append("")

    emb = try_load_embedder()
    summary = []
    for schema, temp, name in levels:
        texts = outputs[name]
        C = cognitive_vectors(texts, emb)
        dc = median_pairwise_l2(C)
        ds = delta_s_level(schema, temp)
        summary.append({"level": name, "schema": schema, "temperature": temp, "delta_s": ds, "delta_c": dc, "product": ds * dc})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "bench_runs" / "uncertainty_levels" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"levels": summary, "lang": args.lang, "model": args.model, "n_per_level": args.per_level}, indent=2), encoding="utf-8")
    print("Wrote:", out_dir / "summary.json")


if __name__ == '__main__':
    main()

