#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

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


def load_run(run_dir: Path) -> List[Dict[str, Any]]:
    raw = run_dir / "raw"
    samples = []
    for fp in sorted(raw.glob("sample_*.json")):
        try:
            samples.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return samples


def compute_c_vectors(samples: List[Dict[str, Any]], emb_model) -> Dict[int, np.ndarray]:
    prot_texts = [PROTOTYPES[k] for k in ("analytical", "creative", "critical")]
    prot_vecs = embed_texts(emb_model, prot_texts)
    out: Dict[int, np.ndarray] = {}
    texts = [s.get("response_text") or "" for s in samples]
    X = embed_texts(emb_model, texts)
    for i, s in enumerate(samples):
        sims = np.array([cos_sim(X[i], prot_vecs[j]) for j in range(3)], dtype=float)
        c = softmax(sims, tau=0.2)
        out[i] = c
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def delta_s(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    d_temp = abs(float(a.get("temperature", 0.0)) - float(b.get("temperature", 0.0)))
    d_style = 0.0 if (a.get("style") == b.get("style")) else 1.0
    d_markers = 1.0 - jaccard(a.get("markers") or [], b.get("markers") or [])
    return 0.5 * d_temp + 0.25 * d_style + 0.25 * d_markers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="lab_runs/<id>")
    ap.add_argument("--h-sem", type=float, default=2e-5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    samples = load_run(run_dir)
    if not samples:
        raise SystemExit("No samples found.")
    emb_model = try_load_embedder()
    c_map = compute_c_vectors(samples, emb_model)

    def base_sig(s: Dict[str, Any]) -> str:
        sig = {"base_id": s.get("base_id"), "prompt": s.get("prompt")}
        return hashlib.md5(json.dumps(sig, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[base_sig(s)].append(i)

    prods = []
    for _, idxs in groups.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a = samples[idxs[i]]; b = samples[idxs[j]]
                ds = delta_s(a, b)
                if ds <= 0: continue
                ca = c_map[idxs[i]]; cb = c_map[idxs[j]]
                dc = float(np.linalg.norm(ca - cb))
                prods.append(ds * dc)

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uncertainty_products.json").write_text(json.dumps({"products": prods}, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(prods), "min": min(prods) if prods else 0.0, "median": float(np.median(prods) if prods else 0.0)}, indent=2))


if __name__ == '__main__':
    main()

