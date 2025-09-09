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


PROTOTYPES = {
    "analytical": "Análisis lógico, premisas, pasos deductivos, conclusión rigurosa, formalidad.",
    "creative": "Imaginación, metáforas, asociaciones libres, síntesis novedosa, expresividad.",
    "critical": "Escepticismo, cuestionamiento de supuestos, evaluación de debilidades, objeciones.",
}


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


def build_transitions(samples: List[Dict[str, Any]], c_map: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    def key(s: Dict[str, Any]) -> str:
        return hashlib.md5(json.dumps({"base_id": s.get("base_id"), "prompt": s.get("prompt")}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[key(s)].append(i)
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    for _, idxs in groups.items():
        by_var: Dict[str, int] = {}
        for idx in idxs:
            var = samples[idx].get("variant", "base")
            if var not in by_var:
                by_var[var] = idx
        base_idx = by_var.get("base")
        if base_idx is None:
            continue
        base_s = samples[base_idx]
        base_c = c_map[base_idx]
        base_temp = float(base_s.get("temperature", 0.0))
        base_mlen = float(len(base_s.get("markers") or []))
        for vtag in ("temp+", "temp-", "no_markers"):
            j = by_var.get(vtag)
            if j is None:
                continue
            s2 = samples[j]
            c2 = c_map[j]
            temp2 = float(s2.get("temperature", 0.0))
            dv_temp = temp2 - base_temp
            mlen2 = float(len(s2.get("markers") or []))
            dv_mlen = mlen2 - base_mlen
            feat = np.array([*base_c.tolist(), base_temp, abs(dv_temp), base_mlen, dv_mlen], dtype=float)
            Xs.append(feat)
            Ys.append(c2)
    if not Xs:
        return np.zeros((0,)), np.zeros((0,))
    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)
    return X, Y


def fit_linear(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    k = max(1, int(0.8 * n))
    tr, te = idx[:k], idx[k:]
    Xtr = np.c_[X[tr], np.ones((len(tr), 1))]
    Xte = np.c_[X[te], np.ones((len(te), 1))]
    Ytr, Yte = Y[tr], Y[te]
    W, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    Yhat = Xte @ W
    r2 = {}
    rmse = {}
    for d in range(Y.shape[1]):
        y = Yte[:, d]
        yhat = Yhat[:, d]
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-9)
        r2[f"dim{d}"] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse[f"dim{d}"] = float(np.sqrt(np.mean((y - yhat) ** 2)))
    det_frac = float(np.mean(list(r2.values())))
    return det_frac, {**r2, **{f"rmse_{k}": v for k, v in rmse.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="lab_runs/<id>")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    samples = load_run(run_dir)
    if not samples:
        raise SystemExit("No samples found.")
    emb_model = try_load_embedder()
    c_map = compute_c_vectors(samples, emb_model)
    X, Y = build_transitions(samples, c_map)
    if X.size == 0:
        raise SystemExit("No transitions built.")
    det_frac, metrics = fit_linear(X, Y)
    out_dir = run_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_transitions": int(X.shape[0]), "deterministic_fraction": det_frac, "metrics": metrics}
    (out_dir / 'dynamics_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

