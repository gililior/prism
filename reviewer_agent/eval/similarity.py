import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

try:
    from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore
except Exception:  # optional dependency
    SentenceTransformer = None
    st_util = None

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # optional dependency
    fuzz = None


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter; keeps non-empty trimmed sentences
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 0]


def _read_review_text(path: str) -> str:
    txt = Path(path).read_text(encoding="utf-8")
    # If JSON, attempt to extract fields
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and any(k in obj for k in ["summary","strengths","weaknesses","suggestions"]):
            parts: List[str] = []
            if obj.get("summary"):
                parts.append(str(obj.get("summary", "")))
            for k in ["strengths","weaknesses","suggestions"]:
                if k in obj and isinstance(obj[k], list):
                    for it in obj[k]:
                        if isinstance(it, dict):
                            s = it.get("text", "")
                        else:
                            s = str(it)
                        if s:
                            parts.append(s)
            return "\n".join(parts)
    except Exception:
        pass
    return txt


def _similarity_matrix(a_sents: List[str], b_sents: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    if SentenceTransformer is not None:
        model = SentenceTransformer(model_name)
        a_emb = model.encode(a_sents, convert_to_tensor=True, normalize_embeddings=True)
        b_emb = model.encode(b_sents, convert_to_tensor=True, normalize_embeddings=True)
        sims = st_util.cos_sim(a_emb, b_emb).cpu().tolist()  # each value in [-1, 1]
        # Map from [-1,1] to [0,1]
        return [[max(0.0, (v + 1.0) / 2.0) for v in row] for row in sims]
    # Fallback to fuzzy string ratio scaled to [0,1]
    sims: List[List[float]] = []
    for a in a_sents:
        row: List[float] = []
        for b in b_sents:
            if fuzz is None:
                score = 0.0
            else:
                score = fuzz.token_set_ratio(a, b) / 100.0
            row.append(score)
        sims.append(row)
    return sims


def sentence_level_similarity(pred_path: str, ref_path: str, embed_model: str = "all-MiniLM-L6-v2") -> Dict[str, float]:
    pred_text = _read_review_text(pred_path)
    ref_text = _read_review_text(ref_path)
    pred_sents = _split_sentences(pred_text)
    ref_sents = _split_sentences(ref_text)
    if not pred_sents or not ref_sents:
        return {"mean_max_similarity": 0.0, "coverage": 0.0, "num_pred": len(pred_sents), "num_ref": len(ref_sents)}

    sims = _similarity_matrix(pred_sents, ref_sents, model_name=embed_model)

    # For each predicted sentence: max similarity over reference sentences
    max_vals: List[float] = []
    matched_ref_indices: set = set()
    for i, row in enumerate(sims):
        if not row:
            continue
        max_val = max(row)
        max_vals.append(max_val)
        # get argmax index (first one on ties)
        j = row.index(max_val)
        matched_ref_indices.add(j)

    mean_max = sum(max_vals) / max(1, len(max_vals))
    coverage = len(matched_ref_indices) / max(1, len(ref_sents))
    return {
        "mean_max_similarity": round(mean_max, 4),
        "coverage": round(coverage, 4),
        "num_pred": len(pred_sents),
        "num_ref": len(ref_sents),
    }


