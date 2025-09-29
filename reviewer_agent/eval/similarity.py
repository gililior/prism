import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore
from rapidfuzz import fuzz  # type: ignore
import torch

# Global model cache with thread safety
_model_cache = {}
_model_cache_lock = threading.Lock()

def _get_best_device() -> str:
    """Automatically detect the best device for computation"""
    import threading
    
    # For safety, only use GPU in main thread to avoid memory issues
    if threading.current_thread() is not threading.main_thread():
        return "cpu"
    
    try:
        if torch.backends.mps.is_available():
            return "mps"  # Mac GPU
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"   # CPU fallback
    except Exception:
        # Fallback to CPU if there are any issues
        return "cpu"

def _get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> SentenceTransformer:
    """Get SentenceTransformer model with caching to avoid reloading"""
    if device is None:
        device = _get_best_device()
    
    # Use device in cache key to avoid conflicts
    cache_key = f"{model_name}_{device}"
    
    with _model_cache_lock:
        if cache_key not in _model_cache:
            try:
                print(f"Loading SentenceTransformer model: {model_name} on device: {device}")
                model = SentenceTransformer(model_name, device=device)
                _model_cache[cache_key] = model
                print(f"✅ Model loaded successfully on {device}")
            except Exception as e:
                print(f"⚠️ Failed to load model on {device}: {e}")
                print(f"🔄 Falling back to CPU...")
                # Fallback to CPU if GPU fails
                device = "cpu"
                cache_key = f"{model_name}_{device}"
                if cache_key not in _model_cache:
                    model = SentenceTransformer(model_name, device=device)
                    _model_cache[cache_key] = model
                    print(f"✅ Model loaded successfully on {device}")
        return _model_cache[cache_key]

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


def _similarity_matrix(a_sents: List[str], b_sents: List[str], model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> List[List[float]]:
    model = _get_sentence_transformer(model_name, device)
    a_emb = model.encode(a_sents, convert_to_tensor=True, normalize_embeddings=True)
    b_emb = model.encode(b_sents, convert_to_tensor=True, normalize_embeddings=True)
    sims = st_util.cos_sim(a_emb, b_emb).cpu().tolist()  # each value in [-1, 1]
    # Map from [-1,1] to [0,1]
    return [[max(0.0, (v + 1.0) / 2.0) for v in row] for row in sims]


def sentence_level_similarity(pred_path: str, ref_path: str, embed_model: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> Dict[str, float]:
    pred_text = _read_review_text(pred_path)
    ref_text = _read_review_text(ref_path)
    return sentence_level_similarity_from_text(pred_text, ref_text, embed_model, device)

def sentence_level_similarity_from_text(pred_text: str, ref_text: str, embed_model: str = "all-MiniLM-L6-v2", device: Optional[str] = None) -> Dict[str, float]:
    """Calculate similarity directly from text without needing file paths"""
    pred_sents = _split_sentences(pred_text)
    ref_sents = _split_sentences(ref_text)
    if not pred_sents or not ref_sents:
        return {"mean_max_similarity": 0.0, "coverage": 0.0, "num_pred": len(pred_sents), "num_ref": len(ref_sents)}

    sims = _similarity_matrix(pred_sents, ref_sents, model_name=embed_model, device=device)

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


