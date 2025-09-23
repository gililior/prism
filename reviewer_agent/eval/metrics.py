
from rapidfuzz import fuzz

def coverage_overlap(pred_points: list, gold_text: str) -> float:
    """Heuristic: compute max token-sort ratio across predicted points vs gold text."""
    if not gold_text or not pred_points:
        return 0.0
    scores = []
    for p in pred_points:
        scores.append(fuzz.token_sort_ratio(p.lower(), gold_text.lower()))
    return max(scores) if scores else 0.0

def genericity_rate(points: list) -> float:
    """Share of points that look generic (very short or keywordy)."""
    if not points: return 1.0
    generic = 0
    for p in points:
        t = p.strip().lower()
        if len(t) < 40 or any(k in t for k in ["interesting", "well-written", "nice"]):
            generic += 1
    return generic / len(points)
