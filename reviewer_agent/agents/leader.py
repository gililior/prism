
from typing import List
from ..schemas import Point, Review, Rubric

def merge_points(points: List[Point], rubric: Rubric) -> Review:
    strengths = []
    weaknesses = []
    suggestions = []
    seen = set()
    for p in points:
        key = (p.kind, p.text.lower())
        if key in seen:
            continue
        seen.add(key)
        if p.kind == "strength":
            strengths.append(p)
        elif p.kind == "weakness":
            weaknesses.append(p)
        elif p.kind == "suggestion":
            suggestions.append(p)
    summary = "This review aggregates facet-specialist feedback. Strengths include clear methods and positioning; weaknesses center on statistical rigor and comparative evidence."
    return Review(summary=summary, strengths=strengths, weaknesses=weaknesses, suggestions=suggestions, scores=None)

def enforce_grounding(review: Review) -> Review:
    def grounded(pt: Point) -> bool:
        return bool(pt.grounding and len(pt.grounding.strip()) > 0)
    review.strengths = [p for p in review.strengths if grounded(p)]
    review.weaknesses = [p for p in review.weaknesses if grounded(p)]
    review.suggestions = [p for p in review.suggestions if grounded(p)]
    return review

def revise_review(review: Review, rebuttals: List[str], verifications: List[tuple]) -> Review:
    """Revise the review based on author rebuttals and verifier outcomes.

    Heuristic MVP:
    - If a rebuttal is marked OK by verifier and indicates MISUNDERSTANDING or MISSING CONTEXT, soften or remove the corresponding weakness/suggestion.
    - If rebuttal provides a concrete citation (Sec/Fig/Table), append it to grounding when missing.
    - If rebuttal is UNVERIFIED, keep original point unchanged.
    """
    # Map verifier outcomes back to rebuttal texts
    status_by_rebuttal = {text: status for status, text in verifications}

    def process(points: List[Point]) -> List[Point]:
        revised: List[Point] = []
        for p in points:
            matched_reb = None
            for r in rebuttals:
                # Simple heuristic match by keyword overlap with point text fragment
                key = p.text.lower().split(" ")[0:4]
                key = " ".join(key)
                if key and key in r.lower():
                    matched_reb = r
                    break
            if not matched_reb:
                revised.append(p)
                continue
            status = status_by_rebuttal.get(matched_reb, "UNVERIFIED")
            reb_lower = matched_reb.lower()
            if status == "OK" and ("misunderstanding" in reb_lower or "missing context" in reb_lower):
                # Soften or convert to suggestion
                if p.kind == "weakness":
                    revised.append(Point(kind="suggestion", text=f"Clarify: {p.text}", grounding=p.grounding, facet=p.facet))
                else:
                    revised.append(p)
            elif status == "OK" and any(tag in matched_reb for tag in ["Sec", "Table", "Fig"]):
                # Add grounding if missing
                if not p.grounding:
                    revised.append(Point(kind=p.kind, text=p.text, grounding="From rebuttal: " + matched_reb, facet=p.facet))
                else:
                    revised.append(p)
            else:
                revised.append(p)
        return revised

    review.weaknesses = process(review.weaknesses)
    review.suggestions = process(review.suggestions)
    # Strengths unchanged in MVP
    return review
