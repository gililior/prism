
from typing import List
from ..schemas import Point, Review, Rubric, Paper
from ..llm.base import LLMClient
import json

def merge_points(points: List[Point], rubric: Rubric, llm: LLMClient = None, paper: Paper = None) -> Review:
    """Merge review points using LLM-powered synthesis or fallback to simple deduplication."""
    return _llm_merge_points(points, rubric, llm, paper)

def _simple_merge_points(points: List[Point], rubric: Rubric) -> Review:
    """Simple deduplication-based merging (original implementation)."""
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

def _llm_merge_points(points: List[Point], rubric: Rubric, llm: LLMClient, paper: Paper = None) -> Review:
    """Use LLM to intelligently merge and synthesize review points."""
    # Read the prompt template
    with open("reviewer_agent/prompts/leader_merge.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()

    # Separate points by kind
    strengths = [p for p in points if p.kind == "strength"]
    weaknesses = [p for p in points if p.kind == "weakness"]  
    suggestions = [p for p in points if p.kind == "suggestion"]
    
    # Add paper context if available
    paper_context = ""
    if paper:
        paper_context = f"Paper Title: {paper.title}"
    
    # Create simple point lists for the prompt
    strengths_text = "\n".join([f"- {p.text} ({p.grounding}) [{p.facet}]" for p in strengths])
    weaknesses_text = "\n".join([f"- {p.text} ({p.grounding}) [{p.facet}]" for p in weaknesses])
    suggestions_text = "\n".join([f"- {p.text} ({p.grounding}) [{p.facet}]" for p in suggestions])
    
    # Use the template with format substitution
    full_prompt = prompt_template.format(
        paper_context=paper_context,
        strengths_count=len(strengths),
        strengths_text=strengths_text,
        weaknesses_count=len(weaknesses),
        weaknesses_text=weaknesses_text,
        suggestions_count=len(suggestions),
        suggestions_text=suggestions_text
    )

    # Generate response
    response = llm.generate(full_prompt, max_tokens=10000, temperature=0.1)
    parsed = json.loads(response)

    # Convert back to Point objects
    merged_strengths = [Point(kind="strength", **p) for p in parsed.get("strengths", [])]
    merged_weaknesses = [Point(kind="weakness", **p) for p in parsed.get("weaknesses", [])]
    merged_suggestions = [Point(kind="suggestion", **p) for p in parsed.get("suggestions", [])]

    return Review(
        summary=parsed["summary"],
        strengths=merged_strengths,
        weaknesses=merged_weaknesses,
        suggestions=merged_suggestions,
        scores=None
    )


def enforce_grounding(review: Review) -> Review:
    def grounded(pt: Point) -> bool:
        return bool(pt.grounding and len(pt.grounding.strip()) > 0)
    review.strengths = [p for p in review.strengths if grounded(p)]
    review.weaknesses = [p for p in review.weaknesses if grounded(p)]
    review.suggestions = [p for p in review.suggestions if grounded(p)]
    return review

def revise_review(review: Review, rebuttals: List[str], verifications: List[tuple]) -> Review:
    """Revise the review based on author rebuttals and verifier outcomes.

    Enhanced logic:
    - OK rebuttals: Soften weaknesses to suggestions if they address misunderstandings, add grounding if missing
    - WEAK rebuttals: Keep original points but add note about author response
    - UNVERIFIED rebuttals: Keep original points unchanged
    """
    # Map verifier outcomes back to rebuttal texts
    status_by_rebuttal = {text: status for status, text in verifications}

    def process(points: List[Point]) -> List[Point]:
        revised: List[Point] = []
        for p in points:
            matched_reb = None
            for r in rebuttals:
                # Improved matching: use multiple approaches
                key_words = p.text.lower().split(" ")[0:4]
                key_phrase = " ".join(key_words)
                
                # Try exact phrase match first, then individual words
                if key_phrase and key_phrase in r.lower():
                    matched_reb = r
                    break
                elif any(word in r.lower() for word in key_words if len(word) > 3):
                    matched_reb = r
                    break
                    
            if not matched_reb:
                revised.append(p)
                continue
                
            status = status_by_rebuttal.get(matched_reb, "UNVERIFIED")
            reb_lower = matched_reb.lower()
            
            if status == "OK":
                # Handle OK rebuttals
                if ("misunderstanding" in reb_lower or "missing context" in reb_lower or "clarification" in reb_lower):
                    # Soften weakness to suggestion
                    if p.kind == "weakness":
                        revised.append(Point(
                            kind="suggestion", 
                            text=f"Consider clarifying: {p.text}", 
                            grounding=p.grounding, 
                            facet=p.facet
                        ))
                    else:
                        revised.append(p)
                elif any(tag in matched_reb for tag in ["Sec", "Table", "Fig", "Appendix"]):
                    # Add or update grounding from rebuttal
                    new_grounding = p.grounding
                    if not new_grounding:
                        new_grounding = f"Author response: {matched_reb[:100]}..."
                    else:
                        new_grounding += f" (Author clarified: {matched_reb[:50]}...)"
                    
                    revised.append(Point(
                        kind=p.kind, 
                        text=p.text, 
                        grounding=new_grounding, 
                        facet=p.facet
                    ))
                else:
                    # Valid rebuttal but no specific action needed
                    revised.append(p)
                    
            elif status == "WEAK":
                # Keep point but note the weak rebuttal
                new_text = f"{p.text} (Author response noted but lacks specificity)"
                revised.append(Point(
                    kind=p.kind,
                    text=new_text,
                    grounding=p.grounding,
                    facet=p.facet
                ))
                
            else:  # UNVERIFIED
                # Keep original point unchanged
                revised.append(p)
                
        return revised

    review.weaknesses = process(review.weaknesses)
    review.suggestions = process(review.suggestions)
    # Strengths unchanged in MVP
    return review
