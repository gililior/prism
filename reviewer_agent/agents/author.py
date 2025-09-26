
import json
from typing import List
from ..schemas import Point, Paper
from ..llm.base import LLMClient

def rebut(points: List[Point], paper: Paper = None, llm: LLMClient = None) -> str:
    """Generate a comprehensive author rebuttal addressing all review points."""
    # Read the prompt template
    with open("reviewer_agent/prompts/author_rebuttal.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Create context from paper sections
    paper_context = ""
    paper_title = ""
    if paper:
        paper_title = paper.title
        paper_context = "\n\n".join([f"## {s.name}\n{s.text[:500]}..." for s in paper.sections[:3]])
    
    # Organize points by type
    weaknesses = [p for p in points if p.kind == "weakness"]
    suggestions = [p for p in points if p.kind == "suggestion"]
    
    # Format points for the prompt
    weaknesses_text = "\n".join([f"- {p.text} ({p.grounding})" for p in weaknesses])
    suggestions_text = "\n".join([f"- {p.text} ({p.grounding})" for p in suggestions])
    
    # Use the template with format substitution
    prompt = prompt_template.format(
        paper_title=paper_title,
        paper_context=paper_context,
        weaknesses_text=weaknesses_text,
        suggestions_text=suggestions_text,
        weaknesses_count=len(weaknesses),
        suggestions_count=len(suggestions)
    )

    rebuttal = llm.generate(prompt, max_tokens=3000)
    return rebuttal.strip()