from typing import List

from reviewer_agent.llm.base import LLMClient
from reviewer_agent.llm.constants import TaskLLMConfigs
from reviewer_agent.schemas import Point, Paper


def rebut(points: List[Point], paper: Paper = None, llm: LLMClient = None) -> str:
    """Generate a comprehensive author rebuttal addressing all review points."""
    # Read the prompt template
    # Get the path relative to this file using pathlib
    from pathlib import Path
    prompt_path = Path(__file__).parent.parent / "prompts" / "author_rebuttal.txt"
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Create context from paper sections
    paper_context = ""
    paper_title = ""
    if paper:
        paper_title = paper.title
        paper_context = "\n\n".join([f"## {s.name}\n{s.text}..." for s in paper.sections])

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

    config = TaskLLMConfigs.AUTHOR_REBUTTAL
    rebuttal = llm.generate(prompt, temperature=config.temperature, max_tokens=config.max_tokens)
    return rebuttal.strip()
