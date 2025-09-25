
from typing import List
from ..schemas import Point, Paper
from ..llm.base import LLMClient

def rebut(points: List[Point], paper: Paper = None, llm: LLMClient = None) -> List[str]:
    """Generate author rebuttals for review points using LLM or fallback to toy implementation."""
    if llm is None:
        # Fallback to toy implementation
        return _toy_rebut(points)
    
    rebuttals = []
    for point in points:
        try:
            # Create context from paper sections for better rebuttals
            paper_context = ""
            if paper:
                paper_context = "\n\n".join([f"## {s.name}\n{s.text[:500]}..." for s in paper.sections[:3]])
            
            prompt = f"""You are the author of this paper responding to a reviewer's point. 
Provide a professional rebuttal that either:
1. Addresses a misunderstanding by pointing to specific sections/figures/tables
2. Acknowledges a valid point and suggests improvements
3. Provides missing context or clarifications

Paper context (first 3 sections):
{paper_context}

Reviewer Point ({point.kind}): {point.text}
Grounding: {point.grounding or 'Not specified'}

Write a concise, professional rebuttal (2-3 sentences max):"""

            rebuttal = llm.generate(prompt, max_tokens=150)
            rebuttals.append(rebuttal.strip())
            
        except Exception as e:
            print(f"Error generating rebuttal for point: {e}")
            # Fallback to toy response
            rebuttals.append(_toy_rebut([point])[0])
    
    return rebuttals

def _toy_rebut(points: List[Point]) -> List[str]:
    """Toy rebuttal: flags points with labels based on keyword matches and returns evidence hooks."""
    outs = []
    for p in points:
        if "confidence interval" in p.text.lower():
            outs.append("MISUNDERSTANDING: CIs provided in Appendix B (see Table B.3).")
        elif "power" in p.text.lower():
            outs.append("VALID POINT: We will add a power analysis for the ablation.")
        elif "statistical" in p.text.lower() or "significance" in p.text.lower():
            outs.append("CLARIFICATION: Statistical analysis details are in Sec 4.2 and Appendix C.")
        elif "baseline" in p.text.lower() or "comparison" in p.text.lower():
            outs.append("MISSING CONTEXT: Additional baselines and comparisons are in Table 2 and Sec 4.4.")
        else:
            outs.append("MISSING CONTEXT: Related results in Sec 4.3 address this partially.")
    return outs
