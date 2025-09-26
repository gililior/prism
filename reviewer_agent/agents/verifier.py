from typing import List, Tuple

from reviewer_agent.llm.base import LLMClient
from reviewer_agent.llm.constants import TaskLLMConfigs


def verify(rebuttals: List[str], llm: LLMClient = None) -> List[Tuple[str, str]]:
    """Verify author rebuttals using LLM or fallback to toy implementation."""
    if llm is None:
        # Fallback to toy implementation
        return _toy_verify(rebuttals)

    verified = []
    for rebuttal in rebuttals:
        try:
            prompt = f"""Evaluate this author rebuttal for validity and evidence quality.
Classify as one of:
- OK: Valid response with specific evidence (section/figure/table references, concrete data, etc.)
- WEAK: Response lacks specific evidence or is too vague
- UNVERIFIED: Cannot verify claims or insufficient information

Rebuttal: {rebuttal}

Classification (just the category):"""

            config = TaskLLMConfigs.VERIFIER_CHECK
            response = llm.generate(prompt, temperature=config.temperature,
                                    max_tokens=config.max_tokens).strip().upper()

            # Normalize response
            if "OK" in response:
                status = "OK"
            elif "WEAK" in response:
                status = "WEAK"
            else:
                status = "UNVERIFIED"

            verified.append((status, rebuttal))

        except Exception as e:
            print(f"Error verifying rebuttal: {e}")
            # Fallback to toy verification
            verified.append(_toy_verify([rebuttal])[0])

    return verified


def _toy_verify(rebuttals: List[str]) -> List[Tuple[str, str]]:
    """Toy verifier ensures the rebuttal includes some 'see Sec'/'Table' hooks."""
    checked = []
    for r in rebuttals:
        if any(tag in r for tag in ["Sec", "Table", "Fig", "Appendix"]):
            checked.append(("OK", r))
        elif any(word in r.lower() for word in ["clarification", "misunderstanding", "context"]):
            checked.append(("WEAK", r))
        else:
            checked.append(("UNVERIFIED", r))
    return checked
