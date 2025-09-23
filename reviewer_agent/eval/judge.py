import json
from typing import Dict
from pathlib import Path

from ..llm.base import LLMClient

PROMPT = (Path(__file__).parents[1] / "prompts" / "judge_compare.txt").read_text(encoding="utf-8")


def judge_compare(paper_context: str, review_a: str, review_b: str, model_name: str = "dummy") -> Dict:
    client = LLMClient(model_name=model_name)
    prompt = PROMPT.format(paper_context=paper_context[:6000], review_a=review_a[:6000], review_b=review_b[:6000])
    out = client.generate(prompt)
    # Try parsing JSON from output tail; fallback to a neutral tie
    try:
        start = out.find("{")
        end = out.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(out[start:end])
    except Exception:
        pass
    return {
        "specificity": "Tie",
        "grounding": "Tie",
        "correctness": "Tie",
        "coverage": "Tie",
        "overall": "Tie",
        "reasons": {k: "model did not return parseable JSON" for k in ["specificity","grounding","correctness","coverage","overall"]},
    }


