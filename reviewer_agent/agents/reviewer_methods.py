
from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient
from pathlib import Path

PROMPT = (Path(__file__).parents[1] / "prompts" / "reviewer_methods.txt").read_text(encoding="utf-8")

class ReviewerMethods(Agent):
    name = "ReviewerMethods"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        prompt = PROMPT.format(text=spans_text[:6000])
        out = self.llm.generate(prompt)
        # For MVP, produce a couple of deterministic points
        pts = [
            Point(kind="weakness", text="Statistical power unclear for ablation study.", grounding="Sec 3.2", facet="methods"),
            Point(kind="suggestion", text="Report confidence intervals for all metrics.", grounding="Table 2", facet="methods"),
            Point(kind="strength", text="Clear training setup with reproducible hyperparameters.", grounding="Appendix A", facet="methods"),
        ]
        return pts
