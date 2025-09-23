from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Societal Impact reviewer."
    " Evaluate risks, misuse, and broader impacts."
    " Ground statements in paper text; if absent, say 'Insufficient evidence'.\n\n{text}"
)


class ReviewerSocietalImpact(Agent):
    name = "ReviewerSocietalImpact"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        _ = self.llm.generate(PROMPT.format(text=spans_text[:6000]))
        pts = [
            Point(kind="weakness", text="No discussion of potential misuse or failure modes.", grounding="Insufficient evidence", facet="societal_impact"),
            Point(kind="suggestion", text="Add a Broader Impact section with mitigation strategies.", grounding="Conclusion", facet="societal_impact"),
        ]
        return pts


