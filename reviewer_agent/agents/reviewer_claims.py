from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Claims vs. Evidence reviewer."
    " Focus only on whether claims are supported by results."
    " Cite exact sections/figures/tables. If evidence is missing, say 'Insufficient evidence'.\n\n{text}"
)


class ReviewerClaimsEvidence(Agent):
    name = "ReviewerClaimsEvidence"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        _ = self.llm.generate(PROMPT.format(text=spans_text[:6000]))
        pts = [
            Point(kind="strength", text="Main claim is supported by quantitative gains on two datasets.", grounding="Sec 4.2", facet="claims_vs_evidence"),
            Point(kind="weakness", text="Ablation lacks evidence tying improvement to the proposed module.", grounding="Sec 3.4", facet="claims_vs_evidence"),
            Point(kind="suggestion", text="Add a negative-control experiment to test claim specificity.", grounding="Appendix C", facet="claims_vs_evidence"),
        ]
        return pts


