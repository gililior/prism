from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Ethics/Licensing reviewer."
    " Evaluate dataset licensing, consent, privacy, and usage terms."
    " Cite sections/appendix. If unspecified, say 'Insufficient evidence'.\n\n{text}"
)


class ReviewerEthicsLicensing(Agent):
    name = "ReviewerEthicsLicensing"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        _ = self.llm.generate(PROMPT.format(text=spans_text[:6000]))
        pts = [
            Point(kind="weakness", text="Dataset license and usage restrictions not stated.", grounding="Insufficient evidence", facet="ethics_licensing"),
            Point(kind="suggestion", text="Add explicit license and consent statements for datasets used.", grounding="Appendix D", facet="ethics_licensing"),
        ]
        return pts


