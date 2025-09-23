from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Reproducibility reviewer."
    " Assess code/data availability, hyperparameters, and seeds."
    " Cite exact sections/appendix.\n\n{text}"
)


class ReviewerReproducibility(Agent):
    name = "ReviewerReproducibility"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        _ = self.llm.generate(PROMPT.format(text=spans_text[:6000]))
        pts = [
            Point(kind="strength", text="Hyperparameters and training recipe are enumerated.", grounding="Appendix A", facet="reproducibility"),
            Point(kind="weakness", text="No seed reporting; variance across runs unknown.", grounding="Sec 3.1", facet="reproducibility"),
            Point(kind="suggestion", text="Release code and scripts with a reproducible environment file.", grounding="GitHub link placeholder", facet="reproducibility"),
        ]
        return pts


