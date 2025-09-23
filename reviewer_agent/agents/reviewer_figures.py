from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Figures/Tables reviewer."
    " Evaluate whether figures/tables are legible, captioned, and support claims."
    " Cite figure/table identifiers.\n\n{text}"
)


class ReviewerFiguresTables(Agent):
    name = "ReviewerFiguresTables"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        _ = self.llm.generate(PROMPT.format(text=spans_text[:6000]))
        pts = [
            Point(kind="strength", text="Figures are readable and match described results.", grounding="Fig 2", facet="figures_tables"),
            Point(kind="weakness", text="Some tables omit standard deviation or CI.", grounding="Table 1", facet="figures_tables"),
            Point(kind="suggestion", text="Add per-class results figure for qualitative insights.", grounding="Fig 4 (proposed)", facet="figures_tables"),
        ]
        return pts


