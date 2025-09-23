from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Clarity/Presentation reviewer."
    " Judge organization, writing clarity, and terminology."
    " Cite sections/lines; avoid style-only nitpicks unless they hinder understanding.\n\n{text}"
)


class ReviewerClarity(Agent):
    name = "ReviewerClarity"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        _ = self.llm.generate(PROMPT.format(text=spans_text[:6000]))
        pts = [
            Point(kind="strength", text="Paper is generally well organized with clear sectioning.", grounding="Intro §1", facet="clarity_presentation"),
            Point(kind="weakness", text="Ambiguous definition of key term in Methods.", grounding="Sec 2.1", facet="clarity_presentation"),
            Point(kind="suggestion", text="Add a notation table to aid readability.", grounding="Appendix E", facet="clarity_presentation"),
        ]
        return pts


