
from typing import List
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient
from pathlib import Path

PROMPT = (Path(__file__).parents[1] / "prompts" / "reviewer_novelty.txt").read_text(encoding="utf-8")

class ReviewerNovelty(Agent):
    name = "ReviewerNovelty"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        prompt = PROMPT.format(text=spans_text[:6000])
        _ = self.llm.generate(prompt)
        pts = [
            Point(kind="strength", text="Positions the work relative to prior art with clear gaps.", grounding="Intro §1.2", facet="novelty"),
            Point(kind="weakness", text="Need stronger evidence distinguishing from closest baseline.", grounding="Related Work", facet="novelty"),
            Point(kind="suggestion", text="Add a head-to-head comparison with contemporaneous method X.", grounding="Sec 4.1", facet="novelty"),
        ]
        return pts
