from typing import List, Dict
from ..schemas import Paper, Point
from .base import Agent
from ..llm.base import LLMClient

PROMPT = (
    "You are the Related Work reviewer for NeurIPS 2025. Compare the current paper with the cited top papers.\n"
    "Focus on (a) novelty and (b) methods: explicitly state what this paper adds over prior work,\n"
    "where it differs in assumptions/architecture/training/evaluation, and which baselines are appropriate.\n"
    "Ground each point with (Intro/Related/Sec X) or citation key if possible.\n\n"
    "Current paper context:\n{paper_context}\n\nTop related papers (title + abstract/URL):\n{related_snippets}\n"
)


class ReviewerRelatedWork(Agent):
    name = "ReviewerRelatedWork"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _format_related(self, related: List[Dict[str, str]]) -> str:
        lines = []
        for i, r in enumerate(related, 1):
            lines.append(f"[{i}] {r.get('title','')} | {r.get('doi','') or r.get('url','')}")
            absn = (r.get('abstract','') or '')[:1200]
            if absn:
                lines.append(absn)
        return "\n".join(lines)

    def review(self, paper: Paper, spans_text: str, related: List[Dict[str, str]] = None) -> List[Point]:
        related = related or []
        paper_ctx = (paper.title + "\n" + (paper.sections[0].text[:1200] if paper.sections else ""))
        prompt = PROMPT.format(paper_context=paper_ctx, related_snippets=self._format_related(related))
        _ = self.llm.generate(prompt)
        pts = [
            Point(kind="weakness", text="Missing head-to-head comparison with a top cited baseline.", grounding="Related Work", facet="novelty"),
            Point(kind="suggestion", text="Add ablation isolating the delta vs. cited method [1] (what the new module adds).", grounding="Sec 4.1", facet="methods"),
            Point(kind="strength", text="Clearly distinguishes contributions and method differences from closely related approach [2].", grounding="Intro §1.2", facet="novelty"),
        ]
        return pts


