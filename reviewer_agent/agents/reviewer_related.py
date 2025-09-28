import json
from typing import List, Dict
from pathlib import Path
from reviewer_agent.agents.base import Agent
from reviewer_agent.llm.base import LLMClient
from reviewer_agent.schemas import Paper, Point
from reviewer_agent.llm.constants import TaskLLMConfigs


class ReviewerRelatedWork(Agent):
    name = "reviewer_related"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "related_work"

    def _format_related(self, related: List[Dict[str, str]]) -> str:
        lines = []
        for i, r in enumerate(related, 1):
            lines.append(f"[{i}] {r.get('title','')} | {r.get('doi','') or r.get('url','')}")
            absn = (r.get('abstract','') or '')[:1200]
            if absn:
                lines.append(absn)
        return "\n".join(lines)
    def review(self, paper: Paper, spans_text: str, related: List[Dict[str, str]] = None) -> List[Point]:
        # Load the prompt for this reviewer type
        prompt_file = Path(__file__).parents[1] / "prompts" / "reviewer_related.txt"
        related = related or []
        paper_ctx = (paper.title + "\n" + (paper.sections[0].text[:1200] if paper.sections else ""))
        prompt_template = prompt_file.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            paper_context=paper_ctx,
            related_snippets=self._format_related(related),
            text=spans_text[:self.config.max_text_length]
        )
        config = TaskLLMConfigs.REVIEWER_RELATED
        response = self.llm.generate(prompt, temperature=config.temperature, max_tokens=config.max_tokens)
        points_data = json.loads(response)

        points = []
        for point_data in points_data:
            if isinstance(point_data, dict) and all(key in point_data for key in ['kind', 'text']):
                point = Point(
                    kind=point_data['kind'],
                    text=point_data['text'],
                    grounding=point_data.get('grounding'),
                    facet=point_data.get('facet')
                )
                points.append(point)

        return points

