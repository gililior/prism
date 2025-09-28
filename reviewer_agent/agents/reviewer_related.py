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
            # Format title and DOI/URL
            title = r.get('title', '')
            doi_or_url = r.get('doi', '') or r.get('url', '')
            lines.append(f"[{i}] {title} | {doi_or_url}")
            
            # Add author and year info if available
            authors = r.get('authors', '')
            year = r.get('year', '')
            venue = r.get('venue', '')
            
            metadata_parts = []
            if authors:
                metadata_parts.append(f"Authors: {authors}")
            if year:
                metadata_parts.append(f"Year: {year}")
            if venue:
                metadata_parts.append(f"Venue: {venue}")
            
            if metadata_parts:
                lines.append(" | ".join(metadata_parts))
            
            # Add summary/abstract
            summary = r.get('summary', '')
            if summary and summary != title[:200]:  # Don't repeat if summary is just truncated title
                # Check if this looks like a real abstract (longer, more descriptive)
                if len(summary) > 200 and not summary.startswith("Authors:"):
                    lines.append(f"Abstract: {summary}")
                else:
                    lines.append(f"Summary: {summary}")
            
            lines.append("")  # Empty line between citations
        
        return "\n".join(lines).strip()
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

