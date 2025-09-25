
import json
from typing import List
from pathlib import Path
from ..schemas import Paper, Point
from ..config import Config

class Agent:
    name: str = "Agent"

    def __init__(self, config: Config = None, llm=None):
        self.config = config or Config()
        self.llm = llm

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        # Load the prompt for this reviewer type
        prompt_file = Path(__file__).parents[1] / "prompts" / f"{self.name.lower()}.txt"
        prompt_template = prompt_file.read_text(encoding="utf-8")
        prompt = prompt_template.format(text=spans_text[:self.config.max_text_length])
        response = self.llm.generate(prompt)
        points_data =  json.loads(response)

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
