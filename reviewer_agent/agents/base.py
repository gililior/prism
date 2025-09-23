
from typing import List
from ..schemas import Paper, Point

class Agent:
    name: str = "Agent"

    def review(self, paper: Paper, spans_text: str) -> List[Point]:
        raise NotImplementedError
