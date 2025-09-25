from .base import Agent
from ..llm.base import LLMClient


class ReviewerSocietalImpact(Agent):
    name = "reviewer_impact"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "societal_impact"


