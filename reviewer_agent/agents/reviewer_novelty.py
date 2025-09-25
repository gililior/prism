
from .base import Agent
from ..llm.base import LLMClient

class ReviewerNovelty(Agent):
    name = "reviewer_novelty"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "novelty"
