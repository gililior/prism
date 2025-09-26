from reviewer_agent.agents.base import Agent
from reviewer_agent.llm.base import LLMClient


class ReviewerClarity(Agent):
    name = "reviewer_clarity"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "clarity_presentation"


