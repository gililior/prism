from reviewer_agent.agents.base import Agent
from reviewer_agent.llm.base import LLMClient


class ReviewerFiguresTables(Agent):
    name = "reviewer_figures"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "figures_tables"


