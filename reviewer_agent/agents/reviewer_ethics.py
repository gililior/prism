from .base import Agent
from ..llm.base import LLMClient


class ReviewerEthicsLicensing(Agent):
    name = "reviewer_ethics"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "ethics_licensing"


