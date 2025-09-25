from .base import Agent
from ..llm.base import LLMClient


class ReviewerClaimsEvidence(Agent):
    name = "reviewer_claims"

    def __init__(self, llm: LLMClient, config=None):
        super().__init__(config, llm)
        self.facet = "claims_vs_evidence"


