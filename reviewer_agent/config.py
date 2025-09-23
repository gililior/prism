
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    facets: List[str] = field(default_factory=lambda: [
        "methods", "claims_vs_evidence", "novelty", "reproducibility",
        "ethics_licensing", "clarity_presentation", "figures_tables", "societal_impact"
    ])
    reviewers_for_facets: Dict[str, str] = field(default_factory=lambda: {
        "methods": "ReviewerMethods",
        "novelty": "ReviewerNovelty",
        "claims_vs_evidence": "ReviewerClaimsEvidence",
        "reproducibility": "ReviewerReproducibility",
        "ethics_licensing": "ReviewerEthicsLicensing",
        "clarity_presentation": "ReviewerClarity",
        "figures_tables": "ReviewerFiguresTables",
        "societal_impact": "ReviewerSocietalImpact",
        # Note: Related work reviewer is invoked separately in CLI for top citations
    })
    grounding_required: bool = True
    max_points_per_facet: int = 4
