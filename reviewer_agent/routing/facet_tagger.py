
from typing import List
from ..schemas import Paper, Span, Section

# Lightweight aspect detector keywords for facet tagging
KEYWORDS = {
    "methods": ["method", "architecture", "training", "loss", "optimization", "experiment design"],
    "novelty": ["novel", "we propose", "contribution", "our approach differs"],
    "claims_vs_evidence": ["we show", "results indicate", "significant improvement", "evidence"],
    "reproducibility": ["code", "available", "seed", "hyperparameter", "release", "dataset"],
    "figures_tables": ["figure", "fig.", "table", "tab."],
    "clarity_presentation": ["clarity", "readability", "typo", "grammar"],
    "ethics_licensing": ["license", "ethic", "consent", "privacy", "terms of use"],
    "societal_impact": ["risk", "bias", "harm", "misuse", "societal", "impact"],
}

# Section-level routing defaults: section → primary facets
SECTION_FACETS = {
    "title": ["novelty"],
    "abstract": ["claims_vs_evidence", "novelty"],
    "introduction": ["novelty", "claims_vs_evidence", "societal_impact"],
    "related work": ["novelty"],
    "background": ["novelty"],
    "methods": ["methods", "reproducibility"],
    "method": ["methods", "reproducibility"],
    "approach": ["methods", "reproducibility"],
    "experiments": ["claims_vs_evidence", "methods"],
    "results": ["claims_vs_evidence", "figures_tables"],
    "discussion": ["claims_vs_evidence", "societal_impact"],
    "conclusion": ["claims_vs_evidence"],
    "appendix": ["methods", "reproducibility"],
    "supplement": ["methods", "reproducibility"],
}

def _norm(name: str) -> str:
    return (name or "").strip().lower()

def tag_facets(paper: Paper, window: int = 500) -> Paper:
    """Tag spans with reusable expertise facets.

    Section routing: assign section-default facets first; then refine with keyword hits.
    We keep section-level granularity by default and only drop to smaller windows in
    dense technical sections like Methods.
    """
    for sec in paper.sections:
        text = sec.text or ""
        spans: List[Span] = []

        sec_name = _norm(sec.name)
        default_facets = []
        # Match by startswith to handle variations like "Section 3: Methods"
        for key, facets in SECTION_FACETS.items():
            if sec_name.startswith(key):
                default_facets = facets
                break

        # Use tighter window for methods-like sections, else one span for whole section
        is_methods_like = any(sec_name.startswith(k) for k in ["methods", "method", "approach", "appendix", "supplement"])
        step = 300 if is_methods_like else len(text) or 1

        for i in range(0, len(text), step):
            chunk = text[i:i+step]
            chunk_lower = chunk.lower()
            kw_facets = [f for f, kws in KEYWORDS.items() if any(kw in chunk_lower for kw in kws)]
            merged = list({*default_facets, *kw_facets})
            spans.append(Span(start=i, end=i+len(chunk), text=chunk, facets=merged))

        # Fallback: if section empty, still record an empty span with defaults
        if not spans:
            spans = [Span(start=0, end=0, text="", facets=default_facets)]

        sec.spans = spans
    return paper
