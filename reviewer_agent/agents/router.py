from typing import Dict, List, Tuple
from ..schemas import Paper, Section


class DynamicRouter:
    """Select relevant facets and map sections per facet using tagged spans.

    Heuristic MVP:
    - Score each facet by total characters covered by spans tagged with that facet.
    - Keep facets with score > 0, optionally cap to top_k.
    - For each kept facet, include sections that contain at least one span tagged with it.
    - Provide concatenated text per facet limited to max_chars.
    """

    def __init__(self, top_k: int = 8, max_chars: int = 12000):
        self.top_k = top_k
        self.max_chars = max_chars

    def route(self, paper: Paper) -> Dict[str, Dict[str, List[str] or str]]:
        facet_score: Dict[str, int] = {}
        facet_to_sections: Dict[str, List[str]] = {}
        facet_to_texts: Dict[str, List[str]] = {}

        for sec in paper.sections:
            for sp in getattr(sec, "spans", []) or []:
                span_len = max(0, (sp.end - sp.start))
                for f in sp.facets:
                    facet_score[f] = facet_score.get(f, 0) + span_len
                    if f not in facet_to_sections:
                        facet_to_sections[f] = []
                    if sec.name not in facet_to_sections[f]:
                        facet_to_sections[f].append(sec.name)
                    facet_to_texts.setdefault(f, []).append(sp.text)

        # Select top facets with any coverage
        ranked = [f for f, s in sorted(facet_score.items(), key=lambda kv: kv[1], reverse=True) if s > 0]
        selected = ranked[: self.top_k] if self.top_k else ranked

        routed: Dict[str, Dict[str, List[str] or str]] = {}
        for f in selected:
            texts = facet_to_texts.get(f, [])
            concat = "\n\n".join(texts)
            if len(concat) > self.max_chars:
                concat = concat[: self.max_chars]
            routed[f] = {
                "sections": facet_to_sections.get(f, []),
                "text": concat,
            }
        return routed


