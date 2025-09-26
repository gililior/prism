from typing import Dict, List

from reviewer_agent.config import Config
from reviewer_agent.schemas import Paper


class SectionBasedRouter:
    """
    Simple router that maps reviewers to sections based on clear, predictable rules.
    
    Philosophy:
    - Each reviewer has specific sections they should focus on
    - Some reviewers see the whole paper (like clarity)
    - Clear, maintainable mapping without complex scoring
    """

    def __init__(self, config: Config = None, max_chars: int = 12000):
        self.config = config or Config()
        self.max_chars = max_chars

        # Define which sections each facet/reviewer should examine
        self.facet_section_mapping = {
            "methods": ["method", "methods", "approach", "methodology", "experiments", "experiment"],
            "novelty": ["introduction", "abstract", "contribution", "related work", "background"],
            "claims_vs_evidence": ["results", "experiments", "evaluation", "discussion", "conclusion"],
            "reproducibility": ["method", "methods", "experiments", "experiment", "appendix", "supplement"],
            "clarity_presentation": ["*"],  # Should see everything for overall clarity
            "figures_tables": ["results", "experiments", "evaluation"],
            "ethics_licensing": ["introduction", "discussion", "conclusion", "ethics", "limitations"],
            "societal_impact": ["introduction", "discussion", "conclusion", "limitations", "impact"]
        }

    def route(self, paper: Paper) -> Dict[str, Dict[str, str]]:
        """
        Route paper sections to appropriate reviewers.
        
        Returns:
            Dict mapping facet -> {"text": relevant_text, "sections": section_names}
        """
        routing = {}

        for facet in self.config.facets:
            if facet not in self.config.reviewers_for_facets:
                continue

            target_sections = self.facet_section_mapping.get(facet, [])
            relevant_text, section_names = self._get_text_for_facet(paper, target_sections)

            if relevant_text.strip():
                routing[facet] = {
                    "text": relevant_text,
                    "sections": section_names
                }

        return routing

    def _get_text_for_facet(self, paper: Paper, target_sections: List[str]) -> tuple[str, List[str]]:
        """
        Extract relevant text and section names for a facet.
        
        Args:
            paper: The paper to extract from
            target_sections: List of section name patterns to match, or ["*"] for all sections
            
        Returns:
            (concatenated_text, list_of_section_names)
        """
        if "*" in target_sections:
            # Return full paper text with section headers
            text_parts = []
            section_names = []
            for section in paper.sections:
                text_parts.append(f"## {section.name}\n{section.text}")
                section_names.append(section.name)
            full_text = "\n\n".join(text_parts)

            # Truncate if too long
            if len(full_text) > self.max_chars:
                full_text = full_text[:self.max_chars] + "\n\n[Text truncated...]"

            return full_text, section_names

        # Find matching sections
        relevant_sections = []
        section_names = []

        for section in paper.sections:
            section_name_lower = section.name.lower().strip()

            # Check if this section matches any of the target patterns
            if any(self._section_matches(section_name_lower, target) for target in target_sections):
                relevant_sections.append(f"## {section.name}\n{section.text}")
                section_names.append(section.name)

        # Concatenate and truncate if needed
        full_text = "\n\n".join(relevant_sections)
        if len(full_text) > self.max_chars:
            full_text = full_text[:self.max_chars] + "\n\n[Text truncated...]"

        return full_text, section_names

    def _section_matches(self, section_name: str, target_pattern: str) -> bool:
        """
        Check if a section name matches a target pattern.
        
        Examples:
        - "3. Methods" matches "method"
        - "Related Work" matches "related work"
        - "Experimental Setup" matches "experiment"
        """
        target_lower = target_pattern.lower().strip()

        # Direct substring match
        if target_lower in section_name:
            return True

        # Handle common variations
        if target_lower == "method" and any(word in section_name for word in ["method", "approach", "technique"]):
            return True
        if target_lower == "experiment" and any(word in section_name for word in ["experiment", "evaluation", "setup"]):
            return True
        if target_lower == "results" and any(word in section_name for word in ["result", "finding", "outcome"]):
            return True

        return False


# Keep the old DynamicRouter for backward compatibility (for now)
class DynamicRouter:
    """
    DEPRECATED: Use SectionBasedRouter instead.
    
    This class is kept for backward compatibility but will be removed in future versions.
    """

    def __init__(self, top_k: int = 8, max_chars: int = 12000):
        print("WARNING: DynamicRouter is deprecated. Use SectionBasedRouter instead.")
        self.router = SectionBasedRouter(max_chars=max_chars)

    def route(self, paper: Paper) -> Dict[str, Dict[str, List[str] or str]]:
        """Route using the new SectionBasedRouter but return old format for compatibility"""
        new_routing = self.router.route(paper)

        # Convert new format to old format
        old_format = {}
        for facet, info in new_routing.items():
            old_format[facet] = {
                "text": info["text"],
                "sections": info["sections"]  # This was List[str] in old format too
            }

        return old_format
