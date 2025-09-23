
from typing import List
from ..schemas import Point

def rebut(points: List[Point]) -> List[str]:
    """Toy rebuttal: flags points with labels based on keyword matches and returns evidence hooks."""
    outs = []
    for p in points:
        if "confidence interval" in p.text.lower():
            outs.append("MISUNDERSTANDING: CIs provided in Appendix B (see Table B.3).")
        elif "power" in p.text.lower():
            outs.append("VALID POINT: We will add a power analysis for the ablation.")
        else:
            outs.append("MISSING CONTEXT: Related results in Sec 4.3 address this partially.")
    return outs
