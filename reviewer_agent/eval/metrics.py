
from rapidfuzz import fuzz
import re
from typing import List, Dict, Any
import numpy as np

def coverage_overlap(pred_points: list, gold_text: str) -> float:
    """Heuristic: compute max token-sort ratio across predicted points vs gold text."""
    if not gold_text or not pred_points:
        return 0.0
    scores = []
    for p in pred_points:
        scores.append(fuzz.token_sort_ratio(p.lower(), gold_text.lower()))
    return max(scores) if scores else 0.0


def specificity_score(points: List[str]) -> float:
    """Calculate how specific/detailed the review points are."""
    if not points:
        return 0.0
    
    specificity_indicators = [
        r'\b(figure|table|equation|section|page|line)\s+\d+',  # References to specific elements
        r'\b(algorithm|method|approach|technique)\b',  # Technical terms
        r'\b(experiment|evaluation|dataset|baseline)\b',  # Research terms
        r'\b(result|performance|accuracy|precision|recall)\b',  # Quantitative terms
        r'\b\d+(\.\d+)?%\b',  # Percentages
        r'\bp\s*[<>=]\s*0\.\d+',  # Statistical significance
    ]
    
    total_score = 0
    for point in points:
        point_score = 0
        for pattern in specificity_indicators:
            matches = len(re.findall(pattern, point.lower()))
            point_score += matches
        
        # Normalize by length (longer points should have more indicators)
        words = len(point.split())
        normalized_score = point_score / max(words / 10, 1)  # Per 10 words
        total_score += min(normalized_score, 1.0)  # Cap at 1.0 per point
    
    return total_score / len(points)

def constructiveness_score(points: List[str]) -> float:
    """Measure how constructive/actionable the feedback is."""
    if not points:
        return 0.0
    
    constructive_indicators = [
        r'\b(should|could|might|consider|suggest|recommend)\b',
        r'\b(improve|enhance|clarify|explain|add|remove|modify)\b',
        r'\b(alternative|instead|rather than|better)\b',
        r'\b(future work|next step|follow-up)\b',
    ]
    
    destructive_indicators = [
        r'\b(bad|poor|terrible|awful|wrong|incorrect|useless)\b',
        r'\b(not|no|never|nothing|none)\b',
    ]
    
    total_constructive = 0
    total_destructive = 0
    
    for point in points:
        point_lower = point.lower()
        
        constructive_count = sum(len(re.findall(pattern, point_lower)) 
                               for pattern in constructive_indicators)
        destructive_count = sum(len(re.findall(pattern, point_lower)) 
                              for pattern in destructive_indicators)
        
        total_constructive += constructive_count
        total_destructive += destructive_count
    
    total_indicators = total_constructive + total_destructive
    if total_indicators == 0:
        return 0.5  # Neutral if no indicators
    
    return total_constructive / total_indicators

def balance_score(review_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate balance between different review aspects."""
    strengths = review_data.get("strengths", [])
    weaknesses = review_data.get("weaknesses", [])
    suggestions = review_data.get("suggestions", [])
    
    # Count points in each category
    strength_count = len(strengths) if isinstance(strengths, list) else 0
    weakness_count = len(weaknesses) if isinstance(weaknesses, list) else 0
    suggestion_count = len(suggestions) if isinstance(suggestions, list) else 0
    
    total_points = strength_count + weakness_count + suggestion_count
    
    if total_points == 0:
        return {"strength_ratio": 0, "weakness_ratio": 0, "suggestion_ratio": 0, "balance_score": 0}
    
    strength_ratio = strength_count / total_points
    weakness_ratio = weakness_count / total_points
    suggestion_ratio = suggestion_count / total_points
    
    # Calculate balance score (how evenly distributed the points are)
    ratios = [strength_ratio, weakness_ratio, suggestion_ratio]
    # Use entropy-like measure for balance
    balance_score = 1.0 - np.std(ratios) / np.sqrt(1/3)  # Normalize by max std
    
    return {
        "strength_ratio": strength_ratio,
        "weakness_ratio": weakness_ratio, 
        "suggestion_ratio": suggestion_ratio,
        "balance_score": max(0, balance_score)
    }

def grounding_quality(points: List[Dict[str, Any]]) -> float:
    """Evaluate quality of grounding/citations in review points."""
    if not points:
        return 0.0
    
    grounded_count = 0
    total_grounding_quality = 0
    
    for point in points:
        if isinstance(point, dict):
            grounding = point.get("grounding", "")
            if grounding and isinstance(grounding, str):
                grounded_count += 1
                
                # Simple quality heuristics for grounding
                quality_score = 0
                grounding_lower = grounding.lower()
                
                # Check for specific references
                if re.search(r'(section|page|line|figure|table|equation)\s+\d+', grounding_lower):
                    quality_score += 0.4
                
                # Check for quotes or specific text references
                if '"' in grounding or "'" in grounding:
                    quality_score += 0.3
                
                # Check for length (more detailed grounding is better)
                if len(grounding.split()) >= 5:
                    quality_score += 0.3
                
                total_grounding_quality += min(quality_score, 1.0)
    
    if grounded_count == 0:
        return 0.0
    
    # Return average quality of grounded points
    return total_grounding_quality / grounded_count

def comprehensive_review_metrics(review_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate comprehensive metrics for a review."""
    
    # Extract text points from different sections
    all_points = []
    structured_points = []
    
    for section in ["strengths", "weaknesses", "suggestions"]:
        section_data = review_data.get(section, [])
        if isinstance(section_data, list):
            for item in section_data:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        all_points.append(text)
                        structured_points.append(item)
                elif isinstance(item, str):
                    all_points.append(item)
                    structured_points.append({"text": item, "grounding": ""})
    
    # Calculate individual metrics
    metrics = {
        "total_points": len(all_points),
        "specificity_score": specificity_score(all_points),
        "constructiveness_score": constructiveness_score(all_points),
        "grounding_quality": grounding_quality(structured_points),
    }
    
    # Add balance metrics
    balance_metrics = balance_score(review_data)
    metrics.update(balance_metrics)
    
    # Calculate overall quality score (weighted combination)
    quality_components = [
        (metrics["specificity_score"], 0.4),
        (metrics["constructiveness_score"], 0.4),
        (metrics["grounding_quality"], 0.2)
    ]
    
    overall_quality = sum(score * weight for score, weight in quality_components)
    metrics["overall_quality"] = overall_quality
    
    return metrics
