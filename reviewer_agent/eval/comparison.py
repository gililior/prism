#!/usr/bin/env python3
"""
Advanced comparison tool for evaluating generated reviews against human reviews.
This module provides comprehensive metrics calculation functionality.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import argparse

# Import existing evaluation modules
from .similarity import sentence_level_similarity, _read_review_text
from .metrics import coverage_overlap

@dataclass
class ReviewComparison:
    """Results of comparing a generated review with human reviews"""
    paper_id: str
    generated_review_path: str
    human_reviews_count: int
    
    # Similarity metrics
    avg_similarity: float
    max_similarity: float
    avg_coverage: float
    
    # Content metrics
    generated_length: int
    avg_human_length: int
    genericity_score: float
    
    # Structural metrics
    has_summary: bool
    has_strengths: bool
    has_weaknesses: bool
    has_suggestions: bool
    
    # Detailed results
    similarity_details: List[Dict[str, Any]]
    human_reviews_summary: List[Dict[str, Any]]


def extract_review_structure(review_data: Dict[str, Any]) -> Dict[str, bool]:
    """Extract structural information from review"""
    return {
        "has_summary": bool(review_data.get("summary", "").strip()),
        "has_strengths": bool(review_data.get("strengths")) and len(review_data.get("strengths", [])) > 0,
        "has_weaknesses": bool(review_data.get("weaknesses")) and len(review_data.get("weaknesses", [])) > 0,
        "has_suggestions": bool(review_data.get("suggestions")) and len(review_data.get("suggestions", [])) > 0
    }

def convert_human_review_to_text(human_review: Dict[str, Any]) -> str:
    """Convert human review structure to comparable text"""
    parts = []
    
    # Add report sections
    report = human_review.get("report", {})
    for section_name, content in report.items():
        if content and isinstance(content, str):
            parts.append(f"{section_name}: {content}")
    
    return "\n\n".join(parts) if parts else ""

def compare_single_paper(paper_dir: Path) -> List[ReviewComparison]:
    """Compare generated review with human reviews for a single paper"""
    
    # Extract paper ID from directory name
    if paper_dir.name.startswith("paper_"):
        # New format: paper_100_model_config_timestamp
        parts = paper_dir.name.split("_")
        paper_id = parts[1] if len(parts) > 1 else paper_dir.name
    elif "_paper_" in paper_dir.name:
        # Old format: timestamp_paper_100
        paper_id = paper_dir.name.split("_paper_")[-1]
    else:
        paper_id = paper_dir.name
    
    # Find all available review files
    review_files = []
    
    # Check for original review
    original_path = paper_dir / "review_original.json"
    if original_path.exists():
        review_files.append(("original", original_path))
    
    # Check for updated review (after rebuttal)
    updated_path = paper_dir / "review_updated.json"
    if updated_path.exists():
        review_files.append(("updated", updated_path))
    
    # Fallback to single review.json if exists
    if not review_files:
        single_review_path = paper_dir / "review.json"
        if single_review_path.exists():
            review_files.append(("single", single_review_path))
    
    if not review_files:
        raise FileNotFoundError(f"No generated review found in {paper_dir}")
    
    # Load human reviews
    human_reviews_path = paper_dir / "human_reviews.json"
    if not human_reviews_path.exists():
        raise FileNotFoundError(f"No human reviews found in {paper_dir}")
    
    with open(human_reviews_path, 'r', encoding='utf-8') as f:
        human_reviews = json.load(f)
    
    # Process each review file separately
    comparisons = []
    
    for review_type, review_path in review_files:
        with open(review_path, 'r', encoding='utf-8') as f:
            generated_review = json.load(f)
        
        # Convert generated review to text
        generated_text = _read_review_text(str(review_path))
        generated_length = len(generated_text.split())
        
        # Convert human reviews to text and compute similarities
        similarity_results = []
        human_lengths = []
        human_summaries = []
        
        for i, human_review in enumerate(human_reviews):
            human_text = convert_human_review_to_text(human_review)
            human_length = len(human_text.split())
            human_lengths.append(human_length)
            
            # Create temporary files for similarity calculation
            temp_gen_path = paper_dir / f"temp_generated_{review_type}_{i}.txt"
            temp_human_path = paper_dir / f"temp_human_{review_type}_{i}.txt"
            
            try:
                with open(temp_gen_path, 'w', encoding='utf-8') as f:
                    f.write(generated_text)
                with open(temp_human_path, 'w', encoding='utf-8') as f:
                    f.write(human_text)
                
                # Calculate similarity
                similarity_result = sentence_level_similarity(
                    str(temp_gen_path), 
                    str(temp_human_path)
                )
                
                similarity_results.append({
                    "human_review_index": i,
                    "reviewer_id": human_review.get("rid", f"reviewer_{i}"),
                    "similarity": similarity_result["mean_max_similarity"],
                    "coverage": similarity_result["coverage"],
                    "human_length": human_length,
                    "scores": human_review.get("scores", {})
                })
                
                human_summaries.append({
                    "reviewer_id": human_review.get("rid", f"reviewer_{i}"),
                    "length": human_length,
                    "scores": human_review.get("scores", {}),
                    "has_content": bool(human_text.strip())
                })
                
            finally:
                # Clean up temporary files
                temp_gen_path.unlink(missing_ok=True)
                temp_human_path.unlink(missing_ok=True)
        
        # Calculate aggregate metrics
        similarities = [r["similarity"] for r in similarity_results if r["similarity"] > 0]
        coverages = [r["coverage"] for r in similarity_results if r["coverage"] > 0]
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        avg_coverage = np.mean(coverages) if coverages else 0.0
        avg_human_length = np.mean(human_lengths) if human_lengths else 0
        
        # Create comparison with review type in paper_id
        comparison_paper_id = f"{paper_id}_{review_type}"
        
        # Calculate additional metrics
        genericity_score = 0.0  # Placeholder - implement if needed
        
        # Extract structural information
        structure = extract_review_structure(generated_review)
        
        comparisons.append(ReviewComparison(
            paper_id=comparison_paper_id,
            generated_review_path=str(review_path),
            human_reviews_count=len(human_reviews),
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            avg_coverage=avg_coverage,
            generated_length=generated_length,
            avg_human_length=avg_human_length,
            genericity_score=genericity_score,
            has_summary=structure["has_summary"],
            has_strengths=structure["has_strengths"],
            has_weaknesses=structure["has_weaknesses"],
            has_suggestions=structure["has_suggestions"],
            similarity_details=similarity_results,
            human_reviews_summary=human_summaries
        ))
    
    return comparisons

def create_comparison_report(comparisons: List[ReviewComparison], output_dir: Path):
    """Create comprehensive comparison report"""
    
    # Create summary statistics
    summary_stats = {
        "total_papers": len(comparisons),
        "avg_similarity": np.mean([c.avg_similarity for c in comparisons]),
        "avg_max_similarity": np.mean([c.max_similarity for c in comparisons]),
        "avg_coverage": np.mean([c.avg_coverage for c in comparisons])
    }
    
    # Save summary statistics
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create detailed CSV report
    rows = []
    for comp in comparisons:
        rows.append({
            "paper_id": comp.paper_id,
            "avg_similarity": comp.avg_similarity,
            "max_similarity": comp.max_similarity,
            "avg_coverage": comp.avg_coverage,
            "generated_length": comp.generated_length,
            "avg_human_length": comp.avg_human_length,
            "human_reviews_count": comp.human_reviews_count
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "detailed_comparison.csv", index=False)
    
    # Save detailed results
    detailed_results = []
    for comp in comparisons:
        detailed_results.append({
            "paper_id": comp.paper_id,
            "summary_metrics": {
                "avg_similarity": comp.avg_similarity,
                "max_similarity": comp.max_similarity,
                "avg_coverage": comp.avg_coverage
            },
            "length_metrics": {
                "generated_length": comp.generated_length,
                "avg_human_length": comp.avg_human_length,
                "length_ratio": comp.generated_length / max(comp.avg_human_length, 1)
            },
            "similarity_details": comp.similarity_details,
            "human_reviews_summary": comp.human_reviews_summary
        })
    
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print(f"\nComparison Summary:")
    print(f"Total papers evaluated: {summary_stats['total_papers']}")
    print(f"Average similarity: {summary_stats['avg_similarity']:.3f}")
    print(f"Average max similarity: {summary_stats['avg_max_similarity']:.3f}")
    print(f"Average coverage: {summary_stats['avg_coverage']:.3f}")

def save_comparison_results(comparisons: List[ReviewComparison], output_dir: Path):
    """Save comparison results in multiple formats"""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    summary_stats = {
        "total_papers": len(comparisons),
        "avg_similarity": np.mean([c.avg_similarity for c in comparisons]),
        "avg_max_similarity": np.mean([c.max_similarity for c in comparisons]),
        "avg_coverage": np.mean([c.avg_coverage for c in comparisons])
    }
    
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create CSV for detailed analysis
    csv_data = []
    for comp in comparisons:
        csv_data.append({
            "paper_id": comp.paper_id,
            "avg_similarity": comp.avg_similarity,
            "max_similarity": comp.max_similarity,
            "avg_coverage": comp.avg_coverage,
            "generated_length": comp.generated_length,
            "avg_human_length": comp.avg_human_length,
            "human_reviews_count": comp.human_reviews_count
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_dir / "detailed_comparison.csv", index=False)
    
    # Save detailed JSON results
    detailed_results = []
    for comp in comparisons:
        detailed_results.append({
            "paper_id": comp.paper_id,
            "summary_metrics": {
                "avg_similarity": comp.avg_similarity,
                "max_similarity": comp.max_similarity,
                "avg_coverage": comp.avg_coverage
            },
            "length_metrics": {
                "generated_length": comp.generated_length,
                "avg_human_length": comp.avg_human_length,
                "length_ratio": comp.generated_length / max(comp.avg_human_length, 1)
            },
            "similarity_details": comp.similarity_details,
            "human_reviews_summary": comp.human_reviews_summary
        })
    
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Total papers evaluated: {len(comparisons)}")
    print(f"Average similarity: {summary_stats['avg_similarity']:.3f}")
    print(f"Average max similarity: {summary_stats['avg_max_similarity']:.3f}")
    print(f"Average coverage: {summary_stats['avg_coverage']:.3f}")
    
    print(f"\nComparison results saved to: {output_dir}")

def create_comprehensive_report(comparisons: List[ReviewComparison], 
                              output_dir: Path,
                              experiment_metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Create a comprehensive evaluation report"""
    
    # Analyze results by review type
    original_results = [c for c in comparisons if '_original' in c.paper_id]
    updated_results = [c for c in comparisons if '_updated' in c.paper_id]
    
    # Calculate separate metrics for each review type
    review_type_analysis = {}
    
    if original_results:
        review_type_analysis["original"] = {
            "count": len(original_results),
            "avg_similarity": np.mean([r.avg_similarity for r in original_results]),
            "avg_coverage": np.mean([r.avg_coverage for r in original_results]),
            "avg_length": np.mean([r.generated_length for r in original_results])
        }
    
    if updated_results:
        review_type_analysis["updated"] = {
            "count": len(updated_results),
            "avg_similarity": np.mean([r.avg_similarity for r in updated_results]),
            "avg_coverage": np.mean([r.avg_coverage for r in updated_results]),
            "avg_length": np.mean([r.generated_length for r in updated_results])
        }
    
    # Create comprehensive report
    report = {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_papers_requested": len(set(c.paper_id.split('_')[0] for c in comparisons)),
            "paper_ids": list(set(c.paper_id.split('_')[0] for c in comparisons)),
            "experiment_directory": str(output_dir.parent) if output_dir.parent else str(output_dir),
            "comparison_directory": str(output_dir)
        },
        "experiment_summary": experiment_metadata or {},
        "comparison_summary": {
            "total_papers": len(comparisons),
            "avg_similarity": np.mean([c.avg_similarity for c in comparisons]),
            "avg_max_similarity": np.mean([c.max_similarity for c in comparisons]),
            "avg_coverage": np.mean([c.avg_coverage for c in comparisons])
        },
        "review_type_analysis": review_type_analysis,
        "key_findings": {
            "success_rate": 0,  # Placeholder - can be calculated based on specific criteria
            "avg_similarity_to_humans": np.mean([c.avg_similarity for c in comparisons]),
            "avg_coverage": np.mean([c.avg_coverage for c in comparisons])
        },
        "detailed_paper_results": []
    }
    
    # Add detailed results for each paper
    for comp in comparisons:
        report["detailed_paper_results"].append({
            "paper_id": comp.paper_id,
            "summary_metrics": {
                "avg_similarity": comp.avg_similarity,
                "max_similarity": comp.max_similarity,
                "avg_coverage": comp.avg_coverage
            },
            "length_metrics": {
                "generated_length": comp.generated_length,
                "avg_human_length": comp.avg_human_length,
                "length_ratio": comp.generated_length / max(comp.avg_human_length, 1)
            },
            "similarity_details": comp.similarity_details,
            "human_reviews_summary": comp.human_reviews_summary
        })
    
    # Save comprehensive report
    report_path = output_dir / "final_evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Compare generated reviews with human reviews")
    parser.add_argument("--runs_dir", type=str, default="evaluation/results/runs",
                       help="Directory containing individual run results")
    parser.add_argument("--experiment_dir", type=str,
                       help="Specific experiment directory to evaluate")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for comparison results (default: evaluation/results/comparisons/comparison_TIMESTAMP)")
    parser.add_argument("--paper_ids", nargs="+",
                       help="Specific paper IDs to evaluate (default: all found)")
    parser.add_argument("--recent_only", action="store_true",
                       help="Only compare runs from today (avoid old duplicates)")
    parser.add_argument("--hours_back", type=int, default=24,
                       help="How many hours back to look for runs (default: 24)")
    parser.add_argument("--latest_only", action="store_true",
                       help="Only compare the latest run for each paper (avoid duplicates)")
    
    args = parser.parse_args()
    
    # Determine input directory
    if args.experiment_dir:
        input_dir = Path(args.experiment_dir)
    else:
        input_dir = Path(args.runs_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Find paper directories
    paper_dirs = []
    if args.experiment_dir:
        # Look for run directories within experiment
        for subdir in input_dir.iterdir():
            if subdir.is_dir() and ("_paper_" in subdir.name or subdir.name.startswith("paper_")):
                paper_dirs.append(subdir)
    else:
        # Look for paper directories in runs directory
        for subdir in input_dir.iterdir():
            if subdir.is_dir() and ("_paper_" in subdir.name or subdir.name.startswith("paper_")):
                paper_dirs.append(subdir)
    
    if not paper_dirs:
        print(f"No paper directories found in {input_dir}")
        return 1
    
    # Filter by paper IDs if specified
    if args.paper_ids:
        filtered_dirs = []
        for paper_dir in paper_dirs:
            # Extract paper ID from directory name
            if paper_dir.name.startswith("paper_"):
                parts = paper_dir.name.split("_")
                paper_id = parts[1] if len(parts) > 1 else paper_dir.name
            elif "_paper_" in paper_dir.name:
                paper_id = paper_dir.name.split("_paper_")[-1]
            else:
                paper_id = paper_dir.name
            
            if paper_id in args.paper_ids:
                filtered_dirs.append(paper_dir)
        paper_dirs = filtered_dirs
    
    if not paper_dirs:
        print(f"No matching paper directories found for IDs: {args.paper_ids}")
        return 1
    
    # Filter by recency if requested
    if args.recent_only:
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=args.hours_back)
        recent_dirs = []
        for paper_dir in paper_dirs:
            if paper_dir.stat().st_mtime > cutoff_time.timestamp():
                recent_dirs.append(paper_dir)
        paper_dirs = recent_dirs
        print(f"Filtered to {len(paper_dirs)} recent directories (within {args.hours_back} hours)")
    
    # Keep only latest run per paper if requested
    if args.latest_only:
        paper_to_latest = {}
        for paper_dir in paper_dirs:
            # Extract paper ID
            if paper_dir.name.startswith("paper_"):
                parts = paper_dir.name.split("_")
                paper_id = parts[1] if len(parts) > 1 else paper_dir.name
            elif "_paper_" in paper_dir.name:
                paper_id = paper_dir.name.split("_paper_")[-1]
            else:
                paper_id = paper_dir.name
            
            if paper_id not in paper_to_latest or paper_dir.stat().st_mtime > paper_to_latest[paper_id].stat().st_mtime:
                paper_to_latest[paper_id] = paper_dir
        
        paper_dirs = list(paper_to_latest.values())
        print(f"Filtered to {len(paper_dirs)} latest runs per paper")
    
    print(f"Processing {len(paper_dirs)} paper directories...")
    
    # Compare each paper
    all_comparisons = []
    for paper_dir in paper_dirs:
        try:
            comparisons = compare_single_paper(paper_dir)
            all_comparisons.extend(comparisons)
            print(f"✓ Processed {paper_dir.name}: {len(comparisons)} comparisons")
        except Exception as e:
            print(f"✗ Error processing {paper_dir.name}: {e}")
    
    if not all_comparisons:
        print("No successful comparisons generated")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("evaluation/results/comparisons") / f"comparison_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_comparison_results(all_comparisons, output_dir)
    
    print(f"\n✓ Comparison completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return 0

def calculate_metrics_for_runs(runs_dir: Path, 
                              paper_ids: Optional[List[str]] = None,
                              output_dir: Optional[Path] = None,
                              recent_only: bool = False,
                              hours_back: int = 24,
                              latest_only: bool = False) -> List[ReviewComparison]:
    """
    Calculate metrics for all runs in the specified directory.
    This is the main function used by the evaluation system.
    """
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return []
    
    # Find paper directories
    paper_dirs = find_paper_directories(runs_dir, paper_ids, recent_only, 
                                      hours_back, latest_only)
    
    if not paper_dirs:
        print(f"Error: No paper directories found in {runs_dir}")
        return []
    
    print(f"Found {len(paper_dirs)} paper directories to evaluate")
    
    # Run comparisons
    comparisons = []
    for paper_dir in paper_dirs:
        try:
            paper_comparisons = compare_single_paper(paper_dir)
            comparisons.extend(paper_comparisons)
            
            # Extract paper ID for display
            paper_id = extract_paper_id_from_path(paper_dir)
            for comparison in paper_comparisons:
                print(f"✓ Evaluated {comparison.paper_id}")
            
        except Exception as e:
            print(f"✗ Error evaluating {paper_dir}: {e}")
    
    if not comparisons:
        print("No successful comparisons completed")
        return []
    
    # Save results if output directory is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_comparison_results(comparisons, output_dir)
    
    return comparisons

def save_comparison_results(comparisons: List[ReviewComparison], output_dir: Path):
    """Save comparison results to files"""
    
    # Save detailed results
    detailed_results = [
        {
            "paper_id": comp.paper_id,
            "generated_review_path": comp.generated_review_path,
            "human_reviews_count": comp.human_reviews_count,
            "summary_metrics": {
                "avg_similarity": comp.avg_similarity,
                "max_similarity": comp.max_similarity,
                "avg_coverage": comp.avg_coverage
            },
            "content_metrics": {
                "genericity_score": comp.genericity_score
            },
            "length_metrics": {
                "generated_length": comp.generated_length,
                "avg_human_length": comp.avg_human_length
            },
            "structural_metrics": {
                "has_summary": comp.has_summary,
                "has_strengths": comp.has_strengths,
                "has_weaknesses": comp.has_weaknesses,
                "has_suggestions": comp.has_suggestions
            },
            "similarity_details": comp.similarity_details,
            "human_reviews_summary": comp.human_reviews_summary
        }
        for comp in comparisons
    ]
    
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create summary statistics
    summary_stats = {
        "total_papers": len(comparisons),
        "avg_similarity": np.mean([c.avg_similarity for c in comparisons]),
        "avg_max_similarity": np.mean([c.max_similarity for c in comparisons]),
        "avg_coverage": np.mean([c.avg_coverage for c in comparisons]),
        "avg_genericity": np.mean([c.genericity_score for c in comparisons]),
        "structural_completeness": {
            "has_summary": sum(c.has_summary for c in comparisons) / len(comparisons),
            "has_strengths": sum(c.has_strengths for c in comparisons) / len(comparisons),
            "has_weaknesses": sum(c.has_weaknesses for c in comparisons) / len(comparisons),
            "has_suggestions": sum(c.has_suggestions for c in comparisons) / len(comparisons)
        }
    }
    
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create CSV for detailed analysis
    csv_data = []
    for comp in comparisons:
        csv_data.append({
            "paper_id": comp.paper_id,
            "avg_similarity": comp.avg_similarity,
            "max_similarity": comp.max_similarity,
            "avg_coverage": comp.avg_coverage,
            "genericity_score": comp.genericity_score,
            "generated_length": comp.generated_length,
            "avg_human_length": comp.avg_human_length,
            "has_summary": comp.has_summary,
            "has_strengths": comp.has_strengths,
            "has_weaknesses": comp.has_weaknesses,
            "has_suggestions": comp.has_suggestions,
            "human_reviews_count": comp.human_reviews_count
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_dir / "detailed_comparison.csv", index=False)
    
    # Print summary
    # Count unique papers and review types
    unique_papers = set()
    review_types = set()
    for comp in comparisons:
        if "_" in comp.paper_id:
            paper_id, review_type = comp.paper_id.rsplit("_", 1)
            unique_papers.add(paper_id)
            review_types.add(review_type)
        else:
            unique_papers.add(comp.paper_id)
    
    print(f"\nComparison Summary:")
    print(f"Unique papers: {len(unique_papers)}")
    print(f"Review types: {', '.join(sorted(review_types))}")
    print(f"Total comparisons: {len(comparisons)}")
    print(f"Average similarity: {summary_stats['avg_similarity']:.3f}")
    print(f"Average max similarity: {summary_stats['avg_max_similarity']:.3f}")
    print(f"Average coverage: {summary_stats['avg_coverage']:.3f}")
    print(f"Average genericity: {summary_stats['avg_genericity']:.3f}")
    print(f"\nStructural completeness:")
    for key, value in summary_stats['structural_completeness'].items():
        print(f"  {key}: {value:.1%}")
    
    print(f"\nComparison results saved to: {output_dir}")

def find_paper_directories(runs_dir: Path, 
                          paper_ids: Optional[List[str]] = None,
                          recent_only: bool = False,
                          hours_back: int = 24,
                          latest_only: bool = False) -> List[Path]:
    """Find paper directories in runs directory with filtering options"""
    
    paper_dirs = []
    
    if not runs_dir.exists():
        return paper_dirs
    
    # Find directories with paper data
    for subdir in runs_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        # Check if it's a paper directory
        if not ("_paper_" in subdir.name or subdir.name.startswith("paper_")):
            continue
        
        # Filter by paper IDs if specified
        if paper_ids:
            paper_id = extract_paper_id_from_path(subdir)
            if paper_id not in paper_ids:
                continue
        
        # Filter by time if recent_only is set
        if recent_only:
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            if subdir.stat().st_mtime < cutoff_time.timestamp():
                continue
        
        paper_dirs.append(subdir)
    
    # If latest_only is set, keep only the most recent run for each paper
    if latest_only and paper_dirs:
        from collections import defaultdict
        paper_groups = defaultdict(list)
        
        # Group by paper ID
        for paper_dir in paper_dirs:
            paper_id = extract_paper_id_from_path(paper_dir)
            paper_groups[paper_id].append(paper_dir)
        
        # Keep only the most recent run for each paper
        latest_dirs = []
        for paper_id, dirs in paper_groups.items():
            # Sort by modification time, keep the most recent
            latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)
            latest_dirs.append(latest_dir)
        
        paper_dirs = latest_dirs
    
    return paper_dirs

def extract_paper_id_from_path(paper_dir: Path) -> str:
    """Extract paper ID from directory path"""
    if paper_dir.name.startswith("paper_"):
        # New format: paper_100_model_config
        parts = paper_dir.name.split("_")
        return parts[1] if len(parts) > 1 else paper_dir.name
    elif "_paper_" in paper_dir.name:
        # Old format: timestamp_paper_100
        return paper_dir.name.split("_paper_")[-1]
    else:
        return paper_dir.name

def create_comprehensive_report(comparisons: List[ReviewComparison], 
                              output_dir: Path,
                              experiment_metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Create a comprehensive evaluation report"""
    
    # Analyze results by review type
    original_results = [c for c in comparisons if '_original' in c.paper_id]
    updated_results = [c for c in comparisons if '_updated' in c.paper_id]
    
    # Calculate separate metrics for each review type
    review_type_analysis = {}
    
    if original_results:
        review_type_analysis["original"] = {
            "count": len(original_results),
            "avg_similarity": np.mean([r.avg_similarity for r in original_results]),
            "avg_coverage": np.mean([r.avg_coverage for r in original_results]),
            "avg_length": np.mean([r.generated_length for r in original_results])
        }
    
    if updated_results:
        review_type_analysis["updated"] = {
            "count": len(updated_results),
            "avg_similarity": np.mean([r.avg_similarity for r in updated_results]),
            "avg_coverage": np.mean([r.avg_coverage for r in updated_results]),
            "avg_length": np.mean([r.generated_length for r in updated_results])
        }
    
    # Create comprehensive report
    report = {
        "evaluation_metadata": experiment_metadata or {
            "timestamp": datetime.now().isoformat(),
            "total_comparisons": len(comparisons)
        },
        "review_type_analysis": review_type_analysis,
        "overall_metrics": {
            "avg_similarity": np.mean([c.avg_similarity for c in comparisons]),
            "avg_coverage": np.mean([c.avg_coverage for c in comparisons]),
            "avg_genericity": np.mean([c.genericity_score for c in comparisons])
        },
        "detailed_comparisons": [
            {
                "paper_id": c.paper_id,
                "metrics": {
                    "similarity": c.avg_similarity,
                    "coverage": c.avg_coverage,
                    "genericity": c.genericity_score,
                    "length": c.generated_length
                }
            }
            for c in comparisons
        ]
    }
    
    # Save comprehensive report
    report_path = output_dir / "comprehensive_evaluation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report_path

if __name__ == "__main__":
    exit(main())