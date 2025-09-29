#!/usr/bin/env python3
"""
Comprehensive evaluation system for reviewer agent.
Combines generation and metrics calculation in a unified interface.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import sys

# Import our modules
from .batch_generation import generate_reviews_batch, create_generation_summary
from .comparison import calculate_metrics_for_runs, create_comprehensive_report

def get_all_paper_ids(emnlp_data: str) -> List[str]:
    """
    Get all available paper IDs from EMNLP dataset directory, sorted numerically.
    
    Args:
        emnlp_data: Path to EMNLP23 data directory
        
    Returns:
        List of paper IDs sorted numerically (0, 1, 2, ... not 10, 2, 20)
    """
    data_path = Path(emnlp_data)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return []
    
    # Get all subdirectories that are numeric paper IDs
    paper_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    # Sort numerically by converting to int, then back to string
    paper_ids = sorted([d.name for d in paper_dirs], key=int)
    
    return paper_ids

def run_full_evaluation(paper_ids: List[str], 
                       emnlp_data: str,
                       model: str = "gemini-2.0-flash-lite",
                       experiment_name: Optional[str] = None,
                       runs_dir: Optional[Path] = None,
                       # Generation parameters
                       routing: str = "dynamic",
                       skip_related: bool = False,
                       skip_rebuttal: bool = False,
                       skip_grounding: bool = False,
                       force: bool = False,
                       workers: int = 4,
                       delay: float = 30.0,
                       max_workers: int = 1,
                       # Evaluation parameters
                       skip_generation: bool = False,
                       skip_metrics: bool = False) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline: generation + metrics calculation.
    
    Args:
        paper_ids: List of paper IDs to evaluate
        emnlp_data: Path to EMNLP23 data directory
        model: Model to use
        experiment_name: Name for this experiment
        runs_dir: Directory for generated reviews
        routing: Routing strategy
        skip_related: Skip related work reviewer
        skip_rebuttal: Skip rebuttal process
        skip_grounding: Skip grounding enforcement
        force: Force regeneration
        workers: Workers for individual papers
        delay: Delay between papers
        max_workers: Parallel paper processes
        skip_generation: Skip generation phase (use existing reviews)
        skip_metrics: Skip metrics calculation
        
    Returns:
        Dictionary with results and paths
    """
    
    # Setup directories
    if runs_dir is None:
        runs_dir = Path("evaluation/results/runs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        # model_short = model.replace("gemini-", "").replace("gpt-", "").replace("-", "_")
        experiment_name = f"eval_{model}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = Path("evaluation/results/experiments") / f"{timestamp}_{experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment_metadata": {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "paper_ids": paper_ids,
            "model": model,
            "experiment_dir": str(experiment_dir),
            "runs_dir": str(runs_dir)
        }
    }
    
    print(f"Starting full evaluation pipeline...")
    print(f"Experiment: {experiment_name}")
    print(f"Papers: {paper_ids}")
    print(f"Model: {model}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Phase 1: Generation
    if not skip_generation:
        print(f"\n{'='*60}")
        print("PHASE 1: Review Generation")
        print("="*60)
        
        generation_results = generate_reviews_batch(
            paper_ids=paper_ids,
            emnlp_data=emnlp_data,
            model=model,
            runs_dir=runs_dir,
            routing=routing,
            skip_related=skip_related,
            skip_rebuttal=skip_rebuttal,
            skip_grounding=skip_grounding,
            force=force,
            workers=workers,
            delay=delay,
            max_workers=max_workers
        )
        
        # Save generation summary
        generation_summary_path = experiment_dir / "generation_summary.json"
        create_generation_summary(generation_results, generation_summary_path)
        
        results["generation_results"] = generation_results
        results["generation_summary_path"] = str(generation_summary_path)
        
        print(f"✓ Generation phase completed")
    else:
        print("⏭ Skipping generation phase (using existing reviews)")
    
    # Phase 2: Metrics Calculation
    if not skip_metrics:
        print(f"\n{'='*60}")
        print("PHASE 2: Metrics Calculation")
        print("="*60)
        
        # Calculate metrics on all available reviews
        comparison_results = calculate_metrics_for_runs(
            runs_dir=runs_dir,
            paper_ids=paper_ids,
            output_dir=experiment_dir
        )
        
        # Convert comparison results to serializable format
        results["comparison_results"] = [
            {
                "paper_id": comp.paper_id,
                "avg_similarity": comp.avg_similarity,
                "avg_coverage": comp.avg_coverage,
                "generated_length": comp.generated_length,
                "strengths_count": comp.strengths_count,
                "weaknesses_count": comp.weaknesses_count,
                "suggestions_count": comp.suggestions_count
            }
            for comp in comparison_results
        ]
        results["metrics_dir"] = str(experiment_dir)
        
        print(f"✓ Metrics calculation completed")
    else:
        print("⏭ Skipping metrics calculation")
    
    # Create final comprehensive report
    final_report_path = experiment_dir / "final_evaluation_report.json"
    with open(final_report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    results["final_report_path"] = str(final_report_path)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Final report: {final_report_path}")
    
    return results

def run_generation_only(paper_ids: List[str], 
                       emnlp_data: str,
                       model: str = "gemini-2.0-flash-lite",
                       runs_dir: Optional[Path] = None,
                       **kwargs) -> Dict[str, Any]:
    """Run only the generation phase"""
    
    if runs_dir is None:
        runs_dir = Path("evaluation/results/runs")
    
    print(f"Starting review generation...")
    print(f"Papers: {paper_ids}")
    print(f"Model: {model}")
    print(f"Runs directory: {runs_dir}")
    
    results = generate_reviews_batch(
        paper_ids=paper_ids,
        emnlp_data=emnlp_data,
        model=model,
        runs_dir=runs_dir,
        **kwargs
    )
    
    # Save summary
    summary_file = runs_dir / "generation_summary.json"
    create_generation_summary(results, summary_file)
    
    print(f"\nGeneration completed!")
    print(f"Summary: {summary_file}")
    print(f"Reviews: {runs_dir}")
    
    return {
        "results": results,
        "summary_path": str(summary_file),
        "runs_dir": str(runs_dir)
    }

def run_metrics_only(runs_dir: Optional[Path] = None,
                    paper_ids: Optional[List[str]] = None,
                    experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """Run only the metrics calculation phase"""
    
    if runs_dir is None:
        runs_dir = Path("evaluation/results/runs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        experiment_name = f"metrics_{timestamp}"
    
    # Create experiment directory
    experiment_dir = Path("evaluation/results/experiments") / f"{timestamp}_{experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting metrics calculation...")
    print(f"Runs directory: {runs_dir}")
    print(f"Experiment directory: {experiment_dir}")
    if paper_ids:
        print(f"Paper IDs: {paper_ids}")
    else:
        print("Paper IDs: All available")
    
    # Calculate metrics
    comparison_results = calculate_metrics_for_runs(
        runs_dir=runs_dir,
        paper_ids=paper_ids,
        output_dir=experiment_dir
    )
    
    print(f"\nMetrics calculation completed!")
    print(f"Results: {experiment_dir}")
    
    return {
        "comparison_results": comparison_results,
        "experiment_dir": str(experiment_dir)
    }

def main():
    """Command line interface for evaluation system"""
    parser = argparse.ArgumentParser(description="Reviewer Agent Evaluation System")
    
    # Mode selection
    parser.add_argument("--mode", choices=["full", "generate", "metrics"], default="full",
                       help="Evaluation mode: full pipeline, generation only, or metrics only")
    
    # Paper selection
    parser.add_argument("--paper_ids", nargs="+", help="List of paper IDs to process")
    parser.add_argument("--paper_list", type=str, help="Path to file containing paper IDs")
    parser.add_argument("--num_papers", type=int, default=100, 
                       help="Number of papers to process (default: 100). Takes first N papers sorted numerically.")
    
    # Basic parameters
    parser.add_argument("--emnlp_data", type=str, 
                       default="/Users/ehabba/Downloads/EMNLP23/data/",
                       help="Path to EMNLP23 data directory")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-lite",
                       help="Model to use")
    parser.add_argument("--experiment_name", type=str, help="Name for this experiment")
    parser.add_argument("--runs_dir", type=str, default="evaluation/results/runs",
                       help="Directory for generated reviews")
    
    # Generation parameters
    parser.add_argument("--routing", type=str, choices=["dynamic", "all"], default="dynamic")
    parser.add_argument("--skip_related", action="store_true")
    parser.add_argument("--skip_rebuttal", action="store_true")
    parser.add_argument("--skip_grounding", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=30.0,
                       help="Delay between processing papers in seconds (default: 30)")
    parser.add_argument("--max_workers", type=int, default=1)
    
    # Phase control
    parser.add_argument("--skip_generation", action="store_true", 
                       help="Skip generation phase (use existing reviews)")
    parser.add_argument("--skip_metrics", action="store_true",
                       help="Skip metrics calculation")
    
    args = parser.parse_args()
    
    # Get paper IDs
    paper_ids = []
    if args.paper_ids:
        paper_ids.extend(args.paper_ids)
    if args.paper_list:
        with open(args.paper_list, 'r') as f:
            paper_ids.extend([line.strip() for line in f if line.strip()])
    
    # If no specific paper IDs provided, use --num_papers to get first N papers
    if not paper_ids:
        print(f"No specific paper IDs provided. Getting first {args.num_papers} papers from dataset...")
        all_paper_ids = get_all_paper_ids(args.emnlp_data)
        if not all_paper_ids:
            print("Error: No papers found in dataset")
            return 1
        paper_ids = all_paper_ids[:args.num_papers]
        print(f"Selected {len(paper_ids)} papers: {paper_ids[:10]}{'...' if len(paper_ids) > 10 else ''}")
    
    runs_dir = Path(args.runs_dir)
    
    try:
        if args.mode == "full":
            
            results = run_full_evaluation(
                paper_ids=paper_ids,
                emnlp_data=args.emnlp_data,
                model=args.model,
                experiment_name=args.experiment_name,
                runs_dir=runs_dir,
                routing=args.routing,
                skip_related=args.skip_related,
                skip_rebuttal=args.skip_rebuttal,
                skip_grounding=args.skip_grounding,
                force=args.force,
                workers=args.workers,
                delay=args.delay,
                max_workers=args.max_workers,
                skip_generation=args.skip_generation,
                skip_metrics=args.skip_metrics
            )
            
        elif args.mode == "generate":
            
            results = run_generation_only(
                paper_ids=paper_ids,
                emnlp_data=args.emnlp_data,
                model=args.model,
                runs_dir=runs_dir,
                routing=args.routing,
                skip_related=args.skip_related,
                skip_rebuttal=args.skip_rebuttal,
                skip_grounding=args.skip_grounding,
                force=args.force,
                workers=args.workers,
                delay=args.delay,
                max_workers=args.max_workers
            )
            
        elif args.mode == "metrics":
            results = run_metrics_only(
                runs_dir=runs_dir,
                paper_ids=paper_ids,
                experiment_name=args.experiment_name
            )
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
