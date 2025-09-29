#!/usr/bin/env python3
"""
Batch generation of reviews with smart caching.
Handles multiple papers efficiently with skip logic for existing reviews.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path for cli import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import LLM constants
from reviewer_agent.llm.constants import LLMModels

def check_review_exists(paper_id: str, model: str, config_flags: List[str], runs_dir: Path) -> bool:
    """Check if a review already exists for the given configuration"""
    config_str = "_".join(config_flags) if config_flags else "default"
    
    # Create model-specific subdirectory path
    model_dir = runs_dir / model
    review_dir = model_dir / f"paper_{paper_id}_{model}_{config_str}"
    
    return review_dir.exists() and (review_dir / "review_original.json").exists()

def run_single_paper_direct(paper_id: str, emnlp_data: str, model: str, runs_dir: Path, 
                           routing: str = "dynamic", skip_related: bool = False, 
                           skip_rebuttal: bool = False, skip_grounding: bool = False,
                           force: bool = False, workers: int = 4) -> Dict[str, Any]:
    """Run review generation directly via Python import (no subprocess)"""
    
    # Check if review already exists (unless force is used)
    if not force:
        config_flags = []
        if skip_rebuttal:
            config_flags.append("no_rebuttal")
        if skip_related:
            config_flags.append("no_related")
        if skip_grounding:
            config_flags.append("no_grounding")
        if routing != "dynamic":
            config_flags.append(f"routing_{routing}")
        
        if check_review_exists(paper_id, model, config_flags, runs_dir):
            return {
                "paper_id": paper_id,
                "status": "skipped",
                "message": f"Review already exists for paper {paper_id}"
            }
    
    try:
        # Import cli main function
        from cli import main as cli_main
        
        # Prepare arguments for cli
        original_argv = sys.argv.copy()
        
        # Build argument list
        cli_args = [
            "cli.py",  # script name
            "--paper_id", paper_id,
            "--emnlp_data", emnlp_data,
            "--model", model,
            "--output_dir", str(runs_dir),
            "--workers", str(workers)
        ]
        
        if routing != "dynamic":
            cli_args.extend(["--routing", routing])
        if skip_related:
            cli_args.append("--skip_related")
        if skip_rebuttal:
            cli_args.append("--skip_rebuttal")
        if skip_grounding:
            cli_args.append("--skip_grounding")
        if force:
            cli_args.append("--force")
        
        # Set sys.argv for argparse
        sys.argv = cli_args
        
        print(f"Generating review for paper {paper_id}...")
        
        # Call cli main function directly
        cli_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        return {
            "paper_id": paper_id,
            "status": "success",
            "message": "Review generated successfully"
        }
        
    except SystemExit as e:
        # cli.py might call sys.exit(), handle gracefully
        sys.argv = original_argv
        if e.code == 0:
            return {
                "paper_id": paper_id,
                "status": "success",
                "message": "Review generated successfully"
            }
        else:
            return {
                "paper_id": paper_id,
                "status": "error",
                "error": f"CLI exited with code {e.code}"
            }
            
    except Exception as e:
        sys.argv = original_argv
        error_msg = str(e)
        
        # Check for specific error patterns
        if "Paper" in error_msg and "not found" in error_msg:
            return {
                "paper_id": paper_id,
                "status": "not_found",
                "error": f"Paper {paper_id} not found in dataset"
            }
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "paper_id": paper_id,
                "status": "quota_error",
                "error": "API quota exceeded"
            }
        else:
            return {
                "paper_id": paper_id,
                "status": "error",
                "error": error_msg
            }

def generate_reviews_batch(paper_ids: List[str], emnlp_data: str, model: str = LLMModels.DEFAULT_MODEL.value,
                          runs_dir: Optional[Path] = None, routing: str = "dynamic",
                          skip_related: bool = False, skip_rebuttal: bool = False, 
                          skip_grounding: bool = False, force: bool = False,
                          workers: int = 1, delay: float = 30.0, 
                          max_workers: int = 1) -> List[Dict[str, Any]]:
    """
    Generate reviews for multiple papers with smart caching.
    
    Args:
        paper_ids: List of paper IDs to process
        emnlp_data: Path to EMNLP23 data directory
        model: Model to use for generation
        runs_dir: Directory to save reviews (default: evaluation/results/runs)
        routing: Routing strategy ("dynamic" or "all")
        skip_related: Skip related work reviewer
        skip_rebuttal: Skip rebuttal process
        skip_grounding: Skip grounding enforcement
        force: Force regeneration even if reviews exist
        workers: Number of workers for individual paper processing
        delay: Delay between papers in seconds
        max_workers: Number of parallel paper processes
        
    Returns:
        List of result dictionaries for each paper
    """
    
    # Setup directories
    if runs_dir is None:
        runs_dir = Path("evaluation/results/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting batch review generation...")
    print(f"Papers: {paper_ids}")
    print(f"Model: {model}")
    print(f"Runs directory: {runs_dir}")
    print(f"Max workers: {max_workers}")
    
    # Check existing reviews (unless --force is used)
    if not force:
        config_flags = []
        if skip_rebuttal:
            config_flags.append("no_rebuttal")
        if skip_related:
            config_flags.append("no_related")
        if skip_grounding:
            config_flags.append("no_grounding")
        if routing != "dynamic":
            config_flags.append(f"routing_{routing}")
        
        existing_count = 0
        for paper_id in paper_ids:
            if check_review_exists(paper_id, model, config_flags, runs_dir):
                existing_count += 1
        
        if existing_count > 0:
            print(f"Found {existing_count} existing reviews (will be skipped unless --force is used)")
    
    # Run papers
    results = []
    
    if max_workers == 1:
        # Sequential execution (recommended for API quota management)
        last_processed_paper = None  # Track last actually processed paper
        
        for paper_id in paper_ids:
            # First, check if we need to skip this paper (without processing)
            if not force:
                config_flags = []
                if skip_rebuttal:
                    config_flags.append("no_rebuttal")
                if skip_related:
                    config_flags.append("no_related")
                if skip_grounding:
                    config_flags.append("no_grounding")
                if routing != "dynamic":
                    config_flags.append(f"routing_{routing}")
                
                if check_review_exists(paper_id, model, config_flags, runs_dir):
                    print(f"⏭ Skipped {paper_id} (already exists)")
                    results.append({
                        "paper_id": paper_id,
                        "status": "skipped",
                        "message": f"Review already exists for paper {paper_id}"
                    })
                    continue  # Skip to next paper without delay
            
            # Add delay before processing (if we had a previous processed paper)
            if last_processed_paper is not None:
                print(f"Waiting {delay} seconds before processing {paper_id}...")
                time.sleep(delay)
            
            try:
                result = run_single_paper_direct(
                    paper_id, emnlp_data, model, runs_dir,
                    routing, skip_related, skip_rebuttal, skip_grounding, force, workers
                )
                results.append(result)
                
                if result["status"] == "success":
                    print(f"✓ Generated review for {paper_id}")
                    last_processed_paper = paper_id
                elif result["status"] == "skipped":
                    print(f"⏭ Skipped {paper_id} (already exists)")
                    # This shouldn't happen since we check above, but just in case
                else:
                    print(f"✗ Failed {paper_id}: {result.get('error', 'Unknown error')}")
                    last_processed_paper = paper_id
                    
            except Exception as e:
                print(f"✗ Exception processing {paper_id}: {e}")
                results.append({
                    "paper_id": paper_id,
                    "status": "exception",
                    "error": str(e)
                })
                last_processed_paper = paper_id
    else:
        # Parallel execution (use with caution for quota limits)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_paper = {
                executor.submit(
                    run_single_paper_direct, paper_id, emnlp_data, model, runs_dir,
                    routing, skip_related, skip_rebuttal, skip_grounding, force, workers
                ): paper_id
                for paper_id in paper_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_paper):
                paper_id = future_to_paper[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        print(f"✓ Generated review for {paper_id}")
                    elif result["status"] == "skipped":
                        print(f"⏭ Skipped {paper_id} (already exists)")
                    else:
                        print(f"✗ Failed {paper_id}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"✗ Exception processing {paper_id}: {e}")
                    results.append({
                        "paper_id": paper_id,
                        "status": "exception",
                        "error": str(e)
                    })
    
    return results

def create_generation_summary(results: List[Dict[str, Any]], output_file: Path):
    """Create a summary of the generation process"""
    total = len(results)
    successful = len([r for r in results if r["status"] == "success"])
    skipped = len([r for r in results if r["status"] == "skipped"])
    failed = len([r for r in results if r["status"] not in ["success", "skipped"]])
    
    summary = {
        "generation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_papers": total,
            "successful": successful,
            "skipped": skipped,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "generation_rate": (successful + skipped) / total if total > 0 else 0
        },
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGeneration Summary:")
    print(f"Total papers: {total}")
    print(f"Successfully generated: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total:.1%}")
    print(f"Generation rate: {(successful + skipped)/total:.1%}")
    
    if failed > 0:
        print(f"\nFailed papers:")
        failed_runs = [r for r in results if r["status"] not in ["success", "skipped"]]
        for run in failed_runs:
            print(f"  - {run['paper_id']}: {run['status']} - {run.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Generate reviews for multiple papers")
    parser.add_argument("--paper_ids", nargs="+", help="List of paper IDs to process")
    parser.add_argument("--paper_list", type=str, help="Path to file containing paper IDs (one per line)")
    parser.add_argument("--emnlp_data", type=str, 
                       default="/Users/ehabba/Downloads/EMNLP23/data/",
                       help="Path to EMNLP23 data directory")
    parser.add_argument("--model", type=str, default=LLMModels.DEFAULT_MODEL.value,
                       help="Model to use for generation")
    parser.add_argument("--runs_dir", type=str, default="evaluation/results/runs",
                       help="Directory to save generated reviews")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of workers for individual paper processing")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Number of parallel paper processes (default: 1 to avoid quota issues)")
    parser.add_argument("--delay", type=float, default=30.0,
                       help="Delay between papers in seconds (default: 30)")
    
    # Pass-through arguments for cli.py
    parser.add_argument("--routing", type=str, choices=["dynamic", "all"], default="dynamic")
    parser.add_argument("--skip_related", action="store_true")
    parser.add_argument("--skip_rebuttal", action="store_true")
    parser.add_argument("--skip_grounding", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if reviews already exist")
    
    # Output options
    parser.add_argument("--summary_file", type=str, help="Path to save generation summary")
    
    args = parser.parse_args()
    
    # Get paper IDs
    paper_ids = []
    if args.paper_ids:
        paper_ids.extend(args.paper_ids)
    
    if args.paper_list:
        with open(args.paper_list, 'r') as f:
            paper_ids.extend([line.strip() for line in f if line.strip()])
    
    if not paper_ids:
        print("Error: No paper IDs provided. Use --paper_ids or --paper_list")
        return 1
    
    # Run batch generation
    runs_dir = Path(args.runs_dir)
    results = generate_reviews_batch(
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
    
    # Create generation summary
    summary_file = Path(args.summary_file) if args.summary_file else runs_dir / "generation_summary.json"
    create_generation_summary(results, summary_file)
    
    print(f"\nReview generation completed!")
    print(f"Summary saved to: {summary_file}")
    print(f"Reviews saved to: {runs_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
