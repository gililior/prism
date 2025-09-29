
import argparse, json, os, datetime, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from reviewer_agent.schemas import Paper, Rubric
from reviewer_agent.parsing.pdf_to_json import parse_pdf_to_paper, parse_pdf_file_to_paper
from reviewer_agent.routing.facet_tagger import tag_facets
from reviewer_agent.config import Config
from reviewer_agent.llm.base import LLMClient
from reviewer_agent.agents.reviewer_methods import ReviewerMethods
from reviewer_agent.agents.reviewer_novelty import ReviewerNovelty
from reviewer_agent.agents.reviewer_claims import ReviewerClaimsEvidence
from reviewer_agent.agents.reviewer_repro import ReviewerReproducibility
from reviewer_agent.agents.reviewer_ethics import ReviewerEthicsLicensing
from reviewer_agent.agents.reviewer_figures import ReviewerFigures
from reviewer_agent.agents.reviewer_tables import ReviewerTables
from reviewer_agent.agents.reviewer_clarity import ReviewerClarity
from reviewer_agent.agents.reviewer_impact import ReviewerSocietalImpact
from reviewer_agent.agents.reviewer_related import ReviewerRelatedWork
from reviewer_agent.agents.leader import merge_points, enforce_grounding, update_review_with_rebuttals
from reviewer_agent.agents.author import rebut
from reviewer_agent.agents.router import SectionBasedRouter, DynamicRouter
from reviewer_agent.services.citations import fetch_top_related
from reviewer_agent.NLPEER_dataset import load_emnlp_paper, get_paper_by_id

from reviewer_agent.llm.constants import LLMModels, LLMTypes

REVIEWER_CLASSES = {
    "ReviewerMethods": ReviewerMethods,
    "ReviewerNovelty": ReviewerNovelty,
    "ReviewerClaimsEvidence": ReviewerClaimsEvidence,
    "ReviewerReproducibility": ReviewerReproducibility,
    "ReviewerEthicsLicensing": ReviewerEthicsLicensing,
    "ReviewerFigures": ReviewerFigures,
    "ReviewerTables": ReviewerTables,
    "ReviewerClarity": ReviewerClarity,
    "ReviewerSocietalImpact": ReviewerSocietalImpact,
    "ReviewerRelatedWork": ReviewerRelatedWork,
}

def run_reviewer_task(facet, route_info, cfg, model_name, model_type):
    """Run a single reviewer task - designed to be thread-safe"""
    cls_name = cfg.reviewers_for_facets.get(facet)
    agent_cls = REVIEWER_CLASSES.get(cls_name)
    spans_text = route_info.get("text", "")
    
    if not spans_text.strip() or not agent_cls:
        return []
    
    # Create a new LLM client for this thread to avoid sharing issues
    llm = LLMClient(model_name=model_name, model_type=model_type)
    agent = agent_cls(llm, cfg)
    
    try:
        paper = route_info.get("paper")  # We'll pass paper in route_info
        points = agent.review(paper, spans_text)
        print(f"✓ Completed {facet} reviewer ({len(points)} points)")
        return points
    except Exception as e:
        print(f"✗ Error in {facet} reviewer: {e}")
        return []

def run_reviewers_parallel(routed, paper, cfg, model_name, model_type, max_workers=4):
    """Run reviewer agents in parallel"""
    # Prepare tasks
    tasks = []
    for facet, route_info in routed.items():
        # Add paper to route_info for the worker function
        route_info_with_paper = route_info.copy()
        route_info_with_paper["paper"] = paper
        tasks.append((facet, route_info_with_paper, cfg, model_name, model_type))
    if not tasks:
        return []
    
    print(f"Running {len(tasks)} reviewers with {max_workers} workers...")
    
    all_points = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_facet = {
            executor.submit(run_reviewer_task, *task): task[0] 
            for task in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_facet):
            facet = future_to_facet[future]
            try:
                points = future.result()
                all_points.extend(points)
            except Exception as e:
                print(f"✗ Exception in {facet} reviewer: {e}")
    
    return all_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_id", type=str, help="Paper ID to process from EMNLP23 dataset (default: 100)", default="100")
    ap.add_argument("--emnlp_data", type=str, help="Path to EMNLP23 data directory", 
                    default="/Users/ehabba/Downloads/EMNLP23/data/")
    # venue flags removed; venue-agnostic pipeline
    ap.add_argument("--model", type=str, default=LLMModels.GEMINI_2_0_FLASH_LITE.value)
    # Ablation flags
    ap.add_argument("--routing", type=str, choices=["dynamic", "all"], default="dynamic", help="Facet routing: dynamic (selected facets) or all (run all reviewers).")
    ap.add_argument("--skip_related", action="store_true", help="Disable Related Work reviewer.")
    ap.add_argument("--skip_rebuttal", action="store_true", help="Disable rebuttal/verify/revise loop.")
    ap.add_argument("--skip_grounding", action="store_true", help="Do not drop ungrounded points.")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers for reviewer execution (default: 4).")
    ap.add_argument("--output_dir", type=str, help="Custom output directory (default: evaluation/results/runs)")
    ap.add_argument("--force", action="store_true", help="Force regeneration even if review already exists")
    args = ap.parse_args()

    cfg = Config()
    llm = LLMClient(model_name=args.model, model_type=LLMTypes.GEMINI)

    # Load paper from EMNLP23 dataset
    print(f"Loading paper {args.paper_id} from EMNLP23 dataset...")
    paper = load_emnlp_paper(args.paper_id, args.emnlp_data)
    
    if not paper:
        print(f"Error: Could not load paper {args.paper_id} from {args.emnlp_data}")
        print("Make sure the paper ID exists and the data directory is correct.")
        return
        
    paper = tag_facets(paper)

    # Fetch top related works by citation for Related Work reviewer
    top_related = fetch_top_related(paper, top_k=3) if not args.skip_related else []

    # Section-based routing: select facets and sections based on clear rules
    router = SectionBasedRouter(cfg)
    if args.routing == "dynamic":
        routed = router.route(paper)
    else:
        # Build routed dict for all facets using full paper text
        routed = {}
        full_text = "\n\n".join([f"## {s.name}\n{s.text}" for s in paper.sections])
        for f in cfg.facets:
            routed[f] = {"sections": [s.name for s in paper.sections], "text": full_text}

    # Run selected reviewers in parallel
    all_points = run_reviewers_parallel(routed, paper, cfg, args.model, LLMTypes.GEMINI, args.workers)

    # Always run Related Work reviewer once with global context + top related papers
    if top_related and not args.skip_related:
        rw_agent = ReviewerRelatedWork(llm, cfg)
        # Use Intro/Related spans text if available, else first routed novelty text or first section
        intro_related_texts = []
        for sec in paper.sections:
            nm = (sec.name or "").lower()
            if "related work" in nm:
                intro_related_texts.append(sec.text)
                break
        if intro_related_texts:
            rw_text = "\n\n".join(intro_related_texts)
            rw_points = rw_agent.review(paper, rw_text, related=top_related)
            all_points.extend(rw_points)

    # Save raw reviewer points before merging
    raw_points = all_points.copy()

    # Merge and ground with LLM enhancement
    review = merge_points(all_points, Rubric(), llm=llm, paper=paper)
    if not args.skip_grounding:
        review = enforce_grounding(review)

    # Save original review before rebuttal
    original_review = review
    
    # Generate author rebuttal and update review
    if not args.skip_rebuttal:
        print("Generating author rebuttal...")
        rebuttal = rebut(review.weaknesses + review.suggestions, paper=paper, llm=llm)
        print("Generated comprehensive rebuttal")
        
        print("Updating review based on rebuttal...")
        updated_review = update_review_with_rebuttals(review, rebuttal, llm=llm, paper=paper)
        print("Review updated")
    else:
        rebuttal = None
        updated_review = None

    # Save outputs with generic naming (no timestamp for caching)
    # model_short = args.model.replace("gemini-", "").replace("gpt-", "").replace("-", "_")
    
    # Create descriptive directory name
    config_flags = []
    if args.skip_rebuttal:
        config_flags.append("no_rebuttal")
    if args.skip_related:
        config_flags.append("no_related")
    if args.skip_grounding:
        config_flags.append("no_grounding")
    if args.routing != "dynamic":
        config_flags.append(f"routing_{args.routing}")
    
    config_str = "_".join(config_flags) if config_flags else "default"
    
    # Use custom output directory if provided, otherwise default
    base_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path("evaluation/results/runs")
    outdir = base_dir / f"paper_{args.paper_id}_{args.model}_{config_str}"
    
    # Check if review already exists (skip if it does, unless --force is used)
    if not args.force and outdir.exists() and (outdir / "review_original.json").exists():
        print(f"✓ Review already exists for paper {args.paper_id} with config {config_str}")
        print(f"Skipping generation. Use --force to regenerate.")
        return
    
    outdir.mkdir(parents=True, exist_ok=True)

    # Save raw reviewer points (before merging)
    raw_points_data = [
        {
            "kind": p.kind,
            "text": p.text,
            "grounding": p.grounding,
            "facet": p.facet
        }
        for p in raw_points
    ]
    with open(outdir / "reviewer_points_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_points_data, f, indent=2)

    # Save original review (before rebuttal)
    with open(outdir / "review_original.json", "w", encoding="utf-8") as f:
        f.write(original_review.model_dump_json(indent=2))

    # Save updated review (after rebuttal)
    if updated_review is not None:
        with open(outdir / "review_updated.json", "w", encoding="utf-8") as f:
            f.write(updated_review.model_dump_json(indent=2))

    # Save rebuttal
    if rebuttal is not None:
        with open(outdir / "rebuttal.txt", "w", encoding="utf-8") as f:
            f.write(rebuttal)

    # Save human reviews for comparison
    paper_data = get_paper_by_id(args.paper_id, args.emnlp_data)
    if paper_data and paper_data.get('reviews'):
        human_reviews = paper_data['reviews']
        with open(outdir / "human_reviews.json", "w", encoding="utf-8") as f:
            json.dump(human_reviews, f, indent=2)
        print(f"Saved {len(human_reviews)} human reviews for comparison")

    files_saved = ["reviewer_points_raw.json", "review_original.json"]
    if updated_review is not None:
        files_saved.append("review_updated.json")
    if rebuttal is not None:
        files_saved.append("rebuttal.txt")
    files_saved.append("human_reviews.json")
    
    print(f"Saved to {outdir}/")
    print(f"Files: {', '.join(files_saved)}")


if __name__ == "__main__":
    main()
