
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
from reviewer_agent.agents.reviewer_figures import ReviewerFiguresTables
from reviewer_agent.agents.reviewer_clarity import ReviewerClarity
from reviewer_agent.agents.reviewer_impact import ReviewerSocietalImpact
from reviewer_agent.agents.reviewer_related import ReviewerRelatedWork
from reviewer_agent.agents.leader import merge_points, enforce_grounding, revise_review
from reviewer_agent.agents.author import rebut
from reviewer_agent.agents.verifier import verify
from reviewer_agent.agents.router import SectionBasedRouter, DynamicRouter
from reviewer_agent.services.citations import fetch_top_related
from reviewer_agent.NLPEER_dataset import load_emnlp_paper

from reviewer_agent.llm.constants import LLMModels, LLMTypes

REVIEWER_CLASSES = {
    "ReviewerMethods": ReviewerMethods,
    "ReviewerNovelty": ReviewerNovelty,
    "ReviewerClaimsEvidence": ReviewerClaimsEvidence,
    "ReviewerReproducibility": ReviewerReproducibility,
    "ReviewerEthicsLicensing": ReviewerEthicsLicensing,
    "ReviewerFiguresTables": ReviewerFiguresTables,
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
    ap.add_argument("--model", type=str, default=LLMModels.GEMINI_2_5_FLASH_LITE.value)
    # Ablation flags
    ap.add_argument("--routing", type=str, choices=["dynamic", "all"], default="dynamic", help="Facet routing: dynamic (selected facets) or all (run all reviewers).")
    ap.add_argument("--skip_related", action="store_true", help="Disable Related Work reviewer.")
    ap.add_argument("--skip_rebuttal", action="store_true", help="Disable rebuttal/verify/revise loop.")
    ap.add_argument("--skip_grounding", action="store_true", help="Do not drop ungrounded points.")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers for reviewer execution (default: 4).")
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

    # Merge and ground with LLM enhancement
    review = merge_points(all_points, Rubric(), llm=llm, paper=paper)
    if not args.skip_grounding:
        review = enforce_grounding(review)

    # Enhanced rebuttal loop with LLM
    if not args.skip_rebuttal:
        print("Running rebuttal loop...")
        rebuttals = rebut(review.weaknesses + review.suggestions, paper=paper, llm=llm)
        print(f"Generated {len(rebuttals)} rebuttals")
        ver = verify(rebuttals, llm=llm)
        print(f"Verified rebuttals: {len([v for v in ver if v[0] == 'OK'])} OK, {len([v for v in ver if v[0] == 'WEAK'])} WEAK, {len([v for v in ver if v[0] == 'UNVERIFIED'])} UNVERIFIED")
        review = revise_review(review, rebuttals, ver)
        print("Review revised based on rebuttals")
    else:
        rebuttals = []
        ver = []

    # Save outputs
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path("runs") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "review.json", "w", encoding="utf-8") as f:
        f.write(review.model_dump_json(indent=2))
    with open(outdir / "review.md", "w", encoding="utf-8") as f:
        f.write(render_md(review))

    with open(outdir / "rebuttals.txt", "w", encoding="utf-8") as f:
        for status, r in ver:
            f.write(f"[{status}] {r}\n")

    print(f"Saved to {outdir}/")

def render_md(review):
    def bullets(points):
        return "\n".join([f"- {p.text} ({p.grounding})" for p in points])
    md = f"""# Structured Review

**Summary**  
{review.summary}

**Strengths**  
{bullets(review.strengths)}

**Weaknesses**  
{bullets(review.weaknesses)}

**Suggestions**  
{bullets(review.suggestions)}

**Scores**  
{review.scores}  
**Overall:** {review.overall} &nbsp;&nbsp; **Confidence:** {review.confidence}
"""
    return md

if __name__ == "__main__":
    main()
