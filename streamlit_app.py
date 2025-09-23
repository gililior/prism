import io
import json
import datetime
from typing import List, Tuple

import streamlit as st

from reviewer_agent.config import Config
from reviewer_agent.llm.base import LLMClient
from reviewer_agent.schemas import Point, Review, Rubric
from reviewer_agent.parsing.pdf_to_json import parse_pdf_file_to_paper
from reviewer_agent.routing.facet_tagger import tag_facets
from reviewer_agent.agents.router import DynamicRouter
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
from reviewer_agent.services.citations import fetch_top_related
from reviewer_agent.eval.metrics import genericity_rate


REVIEWER_CLASSES = {
    "ReviewerMethods": ReviewerMethods,
    "ReviewerNovelty": ReviewerNovelty,
    "ReviewerClaimsEvidence": ReviewerClaimsEvidence,
    "ReviewerReproducibility": ReviewerReproducibility,
    "ReviewerEthicsLicensing": ReviewerEthicsLicensing,
    "ReviewerFiguresTables": ReviewerFiguresTables,
    "ReviewerClarity": ReviewerClarity,
    "ReviewerSocietalImpact": ReviewerSocietalImpact,
}


def _render_md(review: Review) -> str:
    def bullets(points: List[Point]) -> str:
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
**Overall:** {review.overall}  **Confidence:** {review.confidence}
"""
    return md


def _compute_scores(review: Review, rebuttals: List[str], verifications: List[Tuple[str, str]]) -> dict:
    all_texts = [p.text for p in (review.strengths + review.weaknesses + review.suggestions)]
    gen_rate = genericity_rate(all_texts)
    specificity = round(1.0 - gen_rate, 3)

    def has_hook(p: Point) -> bool:
        g = (p.grounding or "")
        return any(tag in g for tag in ["Sec", "Table", "Fig"]) or any(tag in p.text for tag in ["Sec", "Table", "Fig"])

    pts = review.strengths + review.weaknesses + review.suggestions
    grounding_rate = round(sum(1 for p in pts if has_hook(p)) / max(1, len(pts)), 3)

    # correctness proxy: share of rebuttals NOT marked as misunderstanding/missing context AND verified
    ok = 0
    misunderstand = 0
    for status, txt in verifications:
        if status == "OK":
            ok += 1
            if ("MISUNDERSTANDING" in txt) or ("MISSING CONTEXT" in txt):
                misunderstand += 1
    correctness = round(1.0 - (misunderstand / ok) if ok else 0.5, 3)

    return {
        "specificity": specificity,
        "coherence": grounding_rate,
        "correctness_proxy": correctness,
    }


def run_pipeline(pdf_bytes: bytes, model_name: str = "dummy") -> Tuple[Review, dict]:
    cfg = Config()
    llm = LLMClient(model_name=model_name)

    # Save to a temp buffer for parser
    with st.spinner("Extracting PDF..."):
        # PyPDF2 can read from file path; write temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(pdf_bytes)
            temp_path = tf.name
        paper = parse_pdf_file_to_paper(temp_path)

    paper = tag_facets(paper)

    with st.spinner("Fetching related works..."):
        related = fetch_top_related(paper, top_k=3)

    router = DynamicRouter()
    routed = router.route(paper)

    all_points: List[Point] = []
    for facet, route_info in routed.items():
        cls_name = cfg.reviewers_for_facets.get(facet)
        if not cls_name:
            continue
        agent_cls = REVIEWER_CLASSES.get(cls_name)
        if not agent_cls:
            continue
        spans_text = route_info.get("text", "")
        if not spans_text.strip():
            continue
        agent = agent_cls(llm)
        pts = agent.review(paper, spans_text)
        all_points.extend(pts)

    # Related Work reviewer (global)
    if related:
        rw_agent = ReviewerRelatedWork(llm)
        intro_related_texts = []
        for sec in paper.sections:
            nm = (sec.name or "").lower()
            if nm.startswith("introduction") or nm.startswith("related"):
                intro_related_texts.append(sec.text)
        rw_text = "\n\n".join(intro_related_texts) if intro_related_texts else (routed.get("novelty", {}).get("text", paper.sections[0].text if paper.sections else ""))
        rw_points = rw_agent.review(paper, rw_text, related=related)
        all_points.extend(rw_points)

    review = merge_points(all_points, Rubric())
    review = enforce_grounding(review)

    rebuttals = rebut(review.weaknesses + review.suggestions)
    verifications = verify(rebuttals)
    review = revise_review(review, rebuttals, verifications)

    scores = _compute_scores(review, rebuttals, verifications)
    return review, scores


st.set_page_config(page_title="Reviewer Agent MVP", layout="wide")
st.title("Reviewer Agent MVP")
st.caption("Upload a PDF to get a structured, facet-routed review and heuristic scores.")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
with col2:
    model = st.text_input("Model name", value="dummy", help="Backend used by LLMClient. 'dummy' returns canned output.")

if uploaded is not None:
    if st.button("Run Review"):
        pdf_bytes = uploaded.read()
        review, scores = run_pipeline(pdf_bytes, model_name=model)

        st.subheader("Review")
        st.markdown(_render_md(review))

        st.subheader("Scores")
        st.json(scores)

        # Download artifacts
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download review.json", data=review.model_dump_json(indent=2), file_name=f"review_{ts}.json", mime="application/json")

