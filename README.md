# Reviewer Agent MVP

A modular, facet-routed peer review agent that produces structured reviews with dynamic expert routing, citation-based related work analysis, and rebuttal simulation.

## Features

- **Dynamic Facet Routing**: Automatically selects relevant expertise facets based on paper content
- **Multi-Agent Reviewers**: Specialized reviewers for methods, novelty, claims/evidence, reproducibility, ethics, figures, clarity, and societal impact
- **Related Work Analysis**: Fetches and compares against top-cited papers via Crossref API
- **Rebuttal Simulation**: Author rebuttal → verification → review revision loop
- **Evaluation Tools**: LLM-as-judge comparison and sentence-level semantic similarity metrics
- **Ablation Support**: Enable/disable components for systematic evaluation

## Architecture

```
Paper PDF/JSON → Parse → Tag Facets → Route → Specialized Reviewers → Merge → Rebuttal Loop → Structured Review
```

### Core Components

- **Facet Tagger**: Labels paper sections with expertise facets (methods, novelty, etc.)
- **Dynamic Router**: Selects relevant reviewers based on facet coverage
- **Specialized Reviewers**: 8 facet-specific agents with NeurIPS 2025-aligned prompts
- **Related Work Reviewer**: Compares against top-cited papers from Crossref
- **Leader Agent**: Merges, deduplicates, and enforces grounding
- **Rebuttal Loop**: Author → Verifier → Review revision

## Installation

```bash
pip install pydantic rapidfuzz PyPDF2 requests sentence-transformers streamlit
```

## Usage

### Basic Review Generation

```bash
# From EMNLP dataset (recommended)
python cli.py --paper_id 100 --model gemini-2.5-flash-lite

# From PDF
python cli.py --pdf /path/to/paper.pdf --model gpt-4o-mini

# From JSON
python cli.py --paper data/dummy_paper.json --model dummy
```

### Evaluation System

```bash
# Quick test on a single paper
python test_paper.py --paper_id 100

# Full evaluation on multiple papers
python run_evaluation.py --paper_ids 100 101 102

# For more evaluation options, see evaluation/ directory
```

### Ablation Studies

```bash
# Dynamic routing (default)
python cli.py --pdf paper.pdf --routing dynamic

# All reviewers
python cli.py --pdf paper.pdf --routing all

# Skip components
python cli.py --pdf paper.pdf --skip_related --skip_rebuttal --skip_grounding
```

### Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

Upload a PDF to get a structured review with heuristic scores (specificity, coherence, correctness).

### Evaluation

```bash
# LLM-as-judge comparison
python reviewer_agent/eval/run_eval.py \
  --paper_context paper_ctx.txt \
  --review_a review_a.json \
  --review_b review_b.json \
  --model gpt-4o-mini

# Sentence-level similarity
python reviewer_agent/eval/run_eval.py \
  --sim_pred predicted_review.json \
  --sim_ref reference_review.json \
  --embed_model all-MiniLM-L6-v2
```

## Project Structure

```
reviewer_agent/
├── agents/                    # Reviewer agents
│   ├── reviewer_methods.py    # Methods & reproducibility
│   ├── reviewer_novelty.py    # Novelty & positioning  
│   ├── reviewer_claims.py     # Claims vs evidence
│   ├── reviewer_ethics.py     # Ethics & licensing
│   ├── reviewer_figures.py    # Figures & tables
│   ├── reviewer_clarity.py    # Clarity & presentation
│   ├── reviewer_impact.py     # Societal impact
│   ├── reviewer_related.py    # Related work comparison
│   ├── leader.py              # Merge & grounding
│   ├── author.py              # Rebuttal generation
│   ├── verifier.py            # Rebuttal verification
│   └── router.py              # Dynamic routing
├── routing/
│   └── facet_tagger.py        # Section → facet mapping
├── services/
│   └── citations.py           # Crossref integration
├── parsing/
│   └── pdf_to_json.py         # PDF extraction
├── prompts/                   # NeurIPS 2025-aligned prompts
├── eval/                      # Evaluation tools
│   ├── judge.py               # LLM-as-judge
│   ├── similarity.py          # Semantic similarity
│   └── run_eval.py            # Evaluation CLI
├── schemas.py                 # Data models
└── config.py                  # Configuration
```

## Facet Taxonomy

The system uses 8 reusable expertise facets:

- **methods**: Statistical soundness, experimental design, reproducibility
- **novelty**: Originality, positioning, comparative evidence  
- **claims_vs_evidence**: Claim support, evidence quality
- **reproducibility**: Code/data availability, seeds, variance
- **ethics_licensing**: Dataset licensing, consent, privacy
- **figures_tables**: Visual quality, caption accuracy
- **clarity_presentation**: Organization, writing clarity
- **societal_impact**: Risks, misuse, broader impacts

## Cost Estimation (GPT-4o mini)

- **Input**: ~27,000 tokens per review
- **Output**: ~4,400 tokens per review  
- **Cost**: ~$0.007 (0.7 cents) per review

## Output Format

Reviews include:
- **Summary**: Neutral paper summary
- **Strengths/Weaknesses/Suggestions**: Grounded bullet points
- **Questions**: 3-5 actionable questions with score criteria
- **Limitations**: Coverage of limitations and societal impact
- **Ethics Flag**: Whether ethics review needed
- **Ratings**: Quality, clarity, significance, originality (1-4), overall (1-6), confidence (1-5)

## Extending the System

1. **Add Facets**: Update `config.py` facets and create new reviewer agents
2. **Custom Prompts**: Modify `prompts/*.txt` files for different venues
3. **LLM Backend**: Implement `llm/base.py::LLMClient.generate()`
4. **PDF Parser**: Replace `parsing/pdf_to_json.py` with GROBID or similar
5. **Evaluation**: Add metrics to `eval/metrics.py` and datasets to `eval/datasets.py`

## Research Applications

This MVP supports the research agenda outlined in the original specification:
- **Dynamic Expert Routing**: Principled facet taxonomy vs ad-hoc experts
- **Section-Level Routing**: Avoids paragraph-level fragmentation  
- **Adversarial Rebuttal**: Author must cite evidence, verifier checks citations
- **Multi-Lens Evaluation**: Coverage, specificity, decision calibration
- **Robust Parsing**: GROBID integration for structure preservation

## License

MIT License - see LICENSE file for details.