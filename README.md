
# Reviewer Agent — MVP Scaffold

This is a minimal, modular scaffold for a **facet-routed reviewer agent with rebuttal simulation**.

## What you can do right now
- Run a toy pipeline on a dummy "paper" (JSON) and get a structured review.
- Swap in your own LLM backend by implementing `llm/base.py::LLMClient.generate`.
- Extend facets, reviewers, and rubrics without changing the core loop.

## Project layout
```
reviewer_agent/
  __init__.py
  config.py
  schemas.py
  llm/
    __init__.py
    base.py
  parsing/
    __init__.py
    pdf_to_json.py          # stub — replace with GROBID/PyMuPDF
  routing/
    __init__.py
    facet_tagger.py
  agents/
    __init__.py
    base.py
    reviewer_methods.py
    reviewer_novelty.py
    leader.py
    author.py
    verifier.py
  prompts/
    reviewer_methods.txt
    reviewer_novelty.txt
    leader_merge.txt
    author_rebuttal.txt
    verifier_check.txt
  eval/
    __init__.py
    datasets.py             # stubs for PeerRead/NLPeer/ReviewCritique
    metrics.py              # simple coverage/genericity heuristics
    run_eval.py
cli.py                      # end-to-end toy run
data/
  dummy_paper.json
```

## Quickstart
1. **Install**: `pip install pydantic==2.* rapidfuzz`
2. **Implement your LLM** in `reviewer_agent/llm/base.py` (OpenAI/Anthropic/etc).
3. **Run**:
```bash
python cli.py --paper data/dummy_paper.json --venue ICLR
```

The output review (JSON + Markdown) will be saved under `runs/<timestamp>/`.

## Swapping in a real parser
Replace `parsing/pdf_to_json.py` with GROBID+pdffigures2 or PyMuPDF. Make sure to fill the `Paper` schema with:
- `sections`: list of `{name, text}`
- `figures`: `{id, caption, mentions:[...]}`
- `tables`: same as figures
