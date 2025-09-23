from typing import List, Tuple, Dict
import re

try:
    import requests
except Exception:  # optional dependency
    requests = None

from ..schemas import Paper


def extract_citation_strings(paper: Paper) -> List[str]:
    """Extract raw citation strings from a 'References'/'Bibliography' section if present.
    Fallback: heuristic detection of reference-like lines (year patterns, brackets).
    """
    ref_texts: List[str] = []
    for sec in paper.sections:
        name = (sec.name or "").strip().lower()
        if name.startswith("references") or name.startswith("bibliography"):
            ref_texts.extend([ln.strip() for ln in sec.text.splitlines() if ln.strip()])
    if ref_texts:
        return _merge_wrapped_citations(ref_texts)

    # Heuristic: collect lines that look like citations (e.g., "[12]", "(2019)")
    pattern = re.compile(r"(\[[0-9]+\]|\(19[0-9]{2}\)|\(20[0-9]{2}\))")
    cand: List[str] = []
    for sec in paper.sections:
        for ln in sec.text.splitlines():
            if pattern.search(ln):
                cand.append(ln.strip())
    return _merge_wrapped_citations(cand)


def _merge_wrapped_citations(lines: List[str]) -> List[str]:
    """Merge lines that belong to the same reference by simple bullet/number cues."""
    merged: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if re.match(r"^\s*(\[[0-9]+\]|[0-9]+\.|•)\s+", ln) and buf:
            merged.append(" ".join(buf))
            buf = [ln]
        else:
            if not buf:
                buf = [ln]
            else:
                # continuation line
                buf.append(ln)
    if buf:
        merged.append(" ".join(buf))
    # Trim
    return [re.sub(r"\s+", " ", m).strip() for m in merged if m.strip()]


def _simple_relevance_score(query_text: str, citation_text: str) -> float:
    q = set(re.findall(r"[a-zA-Z0-9]+", query_text.lower()))
    c = set(re.findall(r"[a-zA-Z0-9]+", citation_text.lower()))
    if not q or not c:
        return 0.0
    inter = len(q & c)
    denom = len(q) + len(c)
    return (2.0 * inter) / denom


def rank_citations(paper: Paper, citation_strings: List[str], top_k: int = 3) -> List[str]:
    # Use title + first 2 sections as query context
    context_parts: List[str] = [paper.title]
    for sec in paper.sections[:2]:
        context_parts.append(sec.text[:1000])
    ctx = "\n".join(context_parts)
    scored = [(c, _simple_relevance_score(ctx, c)) for c in citation_strings]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k] if c.strip()]


def fetch_metadata_via_crossref(citation_text: str) -> Dict[str, str]:
    """Query Crossref works API using the citation text as a query.
    Returns dict with keys: title, url, doi, abstract (optional).
    """
    if requests is None:
        return {"title": citation_text[:200], "url": "", "doi": "", "abstract": ""}
    try:
        params = {"query.bibliographic": citation_text, "rows": 1}
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return {"title": citation_text[:200], "url": "", "doi": "", "abstract": ""}
        it = items[0]
        title = (" ".join(it.get("title", []))).strip()
        url = it.get("URL", "")
        doi = it.get("DOI", "")
        abstract = re.sub(r"<[^>]+>", "", it.get("abstract", "") or "")
        return {"title": title or citation_text[:200], "url": url, "doi": doi, "abstract": abstract}
    except Exception:
        return {"title": citation_text[:200], "url": "", "doi": "", "abstract": ""}


def fetch_top_related(paper: Paper, top_k: int = 3) -> List[Dict[str, str]]:
    """Extract citations, rank them, and fetch basic metadata for the top_k.
    Returns a list of dicts with title/url/doi/abstract.
    """
    cits = extract_citation_strings(paper)
    top = rank_citations(paper, cits, top_k=top_k)
    return [fetch_metadata_via_crossref(c) for c in top]


