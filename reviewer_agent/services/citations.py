from typing import List, Tuple, Dict
import re
import time
import requests
import arxiv
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
    """Merge lines that belong to the same reference by detecting citation start patterns."""
    merged: List[str] = []
    buf: List[str] = []
    
    def _looks_like_citation_start(line: str) -> bool:
        """Check if line looks like the start of a new citation."""
        line = line.strip()
        if not line:
            return False
            
        # Pattern 1: Numbered citations [1], 1., •
        if re.match(r"^\s*(\[[0-9]+\]|[0-9]+\.|•)\s+", line):
            return True
            
        # Pattern 2: Author-year format (e.g., "Smith, J. 2020." or "Smith et al. (2020)")
        # Look for: Name(s) followed by year (19xx or 20xx)
        if re.match(r"^[A-Z][a-zA-Z\s,.-]+\s+(19|20)\d{2}[\.\s]", line):
            return True
            
        # Pattern 3: Author name at start of line (common in bibliography)
        # Must start with capital letter and contain typical name patterns
        if re.match(r"^[A-Z][a-zA-Z]+,\s+[A-Z]", line):
            return True
            
        return False
    
    for ln in lines:
        if _looks_like_citation_start(ln) and buf:
            # Save previous citation and start new one
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
    
    # Trim and clean up
    return [re.sub(r"\s+", " ", m).strip() for m in merged if m.strip()]


def _compute_relevance_score(query_text: str, citation_text: str) -> float:
    """Compute relevance score using multiple factors"""
    
    # Extract words from both texts
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query_text.lower()))
    citation_words = re.findall(r"[a-zA-Z0-9]+", citation_text.lower())
    citation_word_set = set(citation_words)
    
    if not query_words or not citation_words:
        return 0.0
    
    # Factor 1: Weighted term matching (40% of score)
    # Define domain-specific important terms with weights
    important_terms = {
        # Core ML/AI terms
        'deep': 2.0, 'learning': 2.0, 'neural': 2.0, 'network': 2.0, 'networks': 2.0,
        'machine': 1.8, 'artificial': 1.8, 'intelligence': 1.8,
        'transformer': 2.5, 'attention': 2.5, 'bert': 2.0, 'gpt': 2.0,
        'convolutional': 2.0, 'cnn': 2.0, 'lstm': 1.8, 'rnn': 1.8,
        
        # Vision terms
        'vision': 2.0, 'image': 1.8, 'visual': 1.8, 'computer': 1.5,
        'face': 2.5, 'facial': 2.5, 'expression': 2.0, 'emotion': 2.0,
        'recognition': 1.8, 'detection': 1.8, 'classification': 1.5,
        
        # NLP terms
        'language': 2.0, 'text': 1.8, 'nlp': 2.0, 'processing': 1.5,
        'translation': 1.8, 'generation': 1.8, 'understanding': 1.8,
        
        # General research terms
        'analysis': 1.2, 'method': 1.0, 'approach': 1.0, 'model': 1.5,
        'algorithm': 1.3, 'system': 1.0, 'framework': 1.2,
        
        # Evaluation terms
        'evaluation': 1.2, 'benchmark': 1.5, 'dataset': 1.3, 'performance': 1.2
    }
    
    # Calculate weighted overlap
    weighted_matches = 0.0
    total_query_weight = 0.0
    
    for word in query_words:
        weight = important_terms.get(word, 0.5)  # Default weight for other words
        total_query_weight += weight
        if word in citation_word_set:
            weighted_matches += weight
    
    term_score = weighted_matches / total_query_weight if total_query_weight > 0 else 0.0
    
    # Factor 2: Venue prestige (25% of score)
    venue_scores = {
        'nature': 1.0, 'science': 1.0, 'cell': 0.95,
        'nips': 0.9, 'neurips': 0.9, 'advances in neural information processing': 0.9,
        'icml': 0.9, 'iclr': 0.9, 'cvpr': 0.9, 'iccv': 0.9, 'eccv': 0.9,
        'acl': 0.85, 'emnlp': 0.85, 'naacl': 0.8, 'coling': 0.75,
        'aaai': 0.8, 'ijcai': 0.8, 'kdd': 0.8, 'www': 0.75,
        'ieee': 0.7, 'acm': 0.7, 'springer': 0.6,
        'arxiv': 0.5, 'preprint': 0.5,
        'workshop': 0.4, 'poster': 0.3
    }
    
    venue_score = 0.3  # default
    citation_lower = citation_text.lower()
    for venue, score in venue_scores.items():
        if venue in citation_lower:
            venue_score = score
            break
    
    # Factor 3: Recency (20% of score)
    year_match = re.search(r'(19|20)\d{2}', citation_text)
    if year_match:
        year = int(year_match.group())
        # Papers from 2015+ get higher scores, with 2020+ being optimal
        if year >= 2020:
            recency_score = 1.0
        elif year >= 2015:
            recency_score = 0.5 + (year - 2015) * 0.1  # 0.5 to 1.0
        elif year >= 2010:
            recency_score = (year - 2010) * 0.1  # 0.0 to 0.5
        else:
            recency_score = 0.1  # Very old papers get minimal score
    else:
        recency_score = 0.5  # Unknown year
    
    # Factor 4: Citation quality indicators (15% of score)
    # Number of authors (more authors might indicate larger collaboration)
    author_count = len(re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', citation_text))
    author_score = min(1.0, author_count / 6.0)  # Normalize, 6+ authors = max score
    
    # Title informativeness (longer, more specific titles)
    title_match = re.search(r'\.?\s*([A-Z][^.]*?)\.\s*(?:In|Proceedings|Journal|Conference|arXiv)', citation_text)
    if title_match:
        title_length = len(title_match.group(1))
        length_score = min(1.0, title_length / 80.0)  # 80+ chars = max score
    else:
        length_score = 0.5
    
    quality_score = (author_score + length_score) / 2.0
    
    # Combine all factors with weights
    final_score = (
        0.40 * term_score +      # 40% - semantic relevance
        0.25 * venue_score +     # 25% - venue prestige  
        0.20 * recency_score +   # 20% - recency
        0.15 * quality_score     # 15% - citation quality
    )
    
    return final_score


def rank_citations(paper: Paper, citation_strings: List[str], top_k: int = 3) -> List[str]:
    """Rank citations by relevance using multiple factors"""
    # Use title + first 2 sections as query context
    context_parts: List[str] = [paper.title]
    for sec in paper.sections[:2]:
        context_parts.append(sec.text[:1000])
    ctx = "\n".join(context_parts)
    
    # Score each citation using the improved scoring function
    scored = [(c, _compute_relevance_score(ctx, c)) for c in citation_strings]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k citations with non-zero scores
    return [c for c, score in scored[:top_k] if c.strip() and score > 0.1]


def _extract_title_from_citation(citation_text: str) -> str:
    """Extract likely title from citation text"""
    # Common patterns for titles in citations
    patterns = [
        r'\.?\s*([A-Z][^.]*?)\.\s*(?:In|Proceedings|Journal|Conference)',  # Title before "In"
        r'(?:19|20)\d{2}\.?\s*([A-Z][^.]*?)\.\s*(?:In|Proceedings|Journal)',  # Title after year
        r'([A-Z][^.]*?)\.\s*(?:19|20)\d{2}',  # Title before year
    ]
    
    for pattern in patterns:
        match = re.search(pattern, citation_text)
        if match:
            title = match.group(1).strip()
            # Clean up common artifacts
            title = re.sub(r'\s+', ' ', title)
            if len(title) > 10 and not title.endswith(' et al'):
                return title
    
    # Fallback: take text after authors and before year/venue
    parts = citation_text.split('.')
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) > 20 and not re.match(r'^[A-Z][a-z]+ [A-Z]', part):  # Not author names
            if not re.match(r'^\d{4}', part):  # Not year
                return part
    
    return None


def _search_arxiv_for_abstract(title: str) -> str:
    """Search arXiv for abstract using title"""
    if not ARXIV_AVAILABLE:
        return None
        
    try:
        # Try exact title search first
        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        client = arxiv.Client()
        for result in client.results(search):
            # Check if titles are similar
            if title.lower() in result.title.lower() or result.title.lower() in title.lower():
                return result.summary
        
        # Try broader search without quotes
        search2 = arxiv.Search(
            query=title.replace(":", "").replace(".", ""),
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in client.results(search2):
            # More flexible matching
            title_words = set(title.lower().split())
            result_words = set(result.title.lower().split())
            overlap = len(title_words & result_words)
            if overlap >= 2:  # At least 2 words in common
                return result.summary
                
    except Exception:
        pass
    
    return None


def _search_semantic_scholar_for_abstract(title: str) -> str:
    """Search Semantic Scholar for abstract with retry logic"""
    if not requests:
        return None
        
    for attempt in range(2):  # Max 2 attempts
        try:
            if attempt > 0:
                time.sleep(1)  # Brief wait on retry
                
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": title,
                "fields": "title,abstract,authors,year",
                "limit": 3
            }
            
            resp = requests.get(url, params=params, timeout=8)
            
            if resp.status_code == 200:
                data = resp.json()
                papers = data.get('data', [])
                
                for paper in papers:
                    paper_title = paper.get('title', '').lower()
                    search_title = title.lower()
                    
                    # Check for good title match
                    if (search_title in paper_title or paper_title in search_title or
                        len(set(search_title.split()) & set(paper_title.split())) >= 3):
                        abstract = paper.get('abstract')
                        if abstract:
                            return abstract
                            
            elif resp.status_code == 429:  # Rate limited
                continue
            else:
                break
                
        except Exception:
            pass
            
    return None


def _search_openalex_for_abstract(title: str) -> str:
    """Search OpenAlex for abstract"""
    if not requests:
        return None
        
    try:
        url = "https://api.openalex.org/works"
        params = {
            "search": title,
            "per-page": 3
        }
        
        resp = requests.get(url, params=params, timeout=8)
        
        if resp.status_code == 200:
            data = resp.json()
            works = data.get('results', [])
            
            for work in works:
                work_title = work.get('title', '').lower()
                search_title = title.lower()
                
                if (search_title in work_title or work_title in search_title or
                    len(set(search_title.split()) & set(work_title.split())) >= 3):
                    
                    # Try different abstract fields
                    abstract = work.get('abstract')
                    if not abstract:
                        # Reconstruct from inverted index
                        abstract_index = work.get('abstract_inverted_index')
                        if abstract_index:
                            word_positions = []
                            for word, positions in abstract_index.items():
                                for pos in positions:
                                    word_positions.append((pos, word))
                            word_positions.sort()
                            abstract = ' '.join([word for pos, word in word_positions])
                    
                    if abstract:
                        return abstract
                        
    except Exception:
        pass
        
    return None


def _find_abstract_for_citation(citation_text: str) -> str:
    """Try multiple methods to find abstract for a citation"""
    # Extract title
    title = _extract_title_from_citation(citation_text)
    if not title:
        return None
    
    # Try different sources in order of reliability
    # 1. Try arXiv first (often has full abstracts)
    abstract = _search_arxiv_for_abstract(title)
    if abstract:
        return abstract
    
    # 2. Try Semantic Scholar
    abstract = _search_semantic_scholar_for_abstract(title)
    if abstract:
        return abstract
    
    # 3. Try OpenAlex
    abstract = _search_openalex_for_abstract(title)
    if abstract:
        return abstract
    
    return None


def fetch_metadata_via_crossref(citation_text: str) -> Dict[str, str]:
    """Query Crossref works API using the citation text as a query.
    Returns dict with keys: title, url, doi, authors, year, venue, summary.
    Since abstracts are rarely available, we create a summary from available metadata.
    """
    if requests is None:
        return {
            "title": citation_text[:200], 
            "url": "", 
            "doi": "", 
            "authors": "",
            "year": "",
            "venue": "",
            "summary": citation_text[:300]
        }
    
    try:
        params = {"query.bibliographic": citation_text, "rows": 1}
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        
        if not items:
            return {
                "title": citation_text[:200], 
                "url": "", 
                "doi": "", 
                "authors": "",
                "year": "",
                "venue": "",
                "summary": citation_text[:300]
            }
            
        it = items[0]
        
        # Extract basic info
        title = (" ".join(it.get("title", []))).strip()
        url = it.get("URL", "")
        doi = it.get("DOI", "")
        
        # Extract authors
        authors_list = it.get("author", [])
        if authors_list:
            author_names = []
            for author in authors_list[:3]:  # Limit to first 3 authors
                given = author.get("given", "")
                family = author.get("family", "")
                if family:
                    if given:
                        author_names.append(f"{given} {family}")
                    else:
                        author_names.append(family)
            authors = ", ".join(author_names)
            if len(authors_list) > 3:
                authors += " et al."
        else:
            authors = ""
        
        # Extract year
        year = ""
        if "published" in it:
            date_parts = it["published"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])
        elif "issued" in it:
            date_parts = it["issued"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])
        
        # Extract venue
        venue = ""
        container_title = it.get("container-title", [])
        if container_title:
            venue = container_title[0]
        elif "event" in it:
            venue = it["event"].get("name", "")
        
        # Try to find real abstract first
        abstract = _find_abstract_for_citation(citation_text)
        
        # If no abstract found, create a summary from available metadata
        if not abstract:
            summary_parts = []
            if authors:
                summary_parts.append(f"Authors: {authors}")
            if year:
                summary_parts.append(f"Year: {year}")
            if venue:
                summary_parts.append(f"Published in: {venue}")
            if it.get("type"):
                summary_parts.append(f"Type: {it['type'].replace('-', ' ').title()}")
            
            summary = ". ".join(summary_parts) if summary_parts else citation_text[:300]
        else:
            summary = abstract
        
        return {
            "title": title or citation_text[:200], 
            "url": url, 
            "doi": doi, 
            "authors": authors,
            "year": year,
            "venue": venue,
            "summary": summary
        }
        
    except Exception:
        return {
            "title": citation_text[:200], 
            "url": "", 
            "doi": "", 
            "authors": "",
            "year": "",
            "venue": "",
            "summary": citation_text[:300]
        }


def fetch_top_related(paper: Paper, top_k: int = 3) -> List[Dict[str, str]]:
    """Extract citations, rank them, and fetch basic metadata for the top_k.
    Returns a list of dicts with title/url/doi/authors/year/venue/summary.
    """
    cits = extract_citation_strings(paper)
    top = rank_citations(paper, cits, top_k=top_k)
    return [fetch_metadata_via_crossref(c) for c in top]


