
from ..schemas import Paper, Section
from typing import List, Dict
import re

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

def parse_pdf_to_paper(json_like: dict) -> Paper:
    """Stub: constructs a Paper from a dict (already JSON).
    Replace with a real PDF parser that returns a Paper.
    """
    sections = [Section(name=s['name'], text=s['text']) for s in json_like['sections']]
    return Paper(
        title=json_like.get('title', 'Untitled'),
        authors=json_like.get('authors', []),
        sections=sections,
        figures=json_like.get('figures', []),
        tables=json_like.get('tables', []),
    )


HEADER_PATTERNS = [
    r"^abstract$",
    r"^introduction$",
    r"^related work$|^background$",
    r"^method$|^methods$|^approach$",
    r"^experiments$|^experimental setup$",
    r"^results$",
    r"^discussion$",
    r"^conclusion$|^conclusions$",
    r"^appendix$|^supplement$",
]

def _extract_pdf_text(pdf_path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is required for PDF extraction. Install with `pip install PyPDF2`.")
    reader = PdfReader(pdf_path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def _guess_title(full_text: str) -> str:
    # Heuristic: first non-empty line up to 200 chars
    for line in full_text.splitlines():
        line = line.strip()
        if len(line) > 0 and len(line) < 200:
            return line
    return "Untitled"

def _split_sections(full_text: str) -> List[Dict[str, str]]:
    lines = full_text.splitlines()
    sections: List[Dict[str, str]] = []
    current_name = "Front Matter"
    current_buf: List[str] = []
    header_regexes = [re.compile(pat, re.IGNORECASE) for pat in HEADER_PATTERNS]

    def flush():
        nonlocal current_name, current_buf
        text = "\n".join(current_buf).strip()
        if text:
            sections.append({"name": current_name.title(), "text": text})
        current_buf = []

    for raw in lines:
        line = raw.strip()
        lowered = line.lower()
        is_header = False
        for rx in header_regexes:
            if rx.match(lowered):
                flush()
                current_name = line
                is_header = True
                break
        if not is_header:
            current_buf.append(raw)
    flush()
    # If only one big section, rename to Body
    if len(sections) == 1:
        sections[0]["name"] = "Body"
    return sections

def pdf_to_json_dict(pdf_path: str) -> dict:
    """Extract a minimal JSON structure from a PDF suitable for parse_pdf_to_paper.

    Note: This is a lightweight heuristic extractor. For production, consider GROBID
    or science-parse for robust structure, references, and captions.
    """
    full_text = _extract_pdf_text(pdf_path)
    title = _guess_title(full_text)
    sections = _split_sections(full_text)
    return {
        "title": title,
        "authors": [],
        "sections": sections,
        "figures": [],
        "tables": [],
    }

def parse_pdf_file_to_paper(pdf_path: str) -> Paper:
    return parse_pdf_to_paper(pdf_to_json_dict(pdf_path))
