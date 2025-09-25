
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Span(BaseModel):
    start: int
    end: int
    text: str
    facets: List[str] = []

class Section(BaseModel):
    name: str
    text: str
    spans: List[Span] = []

class Figure(BaseModel):
    id: str
    caption: str
    mentions: List[str] = []

class Table(BaseModel):
    id: str
    caption: str
    mentions: List[str] = []

class Paper(BaseModel):
    title: str
    authors: List[str] = []
    sections: List[Section]
    figures: List[Figure] = []
    tables: List[Table] = []

class Rubric(BaseModel):
    aspects: List[str] = ["originality", "soundness", "clarity", "impact"]
    scale: List[int] = [1,2,3,4,5,6,7,8,9,10]

class Point(BaseModel):
    kind: str  # 'strength' | 'weakness' | 'suggestion'
    text: str
    grounding: Optional[str] = None  # e.g., "Sec 3.2" or "Fig 2"
    facet: Optional[str] = None

class Review(BaseModel):
    summary: str = ""
    strengths: List[Point] = []
    weaknesses: List[Point] = []
    suggestions: List[Point] = []
    scores: Optional[Dict[str, int]] = None
    overall: Optional[int] = None
    confidence: Optional[int] = None
