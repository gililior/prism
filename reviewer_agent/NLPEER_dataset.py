"""
Enhanced Paper Processing Script for EMNLP23 Dataset

This script processes JSON files from the EMNLP23 dataset, extracting and cleaning
text content organized by sections. It provides improved section detection,
text cleaning, and flexible processing options.

Features:
- Smart section header detection using regex patterns
- Advanced text cleaning to filter out noise and irrelevant content
- Support for processing single papers, samples, or entire dataset
- Structured JSON output with section-based organization
- Progress tracking and detailed statistics

Usage:
    python NLPEER_dataset.py sample          # Process 5 papers for testing
    python NLPEER_dataset.py all            # Process all papers (may take time)
    python NLPEER_dataset.py show [paper_id] # Show detailed content of a paper
    python NLPEER_dataset.py <number>       # Process specific number of papers

Author: Enhanced version for better data processing
"""

import json
import re
import os
from pathlib import Path

def walk(node):
    """Extract all text from nested JSON structure"""
    if isinstance(node, dict):
        if "text" in node and node["text"].strip():
            yield node["text"].strip()
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)

def is_section_header(line):
    """Improved logic to identify section headers"""
    line = line.strip()
    
    # Skip if too long (likely not a section header)
    if len(line) > 100:
        return False
    
    # Skip if contains URLs or paths
    if any(x in line.lower() for x in ['http', 'github', 'www', '.com', '.py', '/']):
        return False
    
    # Skip standalone numbers or very short numeric strings
    if re.match(r'^\d{1,5}$', line):
        return False
    
    # Common section patterns
    patterns = [
        r'^Abstract\s*$',
        r'^Introduction\s*$', 
        r'^Conclusion\s*$',
        r'^References\s*$',
        r'^Acknowledgments?\s*$',
        r'^Appendix\s*[A-Z]?\s*$',
        r'^\d+\s+[A-Z][a-zA-Z\s:&\-]+$',  # "1 Introduction", "2 Methods", etc.
        r'^\d+\.\d+\s+[A-Z][a-zA-Z\s:&\-]+$',  # "2.1 Dataset", etc.
        r'^\d+\.\d+\.\d+\s+[A-Z][a-zA-Z\s:&\-]+$',  # "2.1.1 Subsection", etc.
        r'^[A-Z][a-zA-Z\s&\-]*\s+Considerations?\s*$',  # "Ethical Considerations", etc.
        r'^[A-Z][a-zA-Z\s&\-]*\s+Limitations?\s*$',  # "Limitations", etc.
        r'^\d+\s+[A-Z][a-zA-Z\s&\-]*\s+Considerations?\s*$',  # "8 Ethical Considerations", etc.
        r'^\d+\s+[A-Z][a-zA-Z\s&\-]*\s+Limitations?\s*$',  # "8 Limitations", etc.
        # Appendix patterns
        r'^[A-Z]\s+[A-Z][a-zA-Z\s\-:&]+$',  # "A Hyper-parameters", "B Implementation Details", etc.
        r'^[A-Z]\.\d+\s+[A-Z][a-zA-Z\s\-:&]+$',  # "A.1 Overview", "A.2 Effects of Length Settings", etc.
        r'^[A-Z]\.\d+\.\d+\s+[A-Z][a-zA-Z\s\-:&]+$',  # "A.1.1 Detailed Analysis", etc.
        r'^Appendix\s+[A-Z]\s+[A-Z][a-zA-Z\s\-:&]+$',  # "Appendix A Hyper-parameters", etc.
        r'^Appendix\s+[A-Z]\.\d+\s+[A-Z][a-zA-Z\s\-:&]+$',  # "Appendix A.1 Overview", etc.
    ]
    
    return any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns)

def clean_text(text):
    """Clean and filter irrelevant text"""
    text = text.strip()
    
    # Skip very short lines
    if len(text) < 15:
        return None
    
    # Skip lines that are mostly numbers/symbols
    if re.match(r'^[\d\s\.\,\-\(\)\{\}]+$', text):
        return None
        
    # Skip URLs and file paths
    if any(x in text.lower() for x in ['http', 'github', 'www', '.com', '.py', '/', '.edu']):
        return None
    
    # Skip email addresses
    if re.search(r'\S+@\S+', text):
        return None
    
    # Skip figure/table references and captions that are too short
    # Note: We now handle figure/table captions separately in extract_figure_table_captions
    if re.match(r'^(Figure|Table|Fig\.|Tab\.)\s*\d+', text, re.IGNORECASE):
        # Allow longer figure captions, skip short references
        if len(text) < 50:
            return None
    
    # Skip lines that are just single words repeated
    words = text.split()
    if len(set(words)) == 1 and len(words) > 1:
        return None
    
    # Skip lines with too many special characters
    special_chars = sum(1 for c in text if c in '{}[]().,;:')
    if special_chars > len(text) * 0.3:  # More than 30% special chars
        return None
    
    # Skip author affiliations (numbers followed by institution names)
    if re.match(r'^\d+\s+[A-Z][a-zA-Z\s]+$', text) and len(text) < 80:
        return None
    
    # Skip lines that look like citations or references without content
    if re.match(r'^[\d\s,\-\(\)]+$', text):
        return None
        
    # Skip mathematical expressions that are standalone
    if re.match(r'^[\s\d\+\-\*\/\=\(\)\\]+$', text):
        return None
    
    return text

def process_reviews(review_path):
    """Process reviews JSON file"""
    try:
        with open(review_path, "r", encoding="utf-8") as f:
            reviews_data = json.load(f)
        
        processed_reviews = []
        
        for review in reviews_data:
            processed_review = {
                'rid': review.get('rid', ''),
                'reviewer': review.get('reviewer'),
                'report': {},
                'scores': {},
                'meta': {}
            }
            
            # Process report sections
            if 'report' in review:
                report = review['report']
                for key, value in report.items():
                    if value and isinstance(value, str) and value.strip():
                        processed_review['report'][key] = value.strip()
            
            # Process scores
            if 'scores' in review:
                processed_review['scores'] = review['scores']
            
            # Process meta (excluding complex 'sentences' field)
            if 'meta' in review:
                meta = review['meta']
                for key, value in meta.items():
                    if key != 'sentences':  # Skip sentences as it's complex
                        processed_review['meta'][key] = value
            
            processed_reviews.append(processed_review)
        
        return processed_reviews
    
    except Exception as e:
        print(f"Error processing reviews {review_path}: {e}")
        return []

def extract_figure_table_captions(all_lines):
    """Extract figure and table captions from all text lines"""
    figure_captions = []
    table_captions = []
    
    for line in all_lines:
        line = line.strip()
        
        # Match figure captions: "Figure X:" or "Fig. X:" followed by description
        if re.match(r'^(Figure|Fig\.)\s*\d+\s*:', line, re.IGNORECASE):
            figure_captions.append(line)
        
        # Match table captions: "Table X:" followed by description  
        elif re.match(r'^Table\s*\d+\s*:', line, re.IGNORECASE):
            table_captions.append(line)
    
    return figure_captions, table_captions

def process_paper(json_path):
    """Process a single paper JSON file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_lines = list(walk(data))
        
        # Extract figure and table captions
        figure_captions, table_captions = extract_figure_table_captions(all_lines)
        
        sections = {}
        current_section = None
        
        for line in all_lines:
            if is_section_header(line):
                current_section = line.strip()
                sections[current_section] = []
            elif current_section:
                cleaned_text = clean_text(line)
                if cleaned_text:
                    sections[current_section].append(cleaned_text)
        
        # Add figure captions as a dedicated section if any exist
        if figure_captions:
            sections["Figures"] = figure_captions
            
        # Add table captions as a dedicated section if any exist  
        if table_captions:
            sections["Tables"] = table_captions
        
        return sections
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return {}

def process_paper_with_reviews(paper_dir):
    """Process both paper content and reviews"""
    paper_data = {}
    
    # Process paper content (v2)
    json_path = paper_dir / "v2" / "paper.docling.json"
    if json_path.exists():
        paper_data['sections'] = process_paper(json_path)
    else:
        paper_data['sections'] = {}
    
    # Process reviews (v1)
    review_path = paper_dir / "v1" / "reviews.json"
    if review_path.exists():
        paper_data['reviews'] = process_reviews(review_path)
    else:
        paper_data['reviews'] = []
    
    # Add statistics
    paper_data['stats'] = {
        'total_sections': len(paper_data['sections']),
        'sections_with_content': len([s for s in paper_data['sections'].values() if s]),
        'total_reviews': len(paper_data['reviews']),
        'total_paragraphs': sum(len(s) for s in paper_data['sections'].values())
    }
    
    return paper_data

def save_processed_data(all_papers_data, output_path):
    """Save processed data to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_papers_data, f, indent=2, ensure_ascii=False)
        print(f"Saved processed data to: {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_emnlp_papers(num_papers=5, data_dir=None):
    """
    Load EMNLP23 papers and reviews for use in other modules
    
    Args:
        num_papers: Number of papers to load (None for all)
        data_dir: Path to EMNLP23 data directory
    
    Returns:
        dict: Processed papers data with sections and reviews
    """
    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = DATA_DIR
    
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return {}
    
    # Get all subdirectories
    paper_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    # Determine how many papers to process
    if num_papers is None:
        sample_dirs = sorted(paper_dirs)
    else:
        sample_dirs = sorted(paper_dirs)[:num_papers]
    
    all_papers_data = {}
    
    for paper_dir in sample_dirs:
        # Process both paper and reviews
        paper_data = process_paper_with_reviews(paper_dir)
        
        if paper_data['sections'] or paper_data['reviews']:
            all_papers_data[paper_dir.name] = paper_data
    
    return all_papers_data

def get_paper_by_id(paper_id, data_dir=None):
    """
    Get a specific paper by ID
    
    Args:
        paper_id: Paper ID (string)
        data_dir: Path to EMNLP23 data directory
    
    Returns:
        dict: Paper data with sections and reviews, or None if not found
    """
    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = DATA_DIR
    
    paper_dir = data_path / paper_id
    if not paper_dir.exists():
        return None
    
    return process_paper_with_reviews(paper_dir)

def convert_to_paper_schema(paper_data, paper_id):
    """
    Convert processed paper data to Paper schema format
    
    Args:
        paper_data: Processed paper data from load_emnlp_papers
        paper_id: Paper ID string
    
    Returns:
        dict: Paper data in schema format compatible with existing code
    """
    # Extract title from first section or use paper ID as fallback
    title = paper_id  # Default fallback
    
    # Try to find title in sections
    sections_data = paper_data.get('sections', {})
    if 'Abstract' in sections_data and sections_data['Abstract']:
        # Use first line of abstract as potential title
        first_line = sections_data['Abstract'][0] if sections_data['Abstract'] else ""
        if len(first_line) < 200:  # Reasonable title length
            title = first_line
    
    # Convert sections to schema format
    sections = []
    for section_name, paragraphs in sections_data.items():
        if paragraphs:  # Only include sections with content
            # Handle both list and string formats
            if isinstance(paragraphs, list):
                section_text = "\n\n".join(paragraphs)
            else:
                section_text = str(paragraphs)
            sections.append({
                "name": section_name,
                "text": section_text
            })
    
    # Create paper schema
    paper_schema = {
        "title": title,
        "sections": sections,
        "paper_id": paper_id,
        "reviews": paper_data.get('reviews', []),
        "stats": paper_data.get('stats', {})
    }
    
    return paper_schema

def load_emnlp_paper(paper_id, data_dir=None):
    """
    Load a single EMNLP paper and convert to Paper object
    
    Args:
        paper_id: Paper ID string
        data_dir: Path to EMNLP23 data directory
    
    Returns:
        Paper object or None if not found
    """
    try:
        from reviewer_agent.schemas import Paper
        
        paper_data = get_paper_by_id(paper_id, data_dir)
        if not paper_data:
            return None
        
        paper_schema = convert_to_paper_schema(paper_data, paper_id)
        paper = Paper(
            title=paper_schema["title"],
            sections=[{"name": sec["name"], "text": sec["text"]} for sec in paper_schema["sections"]]
        )
        
        print(f"Loaded paper: {paper.title}")
        print(f"Found {len(paper.sections)} sections and {len(paper_data.get('reviews', []))} reviews")
        
        return paper
    except Exception as e:
        print(f"Error loading paper {paper_id}: {e}")
        return None

def load_emnlp_paper_sample(num_papers=1, data_dir=None):
    """
    Load sample papers from EMNLP and return first one as Paper object
    
    Args:
        num_papers: Number of papers to load
        data_dir: Path to EMNLP23 data directory
    
    Returns:
        Paper object or None if no papers found
    """
    try:
        from reviewer_agent.schemas import Paper
        
        papers_data = load_emnlp_papers(num_papers=num_papers, data_dir=data_dir)
        if not papers_data:
            return None
        
        # Use first paper
        paper_id = list(papers_data.keys())[0]
        paper_data = papers_data[paper_id]
        paper_schema = convert_to_paper_schema(paper_data, paper_id)
        
        paper = Paper(
            title=paper_schema["title"],
            sections=[{"name": sec["name"], "text": sec["text"]} for sec in paper_schema["sections"]]
        )
        
        print(f"Loaded paper {paper_id}: {paper.title}")
        print(f"Found {len(paper.sections)} sections and {len(paper_data.get('reviews', []))} reviews")
        
        return paper
    except Exception as e:
        print(f"Error loading sample papers: {e}")
        return None

def show_sample_paper(paper_id=None, show_reviews=True):
    """Show detailed content of a specific paper for inspection"""
    data_dir = Path("/Users/ehabba/Downloads/EMNLP23/data/")
    
    if paper_id is None:
        # Show first available paper
        paper_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not paper_dirs:
            print("No paper directories found")
            return
        paper_id = sorted(paper_dirs)[0].name
    
    paper_dir = data_dir / paper_id
    
    print(f"\n{'='*60}")
    print(f"DETAILED VIEW - Paper {paper_id}")
    print(f"{'='*60}")
    
    # Process paper and reviews
    paper_data = process_paper_with_reviews(paper_dir)
    
    # Show paper sections
    print(f"\n{'='*40} PAPER CONTENT {'='*40}")
    sections = paper_data['sections']
    for section_name, texts in sections.items():
        if texts:  # Only show sections with content
            print(f"\n=== {section_name} ===")
            # Show first few lines of each section
            for text in texts[:3]:
                print(f"  {text}")
            if len(texts) > 3:
                print(f"  ... and {len(texts)-3} more lines")
    
    # Show reviews if available and requested
    if show_reviews and paper_data['reviews']:
        print(f"\n{'='*40} REVIEWS {'='*40}")
        reviews = paper_data['reviews']
        
        for i, review in enumerate(reviews, 1):
            print(f"\n--- Review {i} (RID: {review['rid']}) ---")
            
            # Show scores
            if review['scores']:
                print("Scores:")
                for score_type, score_value in review['scores'].items():
                    print(f"  {score_type}: {score_value}")
                print()
            
            # Show report sections
            if review['report']:
                print("Report:")
                for report_section, content in review['report'].items():
                    print(f"  {report_section}:")
                    # Truncate long content for display
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    print(f"    {display_content}")
                    print()
            
            # Show meta info
            if review['meta']:
                print("Meta:")
                for meta_key, meta_value in review['meta'].items():
                    print(f"  {meta_key}: {meta_value}")
                print()
    
    elif show_reviews:
        print(f"\n{'='*40} REVIEWS {'='*40}")
        print("No reviews found for this paper.")
    
    # Show statistics
    stats = paper_data['stats']
    print(f"\n{'='*40} STATISTICS {'='*40}")
    print(f"Sections: {stats['total_sections']} ({stats['sections_with_content']} with content)")
    print(f"Paragraphs: {stats['total_paragraphs']}")
    print(f"Reviews: {stats['total_reviews']}")

def create_simple_main():
    """Main function for command line usage with argparse"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process EMNLP23 papers and reviews')
    parser.add_argument('--num_papers', type=int, help='Number of papers to process (default: 5 for sample)')
    parser.add_argument('--output', type=str, help='Output filename', default="processed_papers.json")
    parser.add_argument('--show_paper', type=str, help='Show detailed content of a specific paper ID')
    parser.add_argument('--data_dir', type=str, help='Path to EMNLP23 data directory', 
                       default="/Users/ehabba/Downloads/EMNLP23/data/")
    parser.add_argument('--all', action='store_true', help='Process all papers')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress')
    
    args = parser.parse_args()
    
    # Update data directory if provided
    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    
    if args.show_paper:
        show_sample_paper(args.show_paper)
    elif args.all:
        print("Processing ALL papers - this may take a while!")
        main_simple(num_papers=None, output_file=args.output, verbose=args.verbose)
    else:
        num_papers = args.num_papers or 5
        print(f"Processing {num_papers} papers...")
        main_simple(num_papers=num_papers, output_file=args.output, verbose=args.verbose)

def main_simple(num_papers=5, output_file="processed_papers.json", verbose=False):
    """Simplified main function without statistics"""
    data_dir = DATA_DIR
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return {}
    
    # Get all subdirectories
    paper_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if verbose:
        print(f"Found {len(paper_dirs)} paper directories")
    
    # Determine how many papers to process
    if num_papers is None:
        sample_dirs = sorted(paper_dirs)
        if verbose:
            print("Processing ALL papers...")
    else:
        sample_dirs = sorted(paper_dirs)[:num_papers]
        if verbose:
            print(f"Processing first {num_papers} papers...")
    
    all_papers_data = {}
    successful_papers = 0
    
    for i, paper_dir in enumerate(sample_dirs):
        if verbose:
            print(f"\nProcessing Paper {paper_dir.name} ({i+1}/{len(sample_dirs)})...")
        
        # Process both paper and reviews
        paper_data = process_paper_with_reviews(paper_dir)
        
        if paper_data['sections'] or paper_data['reviews']:
            # Store the processed data
            all_papers_data[paper_dir.name] = paper_data
            successful_papers += 1
            
            # Show progress every 10 papers when processing many
            if num_papers is None and (i + 1) % 10 == 0 and verbose:
                print(f"  Progress: {i+1}/{len(sample_dirs)} papers processed")
    
    if verbose:
        print(f"\nSuccessfully processed: {successful_papers}/{len(sample_dirs)} papers")
    
    if all_papers_data and output_file:
        # Save processed data
        output_path = Path(output_file)
        save_processed_data(all_papers_data, output_path)
        if verbose:
            print(f"Saved to: {output_path}")
    
    return all_papers_data

# Global data directory
DATA_DIR = Path("/Users/ehabba/Downloads/EMNLP23/data/")

if __name__ == "__main__":
    create_simple_main()