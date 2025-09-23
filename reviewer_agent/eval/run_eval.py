
import argparse, json
from pathlib import Path
from .judge import judge_compare
from .similarity import sentence_level_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_context", type=str, required=True, help="Path to a text file with paper context (title+abstract or intro snippet).")
    ap.add_argument("--review_a", type=str, required=True, help="Path to first review text (json or md).")
    ap.add_argument("--review_b", type=str, required=True, help="Path to second review text (json or md).")
    ap.add_argument("--model", type=str, default="dummy")
    ap.add_argument("--sim_pred", type=str, help="Path to predicted review (json or text) for similarity eval.")
    ap.add_argument("--sim_ref", type=str, help="Path to reference review (json or text) for similarity eval.")
    ap.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for similarity")
    args = ap.parse_args()

    paper_ctx = Path(args.paper_context).read_text(encoding="utf-8")

    def read_review(p: str) -> str:
        txt = Path(p).read_text(encoding="utf-8")
        # if JSON, try to condense to a plain text summary
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and any(k in obj for k in ["summary","strengths","weaknesses","suggestions"]):
                parts = []
                parts.append(obj.get("summary", ""))
                for k in ["strengths","weaknesses","suggestions"]:
                    if k in obj and isinstance(obj[k], list):
                        parts.extend([f"- {it.get('text','')} ({it.get('grounding','')})" if isinstance(it, dict) else str(it) for it in obj[k]])
                return "\n".join([p for p in parts if p])
        except Exception:
            pass
        return txt

    ra = read_review(args.review_a)
    rb = read_review(args.review_b)
    result = judge_compare(paper_ctx, ra, rb, model_name=args.model)
    print(json.dumps({"judge": result}, indent=2))

    if args.sim_pred and args.sim_ref:
        sim = sentence_level_similarity(args.sim_pred, args.sim_ref, embed_model=args.embed_model)
        print(json.dumps({"similarity": sim}, indent=2))


if __name__ == "__main__":
    main()
