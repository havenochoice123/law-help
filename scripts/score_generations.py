"""Score generated answers with BLEU-4 and ROUGE-L."""
import argparse
import json
import re
from pathlib import Path

try:
    from scripts.generation_utils import clean_generation_text
except ImportError:
    from generation_utils import clean_generation_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "eval" / "rag_compare.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "eval" / "rag_compare_scores.json"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_modes(rows: list[dict]) -> list[str]:
    modes = set()
    for row in rows:
        modes.update(row.get("answers", {}).keys())
    return sorted(modes)


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    try:
        import sacrebleu
    except ImportError as error:
        raise RuntimeError("BLEU-4 scoring requires sacrebleu. Install sacrebleu first.") from error
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def rouge_l(predictions: list[str], references: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError as error:
        raise RuntimeError("ROUGE-L scoring requires rouge-score. Install rouge-score first.") from error

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references, strict=True)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


def count_cited_answers(predictions: list[str]) -> int:
    citation_pattern = re.compile(r"\[S\d+\]|引用片段")
    return sum(1 for item in predictions if citation_pattern.search(item))


def count_english_answers(predictions: list[str]) -> int:
    english_pattern = re.compile(r"[A-Za-z]{4,}")
    return sum(1 for item in predictions if english_pattern.search(item))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score rag_compare JSONL outputs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Comparison JSONL file.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON score summary.")
    args = parser.parse_args()

    rows = read_jsonl(resolve_path(args.input))
    if not rows:
        raise ValueError("No rows to score.")

    references = [str(row.get("reference", "")) for row in rows]
    summary = {
        "input": str(resolve_path(args.input)),
        "count": len(rows),
        "modes": {},
    }
    for mode in collect_modes(rows):
        predictions = [
            clean_generation_text(str(row.get("answers", {}).get(mode, "")))
            for row in rows
        ]
        summary["modes"][mode] = {
            "bleu4": corpus_bleu(predictions, references),
            "rouge_l": rouge_l(predictions, references),
            "non_empty": sum(1 for item in predictions if item.strip()),
            "cited_answers": count_cited_answers(predictions),
            "english_fragment_answers": count_english_answers(predictions),
            "avg_chars": float(sum(len(item) for item in predictions) / len(predictions)) if predictions else 0.0,
        }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
