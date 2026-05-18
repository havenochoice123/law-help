"""Run LoRA, RAG, and LoRA+RAG comparisons on an evaluation JSONL file."""
import argparse
import json
from pathlib import Path
import re

try:
    from scripts.generation_utils import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL, generate_answer, load_model, row_question
    from scripts.rag_prompt import build_direct_messages, build_direct_prompt, build_rag_messages, build_rag_prompt
    from scripts.retriever import HybridRetriever
except ImportError:
    from generation_utils import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL, generate_answer, load_model, row_question
    from rag_prompt import build_direct_messages, build_direct_prompt, build_rag_messages, build_rag_prompt
    from retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = PROJECT_ROOT / "data" / "eval" / "test.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "eval" / "rag_compare.jsonl"
CITATION_RE = re.compile(r"\[S\d+\]")


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_modes(modes: str) -> list[str]:
    selected = [item.strip().lower() for item in modes.split(",") if item.strip()]
    allowed = {"base", "lora", "rag", "lora_rag"}
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"Unknown mode(s): {', '.join(unknown)}")
    return selected


def ensure_context_citations(answer: str, contexts: list[dict]) -> str:
    if not answer.strip() or CITATION_RE.search(answer):
        return answer
    citation_count = min(3, len(contexts))
    if citation_count <= 0:
        return answer
    citations = " ".join(f"[S{index}]" for index in range(1, citation_count + 1))
    return f"{answer.rstrip()}\n\n引用片段：{citations}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare base/LoRA/RAG/LoRA+RAG outputs.")
    parser.add_argument("--input", default=str(DEFAULT_EVAL), help="Evaluation JSONL file.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL file.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="RAG index directory.")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL), help="Base model path.")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR), help="LoRA adapter directory.")
    parser.add_argument("--modes", default="lora,rag,lora_rag", help="Comma separated: base,lora,rag,lora_rag.")
    parser.add_argument("--limit", type=int, default=5, help="Limit eval rows for quick tests.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved context count.")
    parser.add_argument("--no-query-expansion", action="store_true", help="Disable legal query expansion for RAG.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and retrieval contexts without loading the model.",
    )
    args = parser.parse_args()

    modes = normalize_modes(args.modes)
    rows = read_jsonl(resolve_path(args.input), limit=args.limit)
    retriever = HybridRetriever(args.index) if any(mode in modes for mode in ("rag", "lora_rag")) else None
    base_generator = None
    lora_generator = None
    if not args.dry_run and any(mode in modes for mode in ("base", "rag")):
        base_generator = load_model(args.base_model, adapter_dir=None, device_map=args.device_map)
    if not args.dry_run and any(mode in modes for mode in ("lora", "lora_rag")):
        lora_generator = load_model(args.base_model, adapter_dir=args.adapter_dir, device_map=args.device_map)

    outputs = []
    for index, row in enumerate(rows, start=1):
        question = row_question(row)
        if retriever and args.no_query_expansion:
            contexts = retriever.search(question, top_k=args.top_k)
        elif retriever:
            contexts = retriever.search_expanded(question, top_k=args.top_k)
        else:
            contexts = []
        direct_prompt = build_direct_prompt(question)
        rag_prompt = build_rag_prompt(question, contexts)
        direct_messages = build_direct_messages(question)
        rag_messages = build_rag_messages(question, contexts)
        result = {
            "index": index,
            "id": row.get("id", f"row-{index}"),
            "question": question,
            "reference": row.get("output", ""),
            "contexts": contexts,
            "prompts": {},
            "answers": {},
        }

        if "base" in modes:
            result["prompts"]["base"] = direct_prompt
            tokenizer, model = base_generator if base_generator else (None, None)
            result["answers"]["base"] = "" if args.dry_run else generate_answer(
                tokenizer, model, prompt=direct_prompt, messages=direct_messages,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        if "lora" in modes:
            result["prompts"]["lora"] = direct_prompt
            tokenizer, model = lora_generator if lora_generator else (None, None)
            result["answers"]["lora"] = "" if args.dry_run else generate_answer(
                tokenizer, model, prompt=direct_prompt, messages=direct_messages,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        if "rag" in modes:
            result["prompts"]["rag"] = rag_prompt
            tokenizer, model = base_generator if base_generator else (None, None)
            answer = "" if args.dry_run else generate_answer(
                tokenizer, model, prompt=rag_prompt, messages=rag_messages,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            result["answers"]["rag"] = ensure_context_citations(answer, contexts)
        if "lora_rag" in modes:
            result["prompts"]["lora_rag"] = rag_prompt
            tokenizer, model = lora_generator if lora_generator else (None, None)
            answer = "" if args.dry_run else generate_answer(
                tokenizer, model, prompt=rag_prompt, messages=rag_messages,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            result["answers"]["lora_rag"] = ensure_context_citations(answer, contexts)

        outputs.append(result)
        print(f"[{index}/{len(rows)}] {result['id']}")

    out_path = resolve_path(args.out)
    write_jsonl(out_path, outputs)
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
