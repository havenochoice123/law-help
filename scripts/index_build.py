"""Build vector and keyword indexes from embedding JSONL."""
import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "outputs" / "embeddings" / "embeddings.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "indexes"
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{1}|[A-Za-z0-9_]+")


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_embeddings(path: str) -> Path:
    candidate = resolve_path(path)
    if candidate.is_dir():
        candidate = candidate / "embeddings.jsonl"
    return candidate


def tokenize(text: str) -> list[str]:
    compact_tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    terms = list(compact_tokens)
    chinese_chars = [token for token in compact_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    for n in (2, 3, 4):
        for index in range(0, max(0, len(chinese_chars) - n + 1)):
            terms.append("".join(chinese_chars[index:index + n]))
    return terms


def read_embeddings(path: Path) -> tuple[list[dict], "object"]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required to build vector indexes.") from error

    metadata = []
    vectors = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            embedding = row.pop("embedding", None)
            if not embedding:
                raise ValueError(f"Missing embedding at {path}:{line_number}")
            metadata.append(row)
            vectors.append(embedding)

    if not vectors:
        raise ValueError(f"No embeddings found in {path}")
    matrix = np.asarray(vectors, dtype="float32")
    return metadata, matrix


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_vector_index(out_dir: Path, matrix: "object") -> str:
    try:
        import faiss
    except ImportError:
        import numpy as np

        np.save(out_dir / "vectors.npy", matrix)
        return "numpy"

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(out_dir / "vectors.faiss"))
    return "faiss"


def build_keyword_index(out_dir: Path, metadata: list[dict]) -> dict:
    doc_freq: Counter[str] = Counter()
    docs = []
    for row in metadata:
        terms = tokenize(row.get("text", ""))
        term_counts = Counter(terms)
        doc_freq.update(term_counts.keys())
        docs.append(
            {
                "id": row["id"],
                "length": sum(term_counts.values()),
                "terms": dict(term_counts),
            }
        )

    doc_count = len(docs)
    avgdl = sum(doc["length"] for doc in docs) / doc_count if doc_count else 0.0
    idf = {
        term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        for term, freq in doc_freq.items()
    }
    keyword_index = {
        "tokenizer": "char-ngram-bm25-v1",
        "doc_count": doc_count,
        "avgdl": avgdl,
        "idf": idf,
        "docs": docs,
    }
    with open(out_dir / "keyword_bm25.json", "w", encoding="utf-8") as file:
        json.dump(keyword_index, file, ensure_ascii=False)
    return {"backend": "json-bm25", "terms": len(idf)}


def write_metadata(out_dir: Path, metadata: list[dict]) -> None:
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as file:
        for row in metadata:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RAG vector and keyword indexes.")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS), help="Embedding JSONL file or directory.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output index directory.")
    args = parser.parse_args()

    embeddings_path = resolve_embeddings(args.embeddings)
    out_dir = resolve_path(args.out)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    recreate_dir(out_dir)
    metadata, matrix = read_embeddings(embeddings_path)
    vector_backend = build_vector_index(out_dir, matrix)
    keyword_stats = build_keyword_index(out_dir, metadata)
    write_metadata(out_dir, metadata)

    config = {
        "embeddings": str(embeddings_path),
        "count": len(metadata),
        "dim": int(matrix.shape[1]),
        "vector_backend": vector_backend,
        "keyword_backend": keyword_stats["backend"],
        "keyword_terms": keyword_stats["terms"],
    }
    with open(out_dir / "index_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print(f"documents: {len(metadata)}")
    print(f"dim: {matrix.shape[1]}")
    print(f"vector_backend: {vector_backend}")
    print(f"keyword_backend: {keyword_stats['backend']}")
    print(f"index_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
