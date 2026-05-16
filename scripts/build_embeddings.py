"""Build embeddings for knowledge chunks.

The default backend is a deterministic character n-gram hashing encoder. It is
dependency-light and works offline, which keeps the RAG pipeline reproducible in
classroom environments. If a sentence-transformers model is available locally,
pass --embed-model <model-name-or-path> to use it instead.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "knowledge_chunks"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "embeddings" / "embeddings.jsonl"
HASHING_MODEL_NAME = "hashing-char-ngram-v1"
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_output(path: str) -> Path:
    out = resolve_path(path)
    if out.suffix.lower() != ".jsonl":
        out = out / "embeddings.jsonl"
    return out


def read_chunks(chunks_dir: Path, limit: int | None = None) -> list[dict]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunks = []
    for path in sorted(chunks_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as file:
            chunk = json.load(file)
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            {
                "id": str(chunk.get("id") or path.stem),
                "text": text,
                "title": str(chunk.get("title", "")),
                "source": str(chunk.get("source", "")),
                "qa_id": str(chunk.get("qa_id", "")),
            }
        )
        if limit is not None and len(chunks) >= limit:
            break

    if not chunks:
        raise ValueError(f"No non-empty chunk JSON files found in {chunks_dir}")
    return chunks


def char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> Iterable[str]:
    compact = "".join(text.lower().split())
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            continue
        for index in range(0, len(compact) - n + 1):
            yield compact[index:index + n]
    for match in WORD_RE.finditer(text.lower()):
        yield match.group(0)


def hash_embedding(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in char_ngrams(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def encode_with_hashing(texts: list[str], dim: int) -> list[list[float]]:
    return [hash_embedding(text, dim=dim) for text in texts]


def encode_with_sentence_transformers(
    texts: list[str],
    model_name_or_path: str,
    batch_size: int,
    local_files_only: bool,
) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed. Install dependencies or use "
            "--embed-model hashing."
        ) from error

    try:
        model = SentenceTransformer(model_name_or_path, local_files_only=local_files_only)
    except TypeError:
        model = SentenceTransformer(model_name_or_path)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [[round(float(value), 6) for value in row] for row in embeddings]


def write_embeddings(path: Path, chunks: list[dict], embeddings: list[list[float]], model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            row = dict(chunk)
            row["embedding"] = embedding
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "embedding_file": str(path),
        "model": model_name,
        "count": len(chunks),
        "dim": len(embeddings[0]) if embeddings else 0,
    }
    with open(path.with_name("manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build embeddings for RAG knowledge chunks.")
    parser.add_argument("--chunks-dir", "--input", default=str(DEFAULT_CHUNKS_DIR), help="Input chunk JSON directory.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL file or directory.")
    parser.add_argument(
        "--embed-model",
        default=HASHING_MODEL_NAME,
        help="Use 'hashing' for offline hashing or a sentence-transformers model/path.",
    )
    parser.add_argument("--dim", type=int, default=384, help="Hashing embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=64, help="Sentence-transformers encode batch size.")
    parser.add_argument("--limit", type=int, help="Only encode the first N chunks, for quick tests.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow sentence-transformers to download a model if it is not cached.",
    )
    args = parser.parse_args()

    chunks_dir = resolve_path(args.chunks_dir)
    out_path = resolve_output(args.out)
    chunks = read_chunks(chunks_dir, limit=args.limit)
    texts = [chunk["text"] for chunk in chunks]

    model_key = args.embed_model.lower()
    if model_key in {"hashing", "hash", HASHING_MODEL_NAME}:
        embeddings = encode_with_hashing(texts, dim=args.dim)
        model_name = HASHING_MODEL_NAME
    else:
        embeddings = encode_with_sentence_transformers(
            texts,
            model_name_or_path=args.embed_model,
            batch_size=args.batch_size,
            local_files_only=not args.allow_download,
        )
        model_name = args.embed_model

    write_embeddings(out_path, chunks, embeddings, model_name=model_name)
    print(f"chunks: {len(chunks)}")
    print(f"dim: {len(embeddings[0]) if embeddings else 0}")
    print(f"embeddings: {out_path}")
    print(f"manifest: {out_path.with_name('manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
