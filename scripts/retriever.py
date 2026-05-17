"""Hybrid keyword + vector retriever for the local RAG index."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{1}|[A-Za-z0-9_]+")
CRIME_RE = re.compile(r"[\u4e00-\u9fff]{2,16}罪")
LEGAL_QUERY_RULES = [
    (
        ("正当防卫", "防卫过当", "被抢劫后", "犯罪人杀死", "被害后杀死"),
        "正当防卫 第二十条 防卫过当 正在进行 行凶 杀人 抢劫",
    ),
    (
        ("杀死", "杀人", "砍杀", "杀害", "死亡", "故意杀人"),
        "故意杀人罪 第二百三十二条 杀人 死亡 情节较轻",
    ),
    (
        ("抢劫", "劫取", "暴力", "胁迫"),
        "抢劫罪 第二百六十三条 暴力 胁迫 抢劫公私财物",
    ),
    (
        ("卷烟", "烟草", "假烟", "伪劣卷烟", "软中华", "专卖"),
        "非法经营罪 第二百二十五条 烟草 专卖 违反国家规定 扰乱市场秩序",
    ),
    (
        ("毒品", "冰毒", "甲基苯丙胺", "海洛因", "鸦片"),
        "非法持有毒品罪 第三百四十八条 贩卖毒品罪 第三百四十七条 甲基苯丙胺",
    ),
    (
        ("假冒注册商标", "注册商标"),
        "假冒注册商标罪 第二百一十三条",
    ),
    (
        ("信用卡", "套现", "催收", "恶意透支", "超过三个月"),
        "信用卡诈骗罪 第一百九十六条 恶意透支 数额较大 五年以下有期徒刑",
    ),
    (
        ("焚烧", "玉米秆", "火势", "失控", "森林火灾", "过火面积", "有林地"),
        "失火罪 第一百一十五条 放火罪 过失 森林火灾 重大损失",
    ),
    (
        ("假药", "米非司酮", "伟哥", "食品药品监督", "保健品店"),
        "生产销售假药罪 第一百四十一条 销售假药 三年以下有期徒刑",
    ),
    (
        ("罂粟", "种植", "毒品原植物", "铲除", "销毁"),
        "非法种植毒品原植物罪 第三百五十一条 罂粟 五百株以上 三千株以下",
    ),
]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def tokenize(text: str) -> list[str]:
    compact_tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    terms = list(compact_tokens)
    chinese_chars = [token for token in compact_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    for n in (2, 3, 4):
        for index in range(0, max(0, len(chinese_chars) - n + 1)):
            terms.append("".join(chinese_chars[index:index + n]))
    return terms


def char_ngrams(text: str, min_n: int = 2, max_n: int = 4):
    compact = "".join(text.lower().split())
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            continue
        for index in range(0, len(compact) - n + 1):
            yield compact[index:index + n]
    for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        yield token


def hash_embedding(text: str, dim: int):
    import numpy as np

    vector = np.zeros(dim, dtype="float32")
    for token in char_ngrams(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector.reshape(1, -1)


def minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def build_legal_queries(query: str) -> list[str]:
    queries = []
    for terms, expansion in LEGAL_QUERY_RULES:
        if any(term in query for term in terms):
            queries.append(expansion)

    crime_terms = []
    for term in sorted(set(CRIME_RE.findall(query)), key=len, reverse=True):
        if len(term) > 8 or "的" in term:
            continue
        crime_terms.append(term)
    if crime_terms:
        queries.append(" ".join(crime_terms))

    queries.append(query)
    deduped = []
    seen = set()
    for item in queries:
        normalized = " ".join(item.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


class HybridRetriever:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = resolve_path(str(index_dir))
        self.config = self._load_json(self.index_dir / "index_config.json")
        self.metadata = self._load_metadata()
        self.keyword_index = self._load_json(self.index_dir / "keyword_bm25.json")
        self.vector_backend = self.config.get("vector_backend")
        self.dim = int(self.config["dim"])
        self._load_vector_index()

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_metadata(self) -> list[dict]:
        rows = []
        with open(self.index_dir / "metadata.jsonl", "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _load_vector_index(self) -> None:
        import numpy as np

        if self.vector_backend == "faiss":
            import faiss

            self.faiss = faiss
            self.vector_index = faiss.read_index(str(self.index_dir / "vectors.faiss"))
            self.matrix = None
        else:
            self.faiss = None
            self.vector_index = None
            self.matrix = np.load(self.index_dir / "vectors.npy")

    def search_vector(self, query: str, top_k: int) -> dict[int, float]:
        query_vector = hash_embedding(query, dim=self.dim)
        if self.vector_backend == "faiss":
            scores, indices = self.vector_index.search(query_vector, top_k)
            return {
                int(index): float(score)
                for index, score in zip(indices[0], scores[0], strict=False)
                if int(index) >= 0
            }

        scores = (self.matrix @ query_vector.reshape(-1)).astype("float32")
        candidate_count = min(top_k, len(scores))
        if candidate_count <= 0:
            return {}
        indices = scores.argsort()[-candidate_count:][::-1]
        return {int(index): float(scores[index]) for index in indices}

    def search_keyword(self, query: str, top_k: int) -> dict[int, float]:
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return {}
        idf = self.keyword_index["idf"]
        avgdl = float(self.keyword_index["avgdl"])
        k1 = 1.5
        b = 0.75
        scores = {}
        for index, doc in enumerate(self.keyword_index["docs"]):
            score = 0.0
            doc_len = max(1, int(doc["length"]))
            terms = doc["terms"]
            for term, query_count in query_terms.items():
                freq = terms.get(term, 0)
                if not freq:
                    continue
                term_idf = float(idf.get(term, 0.0))
                denom = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
                score += query_count * term_idf * (freq * (k1 + 1)) / denom
            if score > 0:
                scores[index] = score
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])

    def search(self, query: str, top_k: int = 5, candidate_k: int = 50, alpha: float = 0.25) -> list[dict]:
        vector_scores = self.search_vector(query, top_k=candidate_k)
        keyword_scores = self.search_keyword(query, top_k=candidate_k)
        vector_norm = minmax(vector_scores)
        keyword_norm = minmax(keyword_scores)
        merged_ids = set(vector_norm) | set(keyword_norm)
        scored = []
        for index in merged_ids:
            score = alpha * vector_norm.get(index, 0.0) + (1 - alpha) * keyword_norm.get(index, 0.0)
            row = dict(self.metadata[index])
            row["score"] = round(score, 6)
            row["vector_score"] = round(vector_scores.get(index, 0.0), 6)
            row["keyword_score"] = round(keyword_scores.get(index, 0.0), 6)
            scored.append(row)
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def search_expanded(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        alpha: float = 0.25,
        per_query_k: int = 3,
    ) -> list[dict]:
        primary = []
        secondary = []
        seen_texts = set()
        for query_index, expanded_query in enumerate(build_legal_queries(query)):
            results = self.search(expanded_query, top_k=per_query_k, candidate_k=candidate_k, alpha=alpha)
            kept_for_query = 0
            for result in results:
                text_key = " ".join(result["text"].split())
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                item = dict(result)
                item["expanded_query"] = expanded_query
                item["score"] = round(item["score"] + max(0, 0.02 - query_index * 0.002), 6)
                if kept_for_query == 0:
                    primary.append(item)
                else:
                    secondary.append(item)
                kept_for_query += 1

        ranked = sorted(primary + secondary, key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the local hybrid RAG index.")
    parser.add_argument("query", nargs="?", help="Question/query text.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="Index directory.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.25, help="Vector weight in hybrid scoring.")
    parser.add_argument("--no-expand", action="store_true", help="Disable legal query expansion.")
    args = parser.parse_args()

    if not args.query:
        raise ValueError("Please provide a query.")

    retriever = HybridRetriever(args.index)
    if args.no_expand:
        results = retriever.search(args.query, top_k=args.top_k, candidate_k=args.candidate_k, alpha=args.alpha)
    else:
        results = retriever.search_expanded(args.query, top_k=args.top_k, candidate_k=args.candidate_k, alpha=args.alpha)
    for index, item in enumerate(results, start=1):
        print(f"[{index}] score={item['score']} id={item['id']} source={item.get('source', '')}")
        print(item["text"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
