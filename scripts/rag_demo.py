"""Minimal FastAPI RAG demo using the local hybrid retriever."""
import argparse
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from scripts.retriever import HybridRetriever
except ImportError:
    from retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{item['id']}] {item['text']}" for item in contexts
    )
    return (
        "你是法律领域问答助手。请只依据给定知识片段回答问题；"
        "如果知识片段不足以支持结论，请说明无法从当前知识库确认。\n\n"
        f"知识片段：\n{context_text}\n\n"
        f"问题：{question}\n\n"
        "回答要求：给出简洁结论，并在末尾列出引用片段 id。"
    )


def create_app(index_dir: str | Path = DEFAULT_INDEX_DIR) -> FastAPI:
    app = FastAPI(title="ClassDesign RAG Demo")
    retriever = HybridRetriever(index_dir)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(retriever.metadata)}

    @app.post("/retrieve")
    def retrieve(request: AskRequest) -> dict:
        contexts = retriever.search(request.question, top_k=request.top_k)
        return {"question": request.question, "contexts": contexts}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        contexts = retriever.search(request.question, top_k=request.top_k)
        return {
            "question": request.question,
            "prompt": build_prompt(request.question, contexts),
            "contexts": contexts,
        }

    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local RAG demo API.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="Index directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.index), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
