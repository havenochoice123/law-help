"""Minimal FastAPI RAG demo using the local hybrid retriever."""
import argparse
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from scripts.generation_utils import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL, generate_answer, load_model
    from scripts.rag_prompt import build_rag_messages, build_rag_prompt
    from scripts.retriever import HybridRetriever
except ImportError:
    from generation_utils import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL, generate_answer, load_model
    from rag_prompt import build_rag_messages, build_rag_prompt
    from retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    max_new_tokens: int = 256
    repetition_penalty: float = 1.08
    no_repeat_ngram_size: int = 4
    expand_query: bool = True


def create_app(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    adapter_dir: str | Path | None = None,
    load_generator: bool = False,
) -> FastAPI:
    app = FastAPI(title="ClassDesign RAG Demo")
    retriever = HybridRetriever(index_dir)
    tokenizer = model = None
    if load_generator:
        tokenizer, model = load_model(base_model, adapter_dir=adapter_dir)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "documents": len(retriever.metadata),
            "generator_loaded": model is not None,
        }

    @app.post("/retrieve")
    def retrieve(request: AskRequest) -> dict:
        if request.expand_query:
            contexts = retriever.search_expanded(request.question, top_k=request.top_k)
        else:
            contexts = retriever.search(request.question, top_k=request.top_k)
        return {"question": request.question, "contexts": contexts}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        if request.expand_query:
            contexts = retriever.search_expanded(request.question, top_k=request.top_k)
        else:
            contexts = retriever.search(request.question, top_k=request.top_k)
        prompt = build_rag_prompt(request.question, contexts)
        messages = build_rag_messages(request.question, contexts)
        answer = None
        if model is not None:
            answer = generate_answer(
                tokenizer,
                model,
                prompt=prompt,
                messages=messages,
                max_new_tokens=request.max_new_tokens,
                repetition_penalty=request.repetition_penalty,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
            )
        return {
            "question": request.question,
            "answer": answer,
            "prompt": prompt,
            "contexts": contexts,
        }

    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local RAG demo API.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="Index directory.")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL), help="Base model path.")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR), help="LoRA adapter directory.")
    parser.add_argument("--load-generator", action="store_true", help="Load base model + LoRA adapter for /ask.")
    parser.add_argument("--no-adapter", action="store_true", help="Load only the base model when using --load-generator.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    selected_adapter = None if args.no_adapter else args.adapter_dir
    uvicorn.run(
        create_app(
            args.index,
            base_model=args.base_model,
            adapter_dir=selected_adapter,
            load_generator=args.load_generator,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
