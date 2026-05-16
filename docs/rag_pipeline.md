# RAG Knowledge Base Pipeline

This project uses the DISC-Law-SFT triplet `reference` field as the RAG
knowledge source. Do not build the knowledge base from `output`, because that
would leak target answers into retrieval and evaluation.

## Build

Generate embeddings from reference chunks:

```powershell
python scripts/build_embeddings.py --chunks-dir data/knowledge_chunks --out outputs/embeddings
```

Build the vector and keyword indexes:

```powershell
python scripts/index_build.py --embeddings outputs/embeddings --out outputs/indexes
```

Current offline defaults:

- Embedding backend: `hashing-char-ngram-v1`
- Vector backend: FAISS if installed, otherwise numpy matrix search
- Keyword backend: local BM25 JSON index

If a local sentence-transformers embedding model is available, use:

```powershell
python scripts/build_embeddings.py --chunks-dir data/knowledge_chunks --out outputs/embeddings --embed-model path/to/local/embedding-model
```

## Verify Retrieval

```powershell
python scripts/retriever.py "放火罪怎么处罚" --index outputs/indexes --top-k 3
python scripts/retriever.py "拒不执行判决裁定罪量刑" --index outputs/indexes --top-k 3
```

The retriever combines vector scores and BM25 scores. The default vector weight
is `alpha=0.25`, because legal retrieval benefits from exact statute and charge
matching.

## API Demo

After installing API dependencies:

```powershell
pip install fastapi uvicorn pydantic
python scripts/rag_demo.py --index outputs/indexes --port 8000
```

Endpoints:

- `GET /health`
- `POST /retrieve` with `{"question":"放火罪怎么处罚","top_k":5}`
- `POST /ask` returns the optimized RAG prompt plus retrieved contexts

`/ask` currently returns the prompt and contexts. The next integration step is
to connect this prompt to the loaded LoRA adapter for answer generation.
