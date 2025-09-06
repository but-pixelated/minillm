#`README_RAG.md`
#rag — retrieval augmented generation

this folder implements a **rag pipeline**: chunk documents → embed them → build a vector index → query with llm context.

---

## ⚙️ pipeline steps

### 1. ingest docs
split into chunks + compute embeddings.
```bash
python rag/ingest.py --docs data/raw/ --out rag/index/
input: text/pdf/docs

output: vector embeddings

2. build index

python rag/build_index.py --embeddings rag/index/embeddings.pkl --out rag/index/faiss.index
uses faiss for similarity search.

3. query index

python rag/search.py --query "what is quantum computing?"
returns top-k chunks, which can be prepended to llm prompts.

🔗 integration with llm
the server (src/server_hf.py) can be extended to:

take user prompt
fetch relevant docs via rag/search.py
prepend retrieved text as context
send combined prompt to llm
example prompt build:

user: how does x work?
assistant: "replies"

rag indexes (rag/index/) are large → ignored in git

embeddings model used: sentence-transformers/all-MiniLM-L6-v2 (default)

you can swap embeddings model in ingest.py

📊 roadmap
support pdf/docx ingestion (currently text)

add streaming rag responses

add hybrid search (keywords + embeddings)