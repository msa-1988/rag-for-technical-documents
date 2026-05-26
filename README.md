# RAG for Technical Documents

Repository: `https://github.com/msa-1988/rag-for-technical-documents`

## Overview

This project is a local-first retrieval-augmented generation application for public research papers. It indexes a PDF corpus, retrieves relevant evidence with a hybrid search strategy, and generates grounded answers with visible citations.

The current demo corpus is a public `AI music generation` paper set, but the application is designed for any small technical-document collection.

## Features

- local inference with `Ollama`
- GPU-backed chat generation
- `ChromaDB` vector search
- keyword fallback for stronger exact-entity retrieval
- source diversity limits to improve comparison answers
- citation-aware answers with inspectable context
- Streamlit interface for interactive querying

## Architecture

1. PDFs are loaded from `data/input/`
2. pages are split into overlapping chunks
3. chunks are embedded with `nomic-embed-text`
4. embeddings are stored in `ChromaDB`
5. chunk text is exported for lightweight keyword retrieval
6. vector and keyword hits are merged into a grounded context set
7. `phi3:mini` generates the final answer with citations

## Tech Stack

- `Python`
- `Streamlit`
- `LangChain`
- `Ollama`
- `ChromaDB`
- `PyPDF`

## Usage

### Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
./scripts/check_ollama_gpu.sh
./scripts/run_safe_local.sh
```

Then:

1. place public PDFs in `data/input/`
2. open `http://localhost:8501`
3. click `Build / Refresh Index`
4. ask questions about the indexed papers
5. inspect the answer, sources, and retrieved context

### Validation

```bash
python3 scripts/smoke_test_pipeline.py
```

## Public Demo

The application is configured for localhost by default. If you need a temporary public demo, set a real `APP_ACCESS_PASSWORD` in `.env` and run:

```bash
./scripts/start_secure_public_demo.sh
```

This keeps the app password-protected and only exposes it through a temporary tunnel for the duration of the demo.

## Repository Layout

- `app/`: Streamlit app and pipeline code
- `data/input/`: local PDF input folder, not tracked in Git
- `scripts/`: runtime, validation, and demo helpers
- `.streamlit/config.toml`: safe local server defaults

## Security

- the app binds to `127.0.0.1` by default
- `data/input/` is git-ignored
- no private or personal documents are included in the repository
- public sharing is opt-in and password-protected
