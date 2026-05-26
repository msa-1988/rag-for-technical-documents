# 01 - RAG for Technical Documents

This is the first project in the `Learning and building AI skills` roadmap.

## Goal

Build a compact, deployable `RAG` application that can read technical PDFs and answer questions with citations.

The main purpose of this project is to:
- learn the core RAG stack fast
- produce a real GitHub project
- create something demo-able for LinkedIn and interviews

## Why This Project Is First

This is the best first project because it gives you:
- GenAI application experience
- document intelligence experience
- retrieval and grounding fundamentals
- a fast path to deployment
- a strong portfolio story

## Suggested Project Use Case

Use a public, creative topic first:
- AI music generation papers
- open arXiv PDFs
- model papers + review papers

That makes the app:
- public and easy to share
- more interesting to recruiters
- easier to demo on GitHub and LinkedIn
- safe to publish without personal documents

## One-Day MVP Deliverables

By the end of the first build day, the MVP should do this:

1. ingest PDFs
2. split them into chunks
3. create embeddings
4. store them in a vector database
5. accept a user question
6. retrieve relevant chunks with hybrid retrieval
7. generate an answer with citations
8. run through a simple web UI
9. be pushed to GitHub
10. be deployable publicly

## Suggested Stack

- `Python`
- `Streamlit`
- `LangChain`
- `Ollama`
- `Local GPU inference`
- `Local open LLM`
- `ChromaDB`
- `PyPDF`

## Current Retrieval Design

This version uses:
- semantic retrieval from `ChromaDB`
- lightweight keyword retrieval from exported chunks
- source diversity limits so one paper does not dominate the answer

That hybrid design was added after real testing showed that exact model-name comparison questions needed stronger lexical support.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
ollama ps
streamlit run app/streamlit_app.py
```

Then:
- make sure Ollama is installed and running locally
- use the default local chat model `phi3:mini` or change it in `.env`
- put PDFs into `data/input/`
- click `Build / Refresh Index`
- ask real questions from the paper set

GPU note:
- Ollama automatically uses the GPU when available
- this machine has already been verified with `phi3:mini` running at `100% GPU`
- use `ollama ps` to confirm the active processor at any time

Optional local chat model upgrades:
- `ollama pull qwen2.5:3b-instruct`
- `ollama pull phi4-mini`

## Public Demo Path

For this local-first version:
- best day-1 target is a `public tunnel` to the local Streamlit app
- best phase-2 target is a hosted inference deployment

Later, you can:
- deploy a better version on `Azure`
- containerize it
- convert it into an API service
- add a second provider later if you want a provider-agnostic version

## Repo Structure

- `docs/`
  - `PROJECT_PLAN.md`
  - `INPUT_OUTPUT.md`
  - `LEARNING_BASICS.md`
  - `ONE_DAY_EXECUTION.md`
  - `STEP_BY_STEP.md`
  - `LOCAL_OLLAMA_SETUP.md`
  - `TOPIC_MUSIC_GENERATION.md`
  - `PIPELINE_INTERVIEW_GUIDE.md`
  - `PUBLIC_TUNNEL_DEPLOYMENT.md`
- `app/`
  - `streamlit_app.py`
  - `src/`
- `data/input/`
- `data/output/`
- `scripts/`
  - `check_ollama_gpu.sh`
  - `smoke_test_pipeline.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`

## Validation Commands

```bash
./scripts/check_ollama_gpu.sh
python3 scripts/smoke_test_pipeline.py
```

## Definition of Done

The project is good enough to publish when:
- the app answers questions using uploaded PDFs
- the answer shows source chunks or citations
- the README explains architecture and setup
- the repo has screenshots or demo output
- the repo is pushed to GitHub
- the app is publicly accessible

## Next Step

Read the files in `docs/` in this order:

1. `PROJECT_PLAN.md`
2. `LEARNING_BASICS.md`
3. `INPUT_OUTPUT.md`
4. `ONE_DAY_EXECUTION.md`
5. `STEP_BY_STEP.md`
6. `LOCAL_OLLAMA_SETUP.md`
7. `PIPELINE_INTERVIEW_GUIDE.md`
