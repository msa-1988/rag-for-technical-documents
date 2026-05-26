# 01 - RAG for Technical Documents

GitHub URL: `https://github.com/msa-1988/rag-for-technical-documents`

## Goal

Build a compact `RAG` application that reads public research PDFs and answers questions with grounded citations.

The current demo corpus is a public `AI music generation` paper set.

## Description

This project demonstrates:
- local-first RAG with `Ollama`
- GPU-backed inference
- hybrid retrieval over technical papers
- grounded answers with visible evidence
- a simple Streamlit interface for exploration and demos

It is designed to be:
- public and safe to share
- independent from paid APIs
- easy to run locally
- useful as a learning and interview project

## Utilisation

The app lets you:
- index a small PDF collection from `data/input/`
- ask natural-language questions
- inspect answer citations
- inspect retrieved context
- compare model papers and survey papers in one place

Typical usage:

1. put public PDFs into `data/input/`
2. build or refresh the index
3. ask technical questions
4. review the answer, sources, and retrieved context
5. stop the app when you are done

## Stack

- `Python`
- `Streamlit`
- `LangChain`
- `Ollama`
- `Local GPU inference`
- `Local open LLM`
- `ChromaDB`
- `PyPDF`

## Retrieval Design

This version uses:
- semantic retrieval from `ChromaDB`
- lightweight keyword retrieval from exported chunks
- source diversity limits so one paper does not dominate the answer

That hybrid design was added after real testing showed that exact model-name comparison questions needed stronger lexical support.

## Safe Local Run

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
- make sure Ollama is installed and running locally
- use the default local chat model `phi3:mini` or change it in `.env`
- put PDFs into `data/input/`
- click `Build / Refresh Index`
- ask real questions from the paper set

## Secure Public Demo

This repo is `localhost-only` by default.

If you need a temporary public demo:

1. set a real `APP_ACCESS_PASSWORD` in `.env`
2. keep the app bound to `127.0.0.1`
3. start the guarded tunnel script:

```bash
./scripts/start_secure_public_demo.sh
```

That script refuses to start a public tunnel unless a password is configured.

## Security Notes

- no personal PDFs are included in the repository
- `data/input/` is git-ignored
- the app binds to `localhost` by default via `.streamlit/config.toml`
- public sharing should be temporary and password-protected
- stop any demo tunnel immediately after use

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
  - `SECURITY_HARDENING.md`
- `app/`
  - `streamlit_app.py`
  - `src/`
- `data/input/`
- `data/output/`
- `scripts/`
  - `check_ollama_gpu.sh`
  - `smoke_test_pipeline.py`
  - `run_safe_local.sh`
  - `start_secure_public_demo.sh`
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
- the app answers questions using public PDFs
- the answer shows source chunks or citations
- the README explains architecture and setup
- the repo has screenshots or demo output
- the repo is pushed to GitHub
- any public demo is temporary and password-protected

## Next Step

Read the files in `docs/` in this order:

1. `PROJECT_PLAN.md`
2. `LEARNING_BASICS.md`
3. `INPUT_OUTPUT.md`
4. `ONE_DAY_EXECUTION.md`
5. `STEP_BY_STEP.md`
6. `LOCAL_OLLAMA_SETUP.md`
7. `PIPELINE_INTERVIEW_GUIDE.md`
8. `SECURITY_HARDENING.md`
