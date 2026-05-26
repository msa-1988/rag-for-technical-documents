# Pipeline Interview Guide

This document explains the exact pipeline built in this project so you can:
- understand it deeply
- explain it clearly in interviews
- improve it later without guessing

## 1. What this project is

This is a `local-first RAG application` over a small public paper set.

Current dataset:
- `6` public papers on `AI music generation`

Current runtime:
- local `Ollama`
- local chat model: `phi3:mini`
- local embedding model: `nomic-embed-text`
- GPU-backed inference when available

UI:
- `Streamlit`

Storage:
- `ChromaDB`

## 2. Problem statement

Long technical papers are hard to search manually.

A normal chatbot is not enough because:
- it may hallucinate
- it may answer from pretraining instead of the actual papers
- it may not show where the answer came from

So the goal was:

`Ask questions over a paper set and get grounded answers with visible evidence.`

## 3. End-to-end architecture

The pipeline has `8` main stages:

1. load PDFs from `data/input/`
2. split pages into overlapping chunks
3. embed chunks with a local embedding model
4. store vectors in ChromaDB
5. export chunk text for lightweight keyword retrieval
6. retrieve evidence using both vector search and keyword search
7. merge and diversify the evidence
8. send grounded context to the local LLM for answer generation

## 4. Why this architecture was chosen

This is not the most advanced RAG architecture possible.
It is the best one for a fast first project because it is:
- understandable
- fully local
- cheap to run
- easy to demo
- easy to extend later

It also gives good interview coverage across:
- document ingestion
- embeddings
- vector databases
- grounding
- retrieval quality
- local inference
- UI and product thinking

## 5. Step-by-step pipeline explanation

### Step 1: Document loading

Code area:
- `app/src/pipeline.py`
- function: `load_documents()`

What happens:
- the app scans `data/input/` recursively
- each PDF is read with `PyPDFLoader`
- each page becomes a LangChain document
- metadata is attached:
  - source filename
  - source path

Why it matters:
- source metadata is later used for citations
- recursive input lets you keep topic folders clean

Interview explanation:

`I start by converting each PDF into page-level documents and preserving source metadata early, so every later chunk still knows which paper and page it came from.`

### Step 2: Chunking

Code area:
- function: `split_documents()`

Current settings:
- `CHUNK_SIZE = 800`
- `CHUNK_OVERLAP = 150`

Why chunking is needed:
- full PDFs are too long for direct prompting
- retrieval works better on medium-size passages
- overlap prevents important facts from being split across chunk boundaries

Tradeoff:
- small chunks improve precision but may lose context
- large chunks preserve context but hurt retrieval precision

Why these settings were reasonable:
- good enough for a first technical-document MVP
- practical for medium-length research papers

Interview explanation:

`I used overlapping chunks so retrieval stays precise without losing context at page boundaries. The chunk size was a practical first-pass setting, not a theoretical optimum.`

### Step 3: Embeddings

Code area:
- class: `OllamaEmbeddings`

Current embedding model:
- `nomic-embed-text`

What happens:
- chunk text is sent to the local Ollama embedding endpoint
- text becomes vectors
- those vectors are used for semantic search

Why local embeddings:
- no paid API
- no provider lock-in
- privacy-friendly
- stronger “open-model engineering” story

Important implementation detail:
- embeddings are generated in batches
- controlled by `EMBED_BATCH_SIZE`

Interview explanation:

`I wrapped the Ollama embedding API in a small adapter so Chroma could use a local model the same way it would use a hosted embedding provider.`

### Step 4: Vector store

Code area:
- function: `build_vector_store()`

What happens:
- chunk embeddings are written into `ChromaDB`
- the index is persisted locally
- a manifest is stored in `data/output/index_manifest.json`

Why the manifest exists:
- to know whether the current index still matches the current input PDFs
- to avoid rebuilding unnecessarily

Why persistence matters:
- indexing is slower than querying
- once built, the app should reuse the local index

Interview explanation:

`I persisted the vector store locally and added a simple input signature manifest so the app can tell whether the index is still fresh.`

### Step 5: Chunk export for keyword retrieval

Code area:
- functions:
  - `_write_chunk_export()`
  - `_read_chunk_export()`

What happens:
- after chunking, each chunk is also exported to `chunks.jsonl`
- source, page, and text are stored

Why this exists:
- semantic retrieval is not always enough
- named entities like `MusicLM` and `MusicGen` can benefit from direct lexical matching

This was added after a real failure:
- a comparison question retrieved mostly `MusicLM` evidence
- `MusicGen` was underrepresented
- the answer became too cautious

That observation led to a better pipeline design.

Interview explanation:

`I added a lightweight chunk export because pure vector retrieval was not always enough for exact entity matching. That let me add a simple lexical fallback without introducing a heavier search stack.`

### Step 6: Hybrid retrieval

Code area:
- functions:
  - `_keyword_retrieve()`
  - `_merge_retrieved_content()`

What happens now:
- vector retrieval runs from ChromaDB
- keyword retrieval runs over the exported chunks
- both retrieval streams are merged

Why hybrid retrieval helps:
- vector search is good at semantic similarity
- keyword search is good at exact entity matches
- combining them improves robustness

This matters especially for:
- paper names
- model names
- technical terms
- comparison questions

Interview explanation:

`The pipeline now uses hybrid retrieval because semantic search alone was good for broad questions but weaker on exact model-name comparisons.`

### Step 7: Source diversity

Code area:
- `_merge_retrieved_content()`

What it does:
- limits how many chunks come from the same source
- prevents one paper from dominating the prompt

Why it matters:
- comparison questions need evidence from more than one paper
- naive top-k retrieval can over-concentrate on one source

Interview explanation:

`I added source diversity limits because top-k retrieval often overfocused on one document, which weakened comparison-style answers.`

### Step 8: Grounded prompt construction

Code area:
- function: `answer_question()`

What happens:
- retrieved chunks are formatted into a single context block
- each chunk gets an index like `[1]`, `[2]`
- the model is told to answer only from the provided context
- the model is told to cite chunk numbers

Why this prompt style:
- simple
- direct
- easy to inspect
- good for an interview demo

Interview explanation:

`I used a grounded prompt with indexed chunks so the generated answer could reference specific evidence rather than answering from general model memory.`

### Step 9: Local generation

Code area:
- function: `_chat_completion()`

Current model:
- `phi3:mini`

Why this model was used first:
- already available locally
- small enough for a fast MVP
- works on local GPU through Ollama

Why local GPU matters:
- faster than CPU-only local inference
- better developer experience
- stronger story for “end-to-end local AI app”

Interview explanation:

`The answer stage runs through a local Ollama chat model, which keeps the app self-contained and makes the tradeoff between model quality and infrastructure visibility very clear.`

## 6. What the UI is doing

The app UI is not just decoration.
It exposes the pipeline clearly:

- metrics show corpus size and index state
- question panel drives the query flow
- source and context tabs make retrieval visible
- pipeline tab explains the system behavior

This is good product thinking because:
- users can see what the app is doing
- debugging becomes easier
- interviewers can understand the app quickly

## 7. Real issues encountered

### Issue 1: Semantic retrieval alone was not enough

Observed behavior:
- model comparison questions did not reliably retrieve both model papers

Fix:
- added keyword retrieval
- merged keyword and vector evidence
- added source diversity control

Lesson:
- good RAG systems are often retrieval-limited, not model-limited

### Issue 2: Manifest could exist without a live vector store

Observed behavior:
- cached metadata can outlive the actual vector DB directory

Fix:
- status now reports whether the vector store itself exists

Lesson:
- “indexed” should reflect real runtime state, not just metadata files

### Issue 3: Public deployment is harder with a local LLM

Observed behavior:
- local-first apps are great for development but not automatically public

Lesson:
- local development and public deployment are different architecture stages

## 8. How to explain this project in interviews

### 30-second version

`I built a local-first RAG app over public AI music generation papers using Ollama, ChromaDB, and Streamlit. It chunks PDFs, embeds them locally, stores them in a vector DB, adds a lightweight keyword fallback for better exact-name retrieval, and generates grounded answers with citations.`

### 2-minute version

`The project solves a practical paper-navigation problem: asking questions over technical PDFs without relying on a paid API. I used PyPDF for ingestion, overlapping chunking, local embeddings via Ollama, and ChromaDB for semantic retrieval. During testing I noticed comparison questions like MusicLM vs MusicGen were not always retrieving balanced evidence, so I added a hybrid retrieval step that mixes vector search with keyword matching and limits how many chunks come from the same paper. The Streamlit app shows the answer, citations, and raw retrieved context so the system is inspectable rather than black-box.`

### 5-minute version

Explain:
- the ingestion flow
- chunking tradeoffs
- local embeddings
- Chroma persistence
- hybrid retrieval addition
- source diversity control
- prompt grounding
- GPU-local Ollama inference
- deployment tradeoffs

## 9. Strong interview questions you may get

### Why did you choose local Ollama instead of a hosted API?

Good answer:
- independence from paid APIs
- stronger open-model learning
- privacy and portability
- better understanding of the full stack

### Why did you add keyword retrieval?

Good answer:
- real retrieval failure on named entities
- vector retrieval alone was not always sufficient
- hybrid retrieval improved robustness without adding major complexity

### Why did you use ChromaDB?

Good answer:
- easy local setup
- persistent vector store
- good fit for a small first project

### What would you improve next?

Good answer:
- better reranking
- structured evaluation dataset
- better prompt templates
- hosted deployment layer
- source-level confidence reporting

## 10. Future upgrades

Best next upgrades:

1. add a reranker
2. add evaluation scripts and benchmark questions
3. add document comparison mode
4. add summary mode
5. add hosted inference for public deployment
6. replace the base chat model with a stronger local model

## 11. What this project proves on your CV

This project proves:
- you can build a working GenAI application
- you understand RAG end-to-end
- you can debug retrieval quality problems
- you can use local open models, not just hosted APIs
- you can explain architecture clearly

That is strong signal for:
- AI engineer roles
- applied ML roles
- GenAI product engineering roles
- remote or hybrid roles where public GitHub proof matters
