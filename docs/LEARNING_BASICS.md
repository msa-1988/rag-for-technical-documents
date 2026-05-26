# Learning Basics Required

You do not need deep theory before starting.  
You only need the minimum practical basics.

## What To Learn Before Coding

### 1. What RAG is

RAG = `Retrieval-Augmented Generation`

Instead of asking an LLM to answer from memory, you:
- retrieve relevant document chunks
- give them to the model
- force the answer to be grounded in that context

### 2. Embeddings

Embeddings convert text into vectors so similarity search becomes possible.

What you need to know:
- similar text should be close in vector space
- user query is embedded too
- vector search retrieves similar chunks

### 3. Chunking

Long PDFs cannot be sent as one block.

So you split them into chunks.

What you need to know:
- too small = weak context
- too large = noisy retrieval
- start simple: medium chunks with overlap

### 4. Vector database

A vector DB stores embedded chunks and lets you retrieve nearest matches.

For this project:
- use `ChromaDB`

### 5. Retrieval vs generation

Retrieval finds the right chunks.  
Generation writes the answer.

If retrieval is poor, the answer will also be poor.

### 6. Grounding and citations

The model should answer only from retrieved context.

That means:
- show source chunks
- cite file names
- avoid unsupported claims

### 7. Basic evaluation

For day 1, evaluation is simple:
- ask 10 real questions
- check whether retrieved chunks are relevant
- check whether answer is supported by the chunks

### 8. Streamlit basics

You only need to know:
- text input
- file uploader
- buttons
- text display

### 9. Local inference runtime

For this project, the LLM does not come from a paid API.

You are using:
- `Ollama` for local inference
- a small local chat model
- a local embedding model

Important practical point:
- Ollama will use the GPU automatically when available
- you can verify that with `ollama ps`

## Minimum Learning Sequence

Use this order:

1. understand RAG at a high level
2. understand embeddings
3. understand chunking
4. understand vector search
5. understand prompt grounding
6. understand simple evaluation

## What Not To Learn Yet

Do not waste time today on:
- advanced rerankers
- hybrid search theory
- agent frameworks
- long papers on RAG benchmarks
- distributed vector databases

## Day 1 Mindset

Your goal today is:
- not to master RAG academically
- but to build one working RAG app that you understand end to end
