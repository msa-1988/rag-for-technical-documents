# Local Ollama Setup

This project is now configured for `local-first RAG` using `Ollama`.

## GPU status on this machine

GPU inference is now available and verified.

Current machine:
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- VRAM: about `8 GB`
- Verified runtime: `phi3:mini` on `100% GPU`

That means this project now runs as:
- local embeddings
- local generation
- GPU-accelerated inference through Ollama

## Why local first

This version is better for:
- learning the full stack without API cost
- keeping the project independent from a paid provider
- showing open-model and privacy-aware engineering on GitHub

## Default local models

- chat model: `phi3:mini`
- embedding model: `nomic-embed-text`

The chat model is already present on this machine.
The embedding model is small and should be pulled once.

## Required commands

```bash
ollama pull nomic-embed-text
ollama list
ollama ps
```

If Ollama is not already serving in the background, run:

```bash
ollama serve
```

## How to verify GPU usage

Run:

```bash
ollama ps
```

You should see something like:

```text
PROCESSOR    100% GPU
```

You can also inspect GPU memory use with:

```bash
nvidia-smi
```

## Current recommended model choice

For this repo right now, keep:
- chat model: `phi3:mini`
- embedding model: `nomic-embed-text`

Reason:
- already installed
- verified on GPU
- fast enough for a first RAG MVP
- low setup friction

## Optional better chat models

If you want a stronger small local model later:

```bash
ollama pull qwen2.5:3b-instruct
```

or

```bash
ollama pull phi4-mini
```

Then update `.env`:

```bash
OLLAMA_CHAT_MODEL=qwen2.5:3b-instruct
```

or

```bash
OLLAMA_CHAT_MODEL=phi4-mini
```

Use that only after the base app is working.

## CV / LinkedIn framing

A good way to describe this project later:

`Built a local-first RAG application over public research papers using Ollama, ChromaDB, and a small open LLM, with grounded answers and citation-aware retrieval.`
