# Project Plan

## Project Name

`RAG for Technical Documents`

## Problem

Technical documents are long, dense, and hard to search manually.

The project will build a small RAG system that lets a user:
- upload technical PDFs
- ask questions
- get grounded answers with citations

## Project Objective

Learn how to build a modern GenAI app end-to-end:
- ingestion
- chunking
- embedding
- retrieval
- prompt grounding
- answer generation
- UI
- deployment

## Current Public Topic Choice

For the first public build, use:
`AI Music Generation`

This avoids private documents while still giving you a creative, high-signal demo.

## User Story

`As a researcher / engineer / proposal writer, I want to ask questions over technical PDFs and get concise answers with citations, so I can work faster and trust the output.`

## Scope For Day 1 MVP

### In scope

- PDF upload or fixed PDF folder
- chunking and embedding
- vector store creation
- question input
- retrieval of top chunks
- answer generation
- citation display
- basic Streamlit interface

### Out of scope for day 1

- advanced reranking
- agentic decomposition
- authentication
- cloud database
- production monitoring
- advanced evaluation dashboards

## Success Criteria

The project succeeds if:

1. the app can answer at least `10` test questions over your PDFs
2. answers are grounded in retrieved chunks
3. citations are visible
4. the repo is clear enough for a recruiter to understand in under `3` minutes
5. the app can be deployed publicly

## Recommended Data For Version 1

Use the public `AI Music Generation` topic pack first:
- `A Survey of AI Music Generation Tools and Models`
- `Applications and Advances of Artificial Intelligence in Music Generation: A Review`
- `MusicLM: Generating Music From Text`
- `Simple and Controllable Music Generation`
- `Controllable Music Production with Diffusion Models and Guidance Gradients`
- `Fast Timing-Conditioned Latent Audio Diffusion`

Why:
- fully public and easy to share
- creative and memorable
- enough overlap for retrieval and comparison questions
- strong topic for a public portfolio demo

## MVP Outputs

The app should produce:
- short answer
- retrieved source chunks
- file name references
- optional confidence or note on grounding

## Stretch Goals If You Finish Early

1. add a document summary mode
2. add a “compare two documents” mode
3. add a structured extraction mode
4. deploy publicly the same day
5. add screenshots to README

## One-Day Sprint Plan

### Phase 1: Learn the minimum
- read the basics file
- understand embeddings, chunking, retrieval, grounding

### Phase 2: Build the pipeline
- load PDFs
- chunk text
- embed chunks
- store in vector DB
- retrieve on user query

### Phase 3: Build the app
- create a small Streamlit UI
- return answer + citations

### Phase 4: Ship
- test
- clean README
- push to GitHub
- deploy
