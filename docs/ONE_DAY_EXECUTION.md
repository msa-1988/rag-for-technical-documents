# One-Day Execution Plan

This project is scoped as a `1-day MVP`.

## Target End State By Tonight

You should have:
- working local GPU app
- GitHub repo
- screenshots
- README

## Step-by-Step

### Step 1: Learn only the essentials
Time: `45 to 60 minutes`

Read:
- `LEARNING_BASICS.md`

Goal:
- understand RAG flow end-to-end

### Step 2: Prepare the project environment
Time: `15 to 20 minutes`

Do:
- create virtual environment
- install requirements
- copy 2 to 4 PDFs into `data/input/`
- confirm Ollama models are available
- confirm GPU runtime with `ollama ps`

### Step 3: Build ingestion
Time: `45 minutes`

Do:
- load PDFs
- extract text
- split into chunks

Done when:
- you can print number of chunks per file

### Step 4: Build vector store
Time: `30 to 45 minutes`

Do:
- create embeddings
- store chunks in ChromaDB

Done when:
- query returns relevant chunks

### Step 5: Build answer pipeline
Time: `45 to 60 minutes`

Do:
- take user question
- retrieve top chunks
- pass chunks to LLM
- return answer with citations

Done when:
- answer cites the source file

### Step 6: Build UI
Time: `30 to 45 minutes`

Do:
- create Streamlit app
- add question input
- show answer
- show retrieved chunks

Done when:
- another person can use it without reading the code

### Step 7: Test with real questions
Time: `30 minutes`

Create 10 questions from your documents.

Check:
- retrieval relevance
- answer correctness
- citations

### Step 8: Publish
Time: `45 to 60 minutes`

Do:
- update README
- add screenshots
- push to GitHub
- write down next-step deployment plan

## Fast Publish Checklist

- repo name is clear
- README explains:
  - problem
  - architecture
  - stack
  - how to run
  - example questions
- screenshots added
- local GPU setup explained

## If You Finish Early

Add one of these:
- summary mode
- compare-documents mode
- structured extraction mode
- evaluation notebook
