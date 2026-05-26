# Step By Step: Learn to Publish in 1 Day

Use this exact order. Do not try to master everything before building.

## 1. Learn the minimum first

Read these files in order:

1. `LEARNING_BASICS.md`
2. `PROJECT_PLAN.md`
3. `INPUT_OUTPUT.md`
4. `ONE_DAY_EXECUTION.md`

Target time: `45-60 minutes`

Your goal is only to understand:
- what RAG is
- what embeddings do
- why chunking matters
- how retrieval differs from generation
- why citations matter

## 2. Prepare the repo

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
ollama ps
```

The default chat model is already installed on this machine:
- `phi3:mini`

If you want a better small local model later, pull one of:
- `ollama pull qwen2.5:3b-instruct`
- `ollama pull phi4-mini`

Then check that Ollama is running locally:

```bash
ollama list
```

And verify the active model uses the GPU when loaded:

```bash
ollama ps
```

## 3. Add your first documents

Copy `2-4` PDFs into:

`data/input/`

Best first set:
- one recent paper
- one deliverable
- one profile or CV document

Keep the first test set small.

## 4. Run the app locally

```bash
streamlit run app/streamlit_app.py
```

Then:
- click `Build / Refresh Index`
- wait for chunking and embeddings
- ask `5-10` real questions

## 5. Check quality fast

Ask questions where you already know the answer.

Good checks:
- does it answer from the documents instead of guessing?
- does it cite the correct PDF and page?
- does retrieval miss obvious relevant content?
- are chunks too large or too small?

If the answers are weak:
- reduce chunk size
- increase `TOP_K`
- test with fewer PDFs
- rewrite the question more clearly

## 6. Capture proof for GitHub

Before publishing, add:
- one screenshot of the app
- one screenshot of an answer with citations
- one short architecture diagram if possible

## 7. Push to GitHub

```bash
git add .
git commit -m "Build day-1 RAG MVP for technical documents"
git branch -M main
git remote add origin <your-new-repo-url>
git push -u origin main
```

## 8. Deploy publicly

Fastest option: `Streamlit Community Cloud`

Basic flow:
1. push repo to GitHub
2. create a new app in Streamlit Cloud
3. point it to `app/streamlit_app.py`
4. add secrets:
   - `OLLAMA_BASE_URL`
   - `OLLAMA_CHAT_MODEL`
   - `OLLAMA_EMBEDDING_MODEL`
5. deploy

Note:
For a public cloud deployment, a pure local Ollama setup will not work unless you also host Ollama somewhere reachable.
So for day 1, focus first on:
- local GPU demo
- GitHub repo
- screenshots
- architecture notes

Public deployment can come in step 2 using:
- a self-hosted Ollama service
- Azure
- another hosted inference endpoint

## 9. Publish the project

After deployment:
- add the repo to your pinned GitHub projects
- add the live link to LinkedIn Featured
- write one LinkedIn post:
  - what problem you solved
  - what stack you used
  - what you learned
  - link to GitHub

## 10. Upgrade after day 1

Only after the MVP is live, improve:
- better prompts
- better evaluation
- better UI
- caching
- multiple model options
- exportable answer report
