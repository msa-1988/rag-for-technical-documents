"""Streamlit UI for the local-first RAG demo."""

import hmac
import time
from pathlib import Path

import streamlit as st

from src.config import (
    APP_ACCESS_PASSWORD,
    APP_SESSION_TIMEOUT_MINUTES,
    APP_SHOW_DETAILED_ERRORS,
    DEMO_AUTO_FETCH,
    GITHUB_REPO_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    RAG_RUNTIME,
)
from src.demo_corpus import download_demo_corpus, list_demo_papers
from src.pipeline import answer_question, build_vector_store, get_index_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"
SAMPLE_QUESTIONS = [
    "What are the main categories of AI music generation systems?",
    "How do MusicLM and MusicGen differ in architecture and control?",
    "What does controllability mean in AI music generation?",
    "Compare autoregressive and diffusion-based music generation approaches.",
]


def list_input_pdfs() -> list[str]:
    return sorted([str(p.relative_to(INPUT_DIR)) for p in INPUT_DIR.rglob("*.pdf")])


def render_shell() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f2e8;
            --panel: rgba(255, 251, 245, 0.9);
            --ink: #16211d;
            --muted: #5d6b64;
            --line: rgba(22, 33, 29, 0.12);
            --accent: #176b5f;
            --accent-soft: #d7f0ea;
            --warm: #d9863b;
            --shadow: 0 18px 50px rgba(47, 62, 55, 0.08);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(255, 212, 179, 0.45), transparent 30%),
                radial-gradient(circle at top right, rgba(132, 204, 196, 0.30), transparent 30%),
                linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(216, 240, 235, 0.72));
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 1.8rem 1.9rem 1.4rem 1.9rem;
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
            color: var(--ink);
        }

        .hero p {
            margin: 0.8rem 0 0 0;
            color: var(--muted);
            font-size: 1rem;
            max-width: 56rem;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.38rem 0.72rem;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(23, 107, 95, 0.15);
            color: var(--accent);
            font-size: 0.84rem;
            font-weight: 600;
        }

        .metric-card, .panel-card, .result-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .metric-card {
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 112px;
        }

        .metric-label {
            color: var(--muted);
            text-transform: uppercase;
            font-size: 0.73rem;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.5rem;
        }

        .panel-card {
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            margin-bottom: 1rem;
        }

        .panel-card h3 {
            margin: 0 0 0.6rem 0;
            font-size: 1.05rem;
        }

        .subtle {
            color: var(--muted);
            font-size: 0.93rem;
        }

        .answer-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(240, 249, 247, 0.95));
            border-left: 4px solid var(--accent);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 0.85rem;
        }

        .source-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            padding: 0.2rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 700;
        }

        .warm-chip {
            background: rgba(217, 134, 59, 0.12);
            color: #9a571e;
        }

        .helper-list {
            margin: 0;
            padding-left: 1.1rem;
            color: var(--muted);
        }

        .helper-list li {
            margin-bottom: 0.28rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    if RAG_RUNTIME == "lite":
        description = (
            "A public demo over AI music generation papers, tuned for free-tier hosting "
            "with a lightweight local model and citation-aware retrieval."
        )
        badges = [
            "Free-tier demo",
            "Hosted mode",
            "TF-IDF + keyword retrieval",
            "Grounded citations",
            "Extractive answer composer",
        ]
    else:
        description = (
            "A local-first question answering app over AI music generation papers, "
            "running with Ollama, ChromaDB, and GPU-backed inference on your machine."
        )
        badges = [
            "Local GPU",
            "Ollama",
            "Hybrid retrieval",
            "Grounded citations",
            f"Chat: {OLLAMA_CHAT_MODEL}",
            f"Embeddings: {OLLAMA_EMBEDDING_MODEL}",
        ]

    badge_markup = "".join([f'<span class="badge">{badge}</span>' for badge in badges])
    st.markdown(
        f"""
        <section class="hero">
            <h1>RAG Studio for Public Research Papers</h1>
            <p>{description}</p>
            <div class="badge-row">{badge_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def auth_required() -> bool:
    return bool(APP_ACCESS_PASSWORD)


def session_is_valid() -> bool:
    if not auth_required():
        return True

    authenticated = st.session_state.get("authenticated", False)
    if not authenticated:
        return False

    last_seen = st.session_state.get("last_seen_at", 0.0)
    if time.time() - last_seen > APP_SESSION_TIMEOUT_MINUTES * 60:
        st.session_state["authenticated"] = False
        st.session_state["last_seen_at"] = 0.0
        st.session_state["flash_error"] = "Session expired. Enter the access password again."
        return False

    st.session_state["last_seen_at"] = time.time()
    return True


def render_auth_gate() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <h3>Protected Demo</h3>
            <p class="subtle">
                This app is password-gated before public sharing. Enter the access password to use the interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    password = st.text_input("Access password", type="password")
    if st.button("Unlock app", type="primary", width="stretch"):
        if hmac.compare_digest(password, APP_ACCESS_PASSWORD):
            st.session_state["authenticated"] = True
            st.session_state["last_seen_at"] = time.time()
            st.rerun()
        else:
            st.error("Wrong password.")


def handle_error(exc: Exception) -> None:
    if APP_SHOW_DETAILED_ERRORS:
        st.error(str(exc))
    else:
        st.error("The request failed. Confirm the active runtime is available and rebuild the index if needed.")


def update_status_from_stats(stats) -> dict[str, int | bool | str]:
    return {
        "pdf_count": stats.pdf_count,
        "indexed": True,
        "vector_store_ready": True,
        "page_count": stats.page_count,
        "chunk_count": stats.chunk_count,
        "runtime": RAG_RUNTIME,
    }


def init_state() -> None:
    st.session_state.setdefault("question", SAMPLE_QUESTIONS[1])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("flash_message", None)
    st.session_state.setdefault("flash_error", None)
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("last_seen_at", 0.0)


def set_question(prompt: str) -> None:
    st.session_state["question"] = prompt


st.set_page_config(
    page_title="RAG for Technical Documents",
    page_icon="🎵",
    layout="wide",
)

render_shell()
init_state()
status = get_index_status()
pdfs = list_input_pdfs()

render_hero()

if not auth_required():
    if RAG_RUNTIME == "lite":
        st.info("Hosted demo mode is active. The bundled public corpus can be fetched automatically or with the button below.")
    else:
        st.info(
            "Local-only mode is active. For any public sharing, set `APP_ACCESS_PASSWORD` in `.env` first."
        )
elif not session_is_valid():
    render_auth_gate()
    st.stop()

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric_card("Paper Stack", str(status["pdf_count"]), "Public AI music generation papers loaded")
with metric_cols[1]:
    render_metric_card("Indexed Pages", str(status["page_count"]), "Pages parsed into the working corpus")
with metric_cols[2]:
    render_metric_card("Chunks", str(status["chunk_count"]), "Chunked passages ready for retrieval")
with metric_cols[3]:
    render_metric_card(
        "Index Health",
        "Ready" if status["indexed"] else "Pending",
        (
            "Retrieval artifacts built"
            if status["vector_store_ready"]
            else "Index manifest exists but retrieval artifacts are missing"
        ),
    )

if st.session_state["flash_message"]:
    st.success(st.session_state["flash_message"])
    st.session_state["flash_message"] = None

if st.session_state["flash_error"]:
    st.error(st.session_state["flash_error"])
    st.session_state["flash_error"] = None

left_col, right_col = st.columns([1.35, 0.95], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Question Studio")
    st.caption("Pick a prompt, refine it, then run a grounded answer over the indexed papers.")

    prompt_cols = st.columns(2)
    for idx, prompt in enumerate(SAMPLE_QUESTIONS):
        target_col = prompt_cols[idx % 2]
        with target_col:
            if st.button(prompt, key=f"sample_{idx}", width="stretch"):
                set_question(prompt)
                st.rerun()

    question = st.text_area(
        "Ask about the loaded paper set",
        key="question",
        height=160,
        placeholder="Compare MusicLM and MusicGen in terms of architecture and control.",
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        build_index = st.button("Build / Refresh Index", width="stretch")
    with action_col2:
        run_query = st.button("Run RAG Query", type="primary", width="stretch")

    if RAG_RUNTIME == "lite":
        if st.button("Load Bundled Demo Corpus", width="stretch"):
            with st.spinner("Downloading the public AI music generation corpus..."):
                try:
                    download_demo_corpus(force=False)
                except Exception as exc:
                    st.session_state["flash_error"] = (
                        str(exc) if APP_SHOW_DETAILED_ERRORS else
                        "Could not download the bundled public corpus."
                    )
                else:
                    st.session_state["flash_message"] = "Bundled public corpus is ready in data/input/music-generation."
                    st.rerun()

    if build_index:
        with st.spinner("Building retrieval index from the paper stack..."):
            try:
                _, build_stats = build_vector_store(force_rebuild=True)
            except Exception as exc:
                st.session_state["flash_error"] = (
                    str(exc) if APP_SHOW_DETAILED_ERRORS else
                    "Index build failed. Confirm the runtime dependencies are available and try again."
                )
            else:
                status = update_status_from_stats(build_stats)
                st.session_state["flash_message"] = (
                    f"Index refreshed: {build_stats.pdf_count} PDFs, "
                    f"{build_stats.page_count} pages, {build_stats.chunk_count} chunks."
                )
                st.rerun()

    if run_query:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    result = answer_question(question)
                except Exception as exc:
                    st.session_state["flash_error"] = (
                        str(exc)
                        if APP_SHOW_DETAILED_ERRORS
                        else "Query failed. Rebuild the index if needed and confirm the configured runtime is available."
                    )
                    st.rerun()
                else:
                    st.session_state["last_result"] = result
                    status = update_status_from_stats(result["stats"])

    st.markdown("</div>", unsafe_allow_html=True)

    result = st.session_state["last_result"]
    if result:
        vector_hits = sum(1 for source in result["sources"] if source["retrieval"] in {"vector", "tfidf"})
        keyword_hits = sum(1 for source in result["sources"] if source["retrieval"] == "keyword")
        vector_label = "TF-IDF hits" if RAG_RUNTIME == "lite" else "Vector hits"

        st.markdown('<div class="result-card panel-card">', unsafe_allow_html=True)
        tabs = st.tabs(["Answer", "Sources", "Retrieved Context", "Pipeline"])

        with tabs[0]:
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(result["answer"])
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <span class="source-chip">{vector_label}: {vector_hits}</span>
                &nbsp;
                <span class="source-chip warm-chip">Keyword hits: {keyword_hits}</span>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "This answer is grounded only in the retrieved context shown in the other tabs."
            )

        with tabs[1]:
            for source in result["sources"]:
                label = f"[{source['id']}] {source['source']} · page {source['page']}"
                with st.expander(label, expanded=(source["id"] == 1)):
                    st.markdown(
                        f"""
                        <span class="source-chip">{source['retrieval']}</span>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(source["snippet"])

        with tabs[2]:
            for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
                label = f"Chunk {idx}: {chunk['source']} · page {chunk['page']}"
                with st.expander(label, expanded=(idx <= 2)):
                    st.markdown(
                        f"""
                        <span class="source-chip">{chunk['retrieval']}</span>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(chunk["snippet"])

        with tabs[3]:
            if RAG_RUNTIME == "lite":
                st.markdown(
                    f"""
                    1. The bundled public corpus is downloaded into `data/input/`.
                    2. PDFs are split into overlapping chunks.
                    3. `TF-IDF` retrieval runs over the chunked passages, then a keyword pass boosts named entities.
                    4. The app merges those results with source diversity limits, so one paper does not dominate the context.
                    5. A lightweight answer composer turns the retrieved evidence into a cited response for the public demo.
                    """
                )
            else:
                st.markdown(
                    f"""
                    1. PDFs are loaded from `data/input/` and split into overlapping chunks.
                    2. `{OLLAMA_EMBEDDING_MODEL}` converts those chunks into vectors for ChromaDB.
                    3. A user question triggers two retrieval passes:
                       - vector similarity search
                       - lightweight keyword retrieval over exported chunks
                    4. The app merges those results with source diversity limits, so one paper does not dominate the context.
                    5. `{OLLAMA_CHAT_MODEL}` generates the final answer from the retrieved evidence and cites chunk numbers.
                    """
                )

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="panel-card">
                <h3>Ready for the first run</h3>
                <p class="subtle">
                    Build the index once, then try one of the sample questions.
                    The results panel will show the answer, citations, and the retrieved context together.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown(
        """
        <div class="panel-card">
            <h3>Runtime and Retrieval</h3>
            <p class="subtle">
                The app supports a high-fidelity local Ollama runtime and a lighter hosted mode
                tuned for free-tier public demos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if auth_required():
        if st.button("Lock session", width="stretch"):
            st.session_state["authenticated"] = False
            st.session_state["last_seen_at"] = 0.0
            st.rerun()

    st.markdown(
        """
        <div class="panel-card">
            <h3>System Snapshot</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"- Index ready: `{status['indexed']}`")
    st.markdown(f"- Runtime: `{status['runtime']}`")
    st.markdown(f"- Retrieval artifacts ready: `{status['vector_store_ready']}`")
    if RAG_RUNTIME == "lite":
        st.markdown(f"- Demo corpus auto-fetch: `{DEMO_AUTO_FETCH}`")
        st.markdown("- Answer mode: `extractive synthesis with citations`")
        st.markdown("- Retrieval model: `scikit-learn TF-IDF + keyword fallback`")
    else:
        st.markdown(f"- Chat model: `{OLLAMA_CHAT_MODEL}`")
        st.markdown(f"- Embedding model: `{OLLAMA_EMBEDDING_MODEL}`")
    st.markdown(f"- Repository: [rag-for-technical-documents]({GITHUB_REPO_URL})")

    st.markdown(
        """
        <div class="panel-card">
            <h3>Paper Stack</h3>
            <p class="subtle">The current dataset is the public AI music generation topic pack.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pdfs:
        for name in pdfs:
            st.markdown(f"- `{name}`")
    else:
        st.warning("No PDFs found yet in `data/input/`.")
        if RAG_RUNTIME == "lite":
            st.markdown("Bundled papers:")
            for paper in list_demo_papers():
                st.markdown(f"- [{paper['title']}]({paper['pdf_url']})")

    st.markdown(
        """
        <div class="panel-card">
            <h3>Fast Cross-Checks</h3>
            <ul class="helper-list">
                <li>Ask a definition question and verify the citations.</li>
                <li>Ask a comparison question and confirm at least two papers appear.</li>
                <li>Check that both retrieval signals show up for named entities.</li>
                <li>In local mode, use <code>ollama ps</code> to confirm the chat model is still on GPU.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if RAG_RUNTIME == "lite":
    st.caption(
        "Hosted demo mode is designed for free-tier public sharing. It uses a smaller local model and may wake from sleep on first visit."
    )
else:
    st.caption(
        "This app is intended to run on localhost by default. If you share it publicly, keep password protection enabled and stop the tunnel as soon as the demo is over."
    )
