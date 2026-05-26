"""Streamlit UI for the local-first RAG demo."""

from pathlib import Path

import streamlit as st

from src.config import OLLAMA_CHAT_MODEL, OLLAMA_EMBEDDING_MODEL
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
    st.markdown(
        f"""
        <section class="hero">
            <h1>RAG Studio for Public Research Papers</h1>
            <p>
                A local-first question answering app over AI music generation papers,
                running with Ollama, ChromaDB, and GPU-backed inference on your machine.
            </p>
            <div class="badge-row">
                <span class="badge">Local GPU</span>
                <span class="badge">Ollama</span>
                <span class="badge">Hybrid Retrieval</span>
                <span class="badge">Grounded Citations</span>
                <span class="badge">Chat: {OLLAMA_CHAT_MODEL}</span>
                <span class="badge">Embeddings: {OLLAMA_EMBEDDING_MODEL}</span>
            </div>
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


def update_status_from_stats(stats) -> dict[str, int | bool]:
    return {
        "pdf_count": stats.pdf_count,
        "indexed": True,
        "vector_store_ready": True,
        "page_count": stats.page_count,
        "chunk_count": stats.chunk_count,
    }


def init_state() -> None:
    st.session_state.setdefault("question", SAMPLE_QUESTIONS[1])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("flash_message", None)
    st.session_state.setdefault("flash_error", None)


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
        "Chroma store built" if status["vector_store_ready"] else "Manifest exists but vector store is missing",
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
            if st.button(prompt, key=f"sample_{idx}", use_container_width=True):
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
        build_index = st.button("Build / Refresh Index", use_container_width=True)
    with action_col2:
        run_query = st.button("Run RAG Query", type="primary", use_container_width=True)

    if build_index:
        with st.spinner("Building vector index from the paper stack..."):
            try:
                _, build_stats = build_vector_store(force_rebuild=True)
            except Exception as exc:
                st.session_state["flash_error"] = str(exc)
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
                    st.session_state["flash_error"] = str(exc)
                    st.rerun()
                else:
                    st.session_state["last_result"] = result
                    status = update_status_from_stats(result["stats"])

    st.markdown("</div>", unsafe_allow_html=True)

    result = st.session_state["last_result"]
    if result:
        vector_hits = sum(1 for source in result["sources"] if source["retrieval"] == "vector")
        keyword_hits = sum(1 for source in result["sources"] if source["retrieval"] == "keyword")

        st.markdown('<div class="result-card panel-card">', unsafe_allow_html=True)
        tabs = st.tabs(["Answer", "Sources", "Retrieved Context", "Pipeline"])

        with tabs[0]:
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(result["answer"])
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <span class="source-chip">Vector hits: {vector_hits}</span>
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
            st.markdown(
                """
                1. PDFs are loaded from `data/input/` and split into overlapping chunks.
                2. `nomic-embed-text` converts those chunks into vectors for ChromaDB.
                3. A user question triggers two retrieval passes:
                   - vector similarity search
                   - lightweight keyword retrieval over exported chunks
                4. The app merges those results with source diversity limits, so one paper does not dominate the context.
                5. The merged evidence is passed to the local Ollama chat model, which answers with citation tags.
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
                Local GPU-backed inference with Ollama, plus a hybrid retrieval layer designed
                to help named entities like model names surface more reliably.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-card">
            <h3>System Snapshot</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"- Index ready: `{status['indexed']}`")
    st.markdown(f"- Vector store ready: `{status['vector_store_ready']}`")
    st.markdown(f"- Chat model: `{OLLAMA_CHAT_MODEL}`")
    st.markdown(f"- Embedding model: `{OLLAMA_EMBEDDING_MODEL}`")

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

    st.markdown(
        """
        <div class="panel-card">
            <h3>Fast Cross-Checks</h3>
            <ul class="helper-list">
                <li>Ask a definition question and verify the citations.</li>
                <li>Ask a comparison question and confirm at least two papers appear.</li>
                <li>Check that both vector and keyword retrieval show up for named entities.</li>
                <li>Use <code>ollama ps</code> to confirm the chat model is still on GPU.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "For a public hosted version later, keep this local GPU app as the reference implementation and add a hosted inference layer as phase two."
)
