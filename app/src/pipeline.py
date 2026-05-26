"""RAG pipeline for technical-document question answering."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from .config import (
    CHUNK_EXPORT_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEMO_AUTO_FETCH,
    EMBED_BATCH_SIZE,
    INDEX_MANIFEST,
    INPUT_DIR,
    KEYWORD_TOP_K,
    LITE_CONTEXT_CHAR_LIMIT,
    MAX_CHUNKS_PER_SOURCE,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAG_RUNTIME,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
    TOP_K,
    VECTOR_DB_DIR,
)
from .demo_corpus import download_demo_corpus


load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class IndexStats:
    pdf_count: int
    page_count: int
    chunk_count: int
    rebuilt: bool


@dataclass
class ChunkRecord:
    source: str
    page: int
    content: str


@dataclass
class LiteResources:
    vectorizer: TfidfVectorizer
    matrix: object
    chunks: list[ChunkRecord]


def is_lite_runtime() -> bool:
    return RAG_RUNTIME == "lite"


def list_pdf_paths() -> list[Path]:
    pdf_paths = sorted(INPUT_DIR.rglob("*.pdf"))
    if pdf_paths or not DEMO_AUTO_FETCH:
        return pdf_paths

    download_demo_corpus(force=False)
    return sorted(INPUT_DIR.rglob("*.pdf"))


class OllamaEmbeddings:
    """Small adapter so Chroma can use Ollama's local embedding API."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            embeddings.extend(self._embed(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=600,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama at "
                f"{self.base_url}. Start Ollama and confirm it is serving locally."
            ) from exc

        if response.status_code >= 400:
            raise RuntimeError(
                "Ollama embedding request failed. "
                f"Status {response.status_code}: {response.text}"
            )

        payload = response.json()
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embeddings. "
                f"Make sure `{self.model}` is available locally."
            )
        return embeddings


def _embedding_client() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _chat_completion(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=600,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{OLLAMA_BASE_URL}. Start Ollama and confirm it is serving locally."
        ) from exc

    if response.status_code >= 400:
        raise RuntimeError(
            "Ollama chat request failed. "
            f"Status {response.status_code}: {response.text}"
        )

    payload = response.json()
    message = payload.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError(
            "Ollama returned an empty response. "
            f"Make sure `{OLLAMA_CHAT_MODEL}` is available locally."
        )
    return content


def _input_signature(pdf_paths: list[Path]) -> list[dict[str, int | str]]:
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in pdf_paths
    ]


def _read_manifest() -> dict | None:
    if not INDEX_MANIFEST.exists():
        return None
    return json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))


def _write_manifest(payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_chunk_export(chunks: list) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNK_EXPORT_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = {
                "source": chunk.metadata.get("source", "unknown"),
                "page": int(chunk.metadata.get("page", 0)) + 1,
                "content": chunk.page_content,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _read_chunk_export() -> list[ChunkRecord]:
    if not CHUNK_EXPORT_PATH.exists():
        return []

    records = []
    for line in CHUNK_EXPORT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            ChunkRecord(
                source=payload["source"],
                page=payload["page"],
                content=payload["content"],
            )
        )
    return records


def _runtime_artifacts_ready() -> bool:
    if is_lite_runtime():
        return (
            CHUNK_EXPORT_PATH.exists()
            and TFIDF_VECTORIZER_PATH.exists()
            and TFIDF_MATRIX_PATH.exists()
        )
    return CHUNK_EXPORT_PATH.exists() and VECTOR_DB_DIR.exists()


def index_is_fresh(pdf_paths: list[Path] | None = None) -> bool:
    pdf_paths = pdf_paths or list_pdf_paths()
    manifest = _read_manifest()
    if not manifest or not _runtime_artifacts_ready():
        return False
    return (
        manifest.get("inputs") == _input_signature(pdf_paths)
        and manifest.get("runtime") == RAG_RUNTIME
    )


def load_documents() -> list:
    pdf_paths = list_pdf_paths()
    if not pdf_paths:
        raise RuntimeError("No PDFs found in data/input. Add PDFs or enable demo auto-fetch.")

    documents = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            page.metadata["source"] = path.name
            page.metadata["source_path"] = str(path)
        documents.extend(pages)
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _load_existing_vector_store() -> Chroma:
    return Chroma(
        collection_name="technical_documents",
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=_embedding_client(),
    )


@lru_cache(maxsize=1)
def _load_tfidf_resources(cache_token: str) -> LiteResources:
    del cache_token
    return LiteResources(
        vectorizer=joblib.load(TFIDF_VECTORIZER_PATH),
        matrix=load_npz(TFIDF_MATRIX_PATH),
        chunks=_read_chunk_export(),
    )


def _tfidf_cache_token() -> str:
    manifest = _read_manifest() or {}
    return json.dumps(
        {
            "runtime": manifest.get("runtime"),
            "inputs": manifest.get("inputs", []),
            "chunk_count": manifest.get("chunk_count", 0),
        },
        sort_keys=True,
    )


def build_vector_store(force_rebuild: bool = False) -> tuple[Chroma | None, IndexStats]:
    pdf_paths = list_pdf_paths()
    if not pdf_paths:
        raise RuntimeError(
            "No PDFs found in data/input. Add PDFs locally or let the demo corpus download."
        )

    if not force_rebuild and index_is_fresh(pdf_paths):
        manifest = _read_manifest() or {}
        vector_store = None if is_lite_runtime() else _load_existing_vector_store()
        stats = IndexStats(
            pdf_count=manifest.get("pdf_count", len(pdf_paths)),
            page_count=manifest.get("page_count", 0),
            chunk_count=manifest.get("chunk_count", 0),
            rebuilt=False,
        )
        return vector_store, stats

    documents = load_documents()
    chunks = split_documents(documents)
    stats = IndexStats(
        pdf_count=len(pdf_paths),
        page_count=len(documents),
        chunk_count=len(chunks),
        rebuilt=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        {
            "runtime": RAG_RUNTIME,
            "inputs": _input_signature(pdf_paths),
            "pdf_count": len(pdf_paths),
            "page_count": len(documents),
            "chunk_count": len(chunks),
        }
    )
    _write_chunk_export(chunks)

    if is_lite_runtime():
        texts = [chunk.page_content for chunk in chunks]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
        save_npz(TFIDF_MATRIX_PATH, matrix)
        _load_tfidf_resources.cache_clear()
        return None, stats

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=_embedding_client(),
        collection_name="technical_documents",
        persist_directory=str(VECTOR_DB_DIR),
    )
    return vector_store, stats


def get_index_status() -> dict[str, int | bool | str]:
    pdf_paths = list_pdf_paths()
    manifest = _read_manifest() or {}
    return {
        "pdf_count": len(pdf_paths),
        "indexed": index_is_fresh(pdf_paths),
        "vector_store_ready": _runtime_artifacts_ready(),
        "page_count": manifest.get("page_count", 0),
        "chunk_count": manifest.get("chunk_count", 0),
        "runtime": RAG_RUNTIME,
    }


def _question_terms(question: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "what",
        "which",
        "does",
        "mean",
        "over",
        "than",
        "when",
        "where",
        "their",
        "they",
        "them",
        "have",
        "were",
        "how",
        "why",
        "who",
        "can",
        "you",
        "your",
        "using",
        "used",
        "between",
        "compare",
        "difference",
        "differ",
    }
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9:+.-]*", question.lower())
    return [term for term in terms if len(term) > 2 and term not in stopwords]


def _keyword_score(content: str, question_terms: list[str]) -> int:
    lowered = content.lower()
    exact_hits = sum(lowered.count(term) for term in question_terms)
    unique_hits = sum(1 for term in question_terms if term in lowered)
    return (exact_hits * 3) + unique_hits


def _keyword_retrieve(
    question: str,
    limit: int,
    terms_override: list[str] | None = None,
) -> list[dict]:
    question_terms = terms_override or _question_terms(question)
    if not question_terms:
        return []

    scored_chunks = []
    for chunk in _read_chunk_export():
        score = _keyword_score(chunk.content, question_terms)
        if score <= 0:
            continue
        scored_chunks.append(
            {
                "source": chunk.source,
                "page": chunk.page,
                "snippet": chunk.content.strip(),
                "score": score,
            }
        )

    scored_chunks.sort(
        key=lambda item: (item["score"], len(item["snippet"])),
        reverse=True,
    )
    return scored_chunks[:limit]


def _tfidf_retrieve(question: str, limit: int) -> list[dict]:
    resources = _load_tfidf_resources(_tfidf_cache_token())
    query_vector = resources.vectorizer.transform([question])
    scores = linear_kernel(resources.matrix, query_vector).ravel()

    if scores.size == 0:
        return []

    retrieved = []
    for index in scores.argsort()[::-1]:
        score = float(scores[index])
        if score <= 0:
            break
        chunk = resources.chunks[index]
        retrieved.append(
            {
                "source": chunk.source,
                "page": chunk.page,
                "snippet": chunk.content.strip(),
                "retrieval": "tfidf",
                "score": score,
            }
        )
        if len(retrieved) >= limit:
            break

    return retrieved


def _vector_docs_to_items(vector_docs: list) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": int(doc.metadata.get("page", 0)) + 1,
            "snippet": doc.page_content.strip(),
            "retrieval": "vector",
        }
        for doc in vector_docs
    ]


def _doc_key(source: str, page: int, snippet: str) -> tuple[str, int, str]:
    return (source, page, snippet[:180])


def _merge_retrieved_content(vector_items: list[dict], keyword_docs: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    per_source = {}

    keyword_items = [
        {
            "source": doc["source"],
            "page": doc["page"],
            "snippet": doc["snippet"],
            "retrieval": "keyword",
        }
        for doc in keyword_docs
    ]

    queues = [keyword_items, list(vector_items)]
    while len(merged) < TOP_K and any(queues):
        for queue in queues:
            if not queue:
                continue
            item = queue.pop(0)
            key = _doc_key(item["source"], item["page"], item["snippet"])
            if key in seen:
                continue
            if per_source.get(item["source"], 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            seen.add(key)
            per_source[item["source"]] = per_source.get(item["source"], 0) + 1
            merged.append(item)
            if len(merged) >= TOP_K:
                break

    return merged


def _format_hybrid_context(retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[{idx}] Source: {chunk['source']}, page {chunk['page']}, retrieval {chunk['retrieval']}\n"
            f"{chunk['snippet']}"
        )
    return "\n\n".join(context_blocks)


def _generation_chunks(retrieved_chunks: list[dict]) -> list[dict]:
    if not is_lite_runtime():
        return retrieved_chunks

    trimmed = []
    total_chars = 0
    for chunk in retrieved_chunks:
        snippet = chunk["snippet"][:700]
        estimated = len(snippet) + 80
        if total_chars + estimated > LITE_CONTEXT_CHAR_LIMIT and trimmed:
            break

        item = dict(chunk)
        item["snippet"] = snippet
        trimmed.append(item)
        total_chars += estimated

    return trimmed or retrieved_chunks[:3]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _is_comparison_question(question: str) -> bool:
    lowered = question.lower()
    comparison_markers = ("compare", "differ", "difference", "versus", "vs", "contrast")
    return any(marker in lowered for marker in comparison_markers)


def _lite_focus_terms(question: str) -> list[str]:
    lowered = question.lower()
    focus_terms = []

    if any(token in lowered for token in ("categories", "category", "types", "approaches")):
        focus_terms.extend(
            [
                "autoregressive",
                "diffusion",
                "transformer",
                "gan",
                "vae",
                "symbolic",
                "text-to-music",
                "latent",
            ]
        )

    if any(token in lowered for token in ("controllability", "control", "conditioning")):
        focus_terms.extend(
            [
                "control",
                "controllable",
                "conditioning",
                "conditioned",
                "guidance",
                "guided",
                "attribute",
                "steer",
                "text prompt",
            ]
        )

    if "architecture" in lowered:
        focus_terms.extend(
            [
                "architecture",
                "autoregressive",
                "transformer",
                "encoder",
                "decoder",
                "latent",
            ]
        )

    for token in _question_terms(question):
        if any(char.isdigit() for char in token) or token in {"musiclm", "musicgen", "riffusion"}:
            focus_terms.append(token)

    return sorted(set(focus_terms))


def _sentence_is_noise(sentence: str) -> bool:
    lowered = sentence.lower()
    words = sentence.split()
    blocked_phrases = (
        "list of web resources",
        "google search",
        "bing news",
        "google scholar",
        "hacker news",
        "huggingface",
        "youtube",
        "reddit",
        "github",
        "keywords:",
        "table ",
        "figure ",
        "appendix",
        "references",
        "copyright",
    )
    if any(phrase in lowered for phrase in blocked_phrases):
        return True
    if lowered.count(",") >= 7:
        return True
    if sentence.count("%") >= 2:
        return True
    if sentence.count("%") >= 1 and lowered.count("music") >= 4:
        return True
    if "http" in lowered or "www." in lowered:
        return True
    verbs = (
        " is ",
        " are ",
        " uses ",
        " use ",
        " allows ",
        " allow ",
        " enables ",
        " enable ",
        " means ",
        " refers to ",
        " based on ",
        " generates ",
        " condition ",
        " conditioning ",
        " controls ",
        " compares ",
    )
    has_common_verb = any(verb in f" {lowered} " for verb in verbs)
    capitalized_tokens = sum(1 for word in words if word[:1].isupper())
    if not has_common_verb and len(words) >= 10 and (capitalized_tokens >= 4 or any(char.isdigit() for char in sentence)):
        return True
    return False


def _lite_answer_from_chunks(question: str, retrieved_chunks: list[dict]) -> str:
    question_terms = _question_terms(question)
    focus_terms = _lite_focus_terms(question)
    candidates = []
    seen_sentences = set()

    for chunk_id, chunk in enumerate(retrieved_chunks, start=1):
        for sentence in _split_sentences(chunk["snippet"]):
            normalized = " ".join(sentence.split())
            if len(normalized) < 50:
                continue
            if _sentence_is_noise(normalized):
                continue
            sentence_key = normalized.lower()[:180]
            if sentence_key in seen_sentences:
                continue

            score = _keyword_score(normalized, question_terms)
            focus_score = _keyword_score(normalized, focus_terms) if focus_terms else 0
            if focus_terms and focus_score <= 0:
                continue
            if score <= 0 and focus_score <= 0 and question_terms:
                continue

            seen_sentences.add(sentence_key)
            candidates.append(
                {
                    "sentence": normalized,
                    "chunk_id": chunk_id,
                    "source": chunk["source"],
                    "score": score + (focus_score * 4),
                }
            )

    candidates.sort(
        key=lambda item: (item["score"], len(item["sentence"])),
        reverse=True,
    )

    selected = []
    sources_used = set()
    for candidate in candidates:
        if candidate["source"] in sources_used and len(selected) >= 2:
            continue
        selected.append(candidate)
        sources_used.add(candidate["source"])
        if len(selected) >= 3:
            break

    if not selected:
        fallback = []
        for chunk_id, chunk in enumerate(retrieved_chunks[:2], start=1):
            snippet = " ".join(chunk["snippet"].split())
            fallback.append(f"{snippet[:260].rstrip()} [{chunk_id}]")
        prefix = "Comparison from the retrieved context: " if _is_comparison_question(question) else "Answer grounded in the retrieved context: "
        return prefix + " ".join(fallback)

    prefix = "Comparison from the retrieved context: " if _is_comparison_question(question) else "Answer grounded in the retrieved context: "
    summary = " ".join(
        f"{candidate['sentence']} [{candidate['chunk_id']}]"
        for candidate in selected
    )
    return prefix + summary


def answer_question(question: str, force_rebuild: bool = False) -> dict:
    vector_store, stats = build_vector_store(force_rebuild=force_rebuild)

    if is_lite_runtime():
        vector_items = _tfidf_retrieve(question, limit=TOP_K + 2)
        keyword_docs = _keyword_retrieve(
            question,
            limit=2,
            terms_override=_lite_focus_terms(question),
        )
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K + 2})
        vector_items = _vector_docs_to_items(retriever.invoke(question))
        keyword_docs = _keyword_retrieve(question, limit=KEYWORD_TOP_K)
    retrieved_chunks = _merge_retrieved_content(vector_items, keyword_docs)

    if not retrieved_chunks:
        return {
            "answer": "No relevant document chunks were retrieved.",
            "sources": [],
            "retrieved_chunks": [],
            "stats": stats,
        }

    prompt_chunks = _generation_chunks(retrieved_chunks)
    prompt = f"""
Answer the question using only the provided technical-document context.
If the answer is not supported by the context, say so clearly.
Use concise language.
When you make a claim, cite the supporting chunk numbers like [1] or [2].
When the question asks for a comparison, compare the items directly and mention both sides if the context supports it.

Question:
{question}

Context:
{_format_hybrid_context(prompt_chunks)}
""".strip()

    response = (
        _lite_answer_from_chunks(question, prompt_chunks)
        if is_lite_runtime()
        else _chat_completion(prompt)
    )
    sources = []
    for idx, chunk in enumerate(prompt_chunks, start=1):
        sources.append(
            {
                "id": idx,
                "source": chunk["source"],
                "page": chunk["page"],
                "snippet": chunk["snippet"][:500],
                "retrieval": chunk["retrieval"],
            }
        )

    return {
        "answer": response,
        "sources": sources,
        "retrieved_chunks": prompt_chunks,
        "stats": stats,
    }
