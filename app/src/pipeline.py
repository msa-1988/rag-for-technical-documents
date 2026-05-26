"""Minimal RAG pipeline for the first technical-documents MVP."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_EXPORT_PATH,
    EMBED_BATCH_SIZE,
    INDEX_MANIFEST,
    INPUT_DIR,
    KEYWORD_TOP_K,
    MAX_CHUNKS_PER_SOURCE,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TOP_K,
    VECTOR_DB_DIR,
)


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


def list_pdf_paths() -> list[Path]:
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


def index_is_fresh(pdf_paths: list[Path] | None = None) -> bool:
    pdf_paths = pdf_paths or list_pdf_paths()
    manifest = _read_manifest()
    if not manifest or not VECTOR_DB_DIR.exists():
        return False
    return manifest.get("inputs") == _input_signature(pdf_paths)


def load_documents() -> list:
    pdf_paths = list_pdf_paths()
    if not pdf_paths:
        raise RuntimeError("No PDFs found in data/input. Add 2-4 PDFs first.")

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


def build_vector_store(force_rebuild: bool = False) -> tuple[Chroma, IndexStats]:
    pdf_paths = list_pdf_paths()
    if not pdf_paths:
        raise RuntimeError("No PDFs found in data/input. Add 2-4 PDFs first.")

    if not force_rebuild and index_is_fresh(pdf_paths):
        manifest = _read_manifest() or {}
        vector_store = _load_existing_vector_store()
        stats = IndexStats(
            pdf_count=manifest.get("pdf_count", len(pdf_paths)),
            page_count=manifest.get("page_count", 0),
            chunk_count=manifest.get("chunk_count", 0),
            rebuilt=False,
        )
        return vector_store, stats

    documents = load_documents()
    chunks = split_documents(documents)

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=_embedding_client(),
        collection_name="technical_documents",
        persist_directory=str(VECTOR_DB_DIR),
    )

    _write_manifest(
        {
            "inputs": _input_signature(pdf_paths),
            "pdf_count": len(pdf_paths),
            "page_count": len(documents),
            "chunk_count": len(chunks),
        }
    )
    _write_chunk_export(chunks)

    return vector_store, IndexStats(
        pdf_count=len(pdf_paths),
        page_count=len(documents),
        chunk_count=len(chunks),
        rebuilt=True,
    )


def get_index_status() -> dict[str, int | bool]:
    pdf_paths = list_pdf_paths()
    manifest = _read_manifest() or {}
    return {
        "pdf_count": len(pdf_paths),
        "indexed": index_is_fresh(pdf_paths),
        "vector_store_ready": VECTOR_DB_DIR.exists(),
        "page_count": manifest.get("page_count", 0),
        "chunk_count": manifest.get("chunk_count", 0),
    }


def _format_context(retrieved_docs: list) -> str:
    context_blocks = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 0)) + 1
        context_blocks.append(
            f"[{idx}] Source: {source}, page {page}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_blocks)


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
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-:+.]*", question.lower())
    return [term for term in terms if len(term) > 2 and term not in stopwords]


def _keyword_score(content: str, question_terms: list[str]) -> int:
    lowered = content.lower()
    exact_hits = sum(lowered.count(term) for term in question_terms)
    unique_hits = sum(1 for term in question_terms if term in lowered)
    return (exact_hits * 3) + unique_hits


def _keyword_retrieve(question: str, limit: int) -> list[dict]:
    question_terms = _question_terms(question)
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


def _doc_key(source: str, page: int, snippet: str) -> tuple[str, int, str]:
    return (source, page, snippet[:180])


def _merge_retrieved_content(vector_docs: list, keyword_docs: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    per_source = {}

    vector_items = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": int(doc.metadata.get("page", 0)) + 1,
            "snippet": doc.page_content.strip(),
            "retrieval": "vector",
        }
        for doc in vector_docs
    ]
    keyword_items = [
        {
            "source": doc["source"],
            "page": doc["page"],
            "snippet": doc["snippet"],
            "retrieval": "keyword",
        }
        for doc in keyword_docs
    ]

    queues = [keyword_items, vector_items]
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


def answer_question(question: str, force_rebuild: bool = False) -> dict:
    vector_store, stats = build_vector_store(force_rebuild=force_rebuild)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K + 2})
    vector_docs = retriever.invoke(question)
    keyword_docs = _keyword_retrieve(question, limit=KEYWORD_TOP_K)
    retrieved_chunks = _merge_retrieved_content(vector_docs, keyword_docs)

    if not retrieved_chunks:
        return {
            "answer": "No relevant document chunks were retrieved.",
            "sources": [],
            "retrieved_chunks": [],
            "stats": stats,
        }

    prompt = f"""
You answer questions only from the provided technical-document context.
If the answer is not supported by the context, say that clearly.
Use short, direct language.
When you make a claim, cite the supporting chunk numbers like [1] or [2].
When the question asks for a comparison, compare the items directly and mention both sides if the context supports it.

Question:
{question}

Context:
{_format_hybrid_context(retrieved_chunks)}
""".strip()

    response = _chat_completion(prompt)
    sources = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
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
        "retrieved_chunks": retrieved_chunks,
        "stats": stats,
    }
