"""Helpers for the bundled public demo corpus."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from .config import DEMO_CORPUS_MANIFEST, INPUT_DIR


def _load_manifest() -> list[dict[str, str]]:
    return json.loads(DEMO_CORPUS_MANIFEST.read_text(encoding="utf-8"))


def list_demo_papers() -> list[dict[str, str]]:
    return _load_manifest()


def download_demo_corpus(force: bool = False) -> list[Path]:
    target_dir = INPUT_DIR / "music-generation"
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for paper in _load_manifest():
        target_path = target_dir / paper["filename"]
        if target_path.exists() and not force:
            downloaded.append(target_path)
            continue

        response = requests.get(
            paper["pdf_url"],
            timeout=180,
            headers={"User-Agent": "rag-for-technical-documents/1.0"},
        )
        response.raise_for_status()
        target_path.write_bytes(response.content)
        downloaded.append(target_path)

    return downloaded
