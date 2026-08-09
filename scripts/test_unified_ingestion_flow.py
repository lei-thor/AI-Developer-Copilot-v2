from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from services.md_ingestion_service import MarkdownChunk, MarkdownIngestionService  # noqa: E402
from services.pdf_ingestion_service import (  # noqa: E402
    EmbeddingGenerator,
    MilvusChunkWriter,
    PdfChunk,
    PDFIngestionService,
)
from services.word_ingestion_service import WordChunk, WordIngestionService  # noqa: E402


UNIFIED_METADATA_KEYS = {"document", "structure", "source", "content", "processing"}


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.1, 0.2] for index, _ in enumerate(texts)]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    raise AssertionError(f"Unsupported preview payload: {path}")


def as_pdf_chunk(raw: dict[str, Any]) -> PdfChunk:
    return PdfChunk(
        chunk_id=raw["chunk_id"],
        chunk_index=raw["chunk_index"],
        text=raw["text"],
        token_count=raw["token_count"],
        content_hash=raw["content_hash"],
        metadata=dict(raw.get("metadata") or {}),
    )


def as_word_chunk(raw: dict[str, Any]) -> WordChunk:
    return WordChunk(
        chunk_id=raw["chunk_id"],
        chunk_index=raw["chunk_index"],
        text=raw["text"],
        token_count=raw["token_count"],
        content_hash=raw["content_hash"],
        metadata=dict(raw.get("metadata") or {}),
    )


def as_markdown_chunk(raw: dict[str, Any]) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_index=raw["chunk_index"],
        heading_path=raw.get("heading_path", ""),
        text=raw["text"],
        content_hash=raw["content_hash"],
        chunk_id=raw["chunk_id"],
        token_count=raw["token_count"],
        metadata=dict(raw.get("metadata") or {}),
    )


def assert_invariants(raw_chunks: list[Any], unified_chunks: list[Any]) -> None:
    assert len(raw_chunks) == len(unified_chunks)
    for raw, unified in zip(raw_chunks, unified_chunks):
        assert raw.chunk_id == unified.chunk_id
        assert raw.chunk_index == unified.chunk_index
        assert raw.text == unified.text
        assert raw.token_count == unified.token_count
        assert raw.content_hash == unified.content_hash


def assert_rows(unified_chunks: list[Any], vectors: list[list[float]], row_builder: Any) -> None:
    assert len(unified_chunks) == len(vectors)
    for unified, vector in zip(unified_chunks, vectors):
        row = row_builder(unified, vector)
        metadata = json.loads(row["metadata_json"])
        assert row["id"] == unified.chunk_id
        assert row["text"] == unified.text
        assert row["chunk_index"] == unified.chunk_index
        assert row["token_count"] == unified.token_count
        assert row["content_hash"] == unified.content_hash
        assert set(metadata.keys()) == UNIFIED_METADATA_KEYS


def test_pdf_flow() -> int:
    raw_chunks = [as_pdf_chunk(raw) for raw in load_chunks(ROOT_DIR / "files" / "swin_trans_pdf_chunks_preview.json")]
    writer = MilvusChunkWriter(milvus_service=object())
    service = PDFIngestionService(
        parser=object(),
        normalizer=object(),
        cleaner=object(),
        chunk_builder=object(),
        embedding_generator=EmbeddingGenerator(FakeEmbeddingProvider()),
        vector_writer=writer,
    )
    unified_chunks = service.normalize_chunks(raw_chunks)
    assert_invariants(raw_chunks, unified_chunks)
    vectors = service.generate_embeddings(unified_chunks)
    assert_rows(unified_chunks, vectors, writer._row)
    return len(unified_chunks)


def test_word_flow() -> int:
    raw_chunks = [as_word_chunk(raw) for raw in load_chunks(ROOT_DIR / "files" / "langchain_chunks_preview.json")]
    service = WordIngestionService(embedding_provider=FakeEmbeddingProvider(), milvus_service=object())
    unified_chunks = service.normalize_chunks(raw_chunks)
    assert_invariants(raw_chunks, unified_chunks)
    vectors = service.generate_embeddings(unified_chunks)
    assert_rows(unified_chunks, vectors, service.writer._build_row)
    return len(unified_chunks)


def test_markdown_flow() -> int:
    preview = next((ROOT_DIR / "files").glob("*_md_chunks_preview.json"))
    raw_chunks = [as_markdown_chunk(raw) for raw in load_chunks(preview)]
    service = MarkdownIngestionService(embedding_provider=FakeEmbeddingProvider(), milvus_service=object())
    unified_chunks = service.normalize_chunks(raw_chunks)
    assert_invariants(raw_chunks, unified_chunks)
    vectors = service.generate_embeddings(unified_chunks)
    assert_rows(unified_chunks, vectors, service.writer._build_row)
    return len(unified_chunks)


def main() -> None:
    print(f"pdf unified flow chunks: {test_pdf_flow()}")
    print(f"word unified flow chunks: {test_word_flow()}")
    print(f"markdown unified flow chunks: {test_markdown_flow()}")
    print("unified ingestion flow checks passed")


if __name__ == "__main__":
    main()
