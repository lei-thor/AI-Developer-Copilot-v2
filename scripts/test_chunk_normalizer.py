from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from services.chunk_normalizer_service import ChunkNormalizerService  # noqa: E402


PDF_PREVIEW = ROOT_DIR / "files" / "swin_trans_pdf_chunks_preview.json"
WORD_PREVIEW = ROOT_DIR / "files" / "langchain_chunks_preview.json"
MD_PREVIEW = ROOT_DIR / "files" / "机器学习_md_chunks_preview.json"


def load_chunks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    raise AssertionError(f"Unsupported preview payload: {path}")


def assert_unified_shape(chunk: dict) -> None:
    assert set(chunk.keys()) == {
        "chunk_id",
        "chunk_index",
        "text",
        "token_count",
        "content_hash",
        "metadata",
    }
    assert set(chunk["metadata"].keys()) == {
        "document",
        "structure",
        "source",
        "content",
        "processing",
    }
    assert set(chunk["metadata"]["document"].keys()) == {"file_id", "file_name", "file_type"}
    assert set(chunk["metadata"]["structure"].keys()) == {
        "heading_path",
        "chapter",
        "section",
        "heading",
    }
    assert set(chunk["metadata"]["source"].keys()) == {
        "source_path",
        "page_numbers",
        "line_range",
    }
    assert set(chunk["metadata"]["content"].keys()) == {"types"}
    assert set(chunk["metadata"]["processing"].keys()) == {
        "parser",
        "language",
        "create_time",
    }


def assert_invariants(raw_chunks: list[dict], normalized_chunks: list[dict]) -> None:
    assert len(raw_chunks) == len(normalized_chunks)
    for raw_chunk, normalized_chunk in zip(raw_chunks, normalized_chunks):
        assert raw_chunk.get("text", "") == normalized_chunk["text"]
        assert raw_chunk.get("chunk_index") == normalized_chunk["chunk_index"]
        assert_unified_shape(normalized_chunk)


def test_pdf(normalizer: ChunkNormalizerService) -> int:
    raw_chunks = load_chunks(PDF_PREVIEW)
    original = copy.deepcopy(raw_chunks)
    normalized = normalizer.normalize_many_to_dict(raw_chunks, file_type="pdf")
    assert raw_chunks == original
    assert_invariants(raw_chunks, normalized)

    first_raw_metadata = raw_chunks[0]["metadata"]
    expected_pages = first_raw_metadata.get("page_numbers") or [first_raw_metadata["page_number"]]
    assert normalized[0]["metadata"]["source"]["page_numbers"] == expected_pages
    assert normalized[0]["metadata"]["content"]["types"] == [first_raw_metadata["block_type"]]
    assert normalized[0]["metadata"]["processing"]["parser"] == first_raw_metadata["parser"]
    return len(normalized)


def test_word(normalizer: ChunkNormalizerService) -> int:
    raw_chunks = load_chunks(WORD_PREVIEW)
    original = copy.deepcopy(raw_chunks)
    normalized = normalizer.normalize_many_to_dict(raw_chunks, file_type="word")
    assert raw_chunks == original
    assert_invariants(raw_chunks, normalized)

    first = normalized[0]["metadata"]
    assert first["document"]["file_type"] == "word"
    assert first["source"]["page_numbers"] == []
    assert first["source"]["line_range"] == []
    return len(normalized)


def test_markdown(normalizer: ChunkNormalizerService) -> int:
    raw_chunks = load_chunks(MD_PREVIEW)
    original = copy.deepcopy(raw_chunks)
    normalized = normalizer.normalize_many_to_dict(raw_chunks, file_type="markdown")
    assert raw_chunks == original
    assert_invariants(raw_chunks, normalized)

    first_raw_metadata = raw_chunks[0]["metadata"]
    first_normalized = normalized[0]["metadata"]
    expected_heading_path = first_raw_metadata["heading_path"]
    if isinstance(expected_heading_path, str):
        expected_heading_path = [expected_heading_path]
    assert first_normalized["structure"]["heading_path"] == expected_heading_path
    assert first_normalized["source"]["line_range"] == [
        first_raw_metadata["line_start"],
        first_raw_metadata["line_end"],
    ]
    assert first_normalized["content"]["types"] == first_raw_metadata["block_types"]
    assert "heading_levels" not in first_normalized["structure"]
    assert "block_type" not in first_normalized["content"]
    return len(normalized)


def main() -> None:
    normalizer = ChunkNormalizerService()
    pdf_count = test_pdf(normalizer)
    word_count = test_word(normalizer)
    markdown_count = test_markdown(normalizer)
    print(f"pdf chunks normalized: {pdf_count}")
    print(f"word chunks normalized: {word_count}")
    print(f"markdown chunks normalized: {markdown_count}")
    print("chunk normalizer checks passed")


if __name__ == "__main__":
    main()
