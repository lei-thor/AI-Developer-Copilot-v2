from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
from collections.abc import Mapping, Sequence


_MISSING = object()


@dataclass(frozen=True)
class UnifiedDocumentMetadata:
    file_id: Any = ""
    file_name: Any = ""
    file_type: Any = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": deepcopy(self.file_id),
            "file_name": deepcopy(self.file_name),
            "file_type": deepcopy(self.file_type),
        }


@dataclass(frozen=True)
class UnifiedStructureMetadata:
    heading_path: list[Any]
    chapter: Any = ""
    section: Any = ""
    heading: Any = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading_path": deepcopy(self.heading_path),
            "chapter": deepcopy(self.chapter),
            "section": deepcopy(self.section),
            "heading": deepcopy(self.heading),
        }


@dataclass(frozen=True)
class UnifiedSourceMetadata:
    source_path: Any = ""
    page_numbers: list[Any] | None = None
    line_range: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": deepcopy(self.source_path),
            "page_numbers": deepcopy(self.page_numbers or []),
            "line_range": deepcopy(self.line_range or []),
        }


@dataclass(frozen=True)
class UnifiedContentMetadata:
    types: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {"types": deepcopy(self.types)}


@dataclass(frozen=True)
class UnifiedProcessingMetadata:
    parser: Any = ""
    language: Any = ""
    create_time: Any = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser": deepcopy(self.parser),
            "language": deepcopy(self.language),
            "create_time": deepcopy(self.create_time),
        }


@dataclass(frozen=True)
class UnifiedChunkMetadata:
    document: UnifiedDocumentMetadata
    structure: UnifiedStructureMetadata
    source: UnifiedSourceMetadata
    content: UnifiedContentMetadata
    processing: UnifiedProcessingMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "structure": self.structure.to_dict(),
            "source": self.source.to_dict(),
            "content": self.content.to_dict(),
            "processing": self.processing.to_dict(),
        }


@dataclass(frozen=True)
class UnifiedChunk:
    chunk_id: Any
    chunk_index: Any
    text: str
    token_count: Any
    content_hash: Any
    metadata: UnifiedChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": deepcopy(self.chunk_id),
            "chunk_index": deepcopy(self.chunk_index),
            "text": self.text,
            "token_count": deepcopy(self.token_count),
            "content_hash": deepcopy(self.content_hash),
            "metadata": self.metadata.to_dict(),
        }


class ChunkNormalizerService:
    """Map parser-specific chunks into one stable schema.

    The normalizer only moves fields into a shared shape. It never edits text,
    never changes chunk boundaries, and never derives headings from content.
    """

    def normalize(self, raw_chunk: Any, file_type: str | None = None) -> UnifiedChunk:
        raw = self._as_mapping(raw_chunk)
        metadata = self._mapping_or_empty(self._lookup(raw, "metadata"))

        return UnifiedChunk(
            chunk_id=self._empty_if_missing(
                self._first_present(
                    self._lookup(raw, "chunk_id"),
                    self._lookup(raw, "id"),
                )
            ),
            chunk_index=self._none_if_missing(
                self._first_present(
                    self._lookup(raw, "chunk_index"),
                    self._lookup(metadata, "chunk_index"),
                )
            ),
            text=self._extract_text(raw),
            token_count=self._empty_number_if_missing(
                self._first_present(
                    self._lookup(raw, "token_count"),
                    self._lookup(metadata, "token_count"),
                )
            ),
            content_hash=self._empty_if_missing(
                self._first_present(
                    self._lookup(raw, "content_hash"),
                    self._lookup(metadata, "content_hash"),
                )
            ),
            metadata=UnifiedChunkMetadata(
                document=self._build_document_metadata(raw, metadata, file_type),
                structure=self._build_structure_metadata(raw, metadata),
                source=self._build_source_metadata(raw, metadata),
                content=self._build_content_metadata(raw, metadata),
                processing=self._build_processing_metadata(metadata),
            ),
        )

    def normalize_many(
        self,
        raw_chunks: Sequence[Any],
        file_type: str | None = None,
    ) -> list[UnifiedChunk]:
        return [self.normalize(raw_chunk, file_type=file_type) for raw_chunk in raw_chunks]

    def normalize_to_dict(
        self,
        raw_chunk: Any,
        file_type: str | None = None,
    ) -> dict[str, Any]:
        return self.normalize(raw_chunk, file_type=file_type).to_dict()

    def normalize_many_to_dict(
        self,
        raw_chunks: Sequence[Any],
        file_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.normalize_many(raw_chunks, file_type=file_type)]

    def _build_document_metadata(
        self,
        raw: Mapping[str, Any],
        metadata: Mapping[str, Any],
        file_type: str | None,
    ) -> UnifiedDocumentMetadata:
        document = self._mapping_or_empty(self._lookup(metadata, "document"))

        return UnifiedDocumentMetadata(
            file_id=self._empty_if_missing(
                self._first_present(
                    self._lookup(document, "file_id"),
                    self._lookup(metadata, "file_id"),
                    self._lookup(raw, "file_id"),
                )
            ),
            file_name=self._empty_if_missing(
                self._first_present(
                    self._lookup(document, "file_name"),
                    self._lookup(metadata, "file_name"),
                    self._lookup(raw, "file_name"),
                )
            ),
            file_type=self._empty_if_missing(
                self._first_present(
                    self._lookup(document, "file_type"),
                    self._lookup(metadata, "file_type"),
                    self._lookup(raw, "file_type"),
                    file_type if file_type else _MISSING,
                )
            ),
        )

    def _build_structure_metadata(
        self,
        raw: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> UnifiedStructureMetadata:
        structure = self._mapping_or_empty(self._lookup(metadata, "structure"))

        heading_path = self._as_list(
            self._first_present(
                self._lookup(structure, "heading_path"),
                self._lookup(metadata, "heading_path"),
                self._lookup(raw, "heading_path"),
            )
        )

        return UnifiedStructureMetadata(
            heading_path=heading_path,
            chapter=self._empty_if_missing(
                self._first_present(
                    self._lookup(structure, "chapter"),
                    self._lookup(metadata, "chapter"),
                )
            ),
            section=self._empty_if_missing(
                self._first_present(
                    self._lookup(structure, "section"),
                    self._lookup(metadata, "section"),
                )
            ),
            heading=self._empty_if_missing(
                self._first_present(
                    self._lookup(structure, "heading"),
                    self._lookup(metadata, "heading"),
                )
            ),
        )

    def _build_source_metadata(
        self,
        raw: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> UnifiedSourceMetadata:
        source = self._mapping_or_empty(self._lookup(metadata, "source"))

        page_numbers = self._first_present(
            self._lookup(source, "page_numbers"),
            self._lookup(metadata, "page_numbers"),
            self._lookup(raw, "page_numbers"),
        )
        if page_numbers is _MISSING:
            page_numbers = self._first_present(
                self._lookup(source, "page_number"),
                self._lookup(metadata, "page_number"),
                self._lookup(raw, "page_number"),
            )

        line_range = self._first_present(
            self._lookup(source, "line_range"),
            self._lookup(metadata, "line_range"),
            self._lookup(raw, "line_range"),
        )
        if line_range is _MISSING:
            line_range = self._build_line_range(
                self._first_present(
                    self._lookup(source, "line_start"),
                    self._lookup(metadata, "line_start"),
                    self._lookup(raw, "line_start"),
                ),
                self._first_present(
                    self._lookup(source, "line_end"),
                    self._lookup(metadata, "line_end"),
                    self._lookup(raw, "line_end"),
                ),
            )

        return UnifiedSourceMetadata(
            source_path=self._empty_if_missing(
                self._first_present(
                    self._lookup(source, "source_path"),
                    self._lookup(metadata, "source_path"),
                    self._lookup(raw, "source_path"),
                )
            ),
            page_numbers=self._as_list(page_numbers),
            line_range=self._as_list(line_range),
        )

    def _build_content_metadata(
        self,
        raw: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> UnifiedContentMetadata:
        content = self._mapping_or_empty(self._lookup(metadata, "content"))
        content_types = self._first_present(
            self._lookup(content, "types"),
            self._lookup(metadata, "types"),
            self._lookup(raw, "types"),
            self._lookup(metadata, "block_types"),
            self._lookup(raw, "block_types"),
            self._lookup(metadata, "block_type"),
            self._lookup(raw, "block_type"),
        )

        return UnifiedContentMetadata(types=self._as_list(content_types))

    def _build_processing_metadata(self, metadata: Mapping[str, Any]) -> UnifiedProcessingMetadata:
        processing = self._mapping_or_empty(self._lookup(metadata, "processing"))

        return UnifiedProcessingMetadata(
            parser=self._empty_if_missing(
                self._first_present(
                    self._lookup(processing, "parser"),
                    self._lookup(metadata, "parser"),
                )
            ),
            language=self._empty_if_missing(
                self._first_present(
                    self._lookup(processing, "language"),
                    self._lookup(metadata, "language"),
                )
            ),
            create_time=self._empty_if_missing(
                self._first_present(
                    self._lookup(processing, "create_time"),
                    self._lookup(metadata, "create_time"),
                )
            ),
        )

    def _as_mapping(self, raw_chunk: Any) -> Mapping[str, Any]:
        if isinstance(raw_chunk, Mapping):
            return raw_chunk

        if hasattr(raw_chunk, "to_dict"):
            raw_dict = raw_chunk.to_dict()
            if isinstance(raw_dict, Mapping):
                return raw_dict

        if is_dataclass(raw_chunk) and not isinstance(raw_chunk, type):
            return asdict(raw_chunk)

        raise TypeError("raw_chunk must be a mapping, dataclass instance, or expose to_dict().")

    def _extract_text(self, raw: Mapping[str, Any]) -> str:
        text = self._lookup(raw, "text")
        if text is _MISSING or text is None:
            return ""
        if not isinstance(text, str):
            raise TypeError("raw_chunk['text'] must be a string.")
        return text

    def _mapping_or_empty(self, value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    def _lookup(self, mapping: Mapping[str, Any], key: str) -> Any:
        if not isinstance(mapping, Mapping):
            return _MISSING
        return mapping[key] if key in mapping else _MISSING

    def _first_present(self, *values: Any) -> Any:
        for value in values:
            if value is not _MISSING and value is not None:
                return value
        return _MISSING

    def _empty_if_missing(self, value: Any) -> Any:
        return "" if value is _MISSING or value is None else deepcopy(value)

    def _none_if_missing(self, value: Any) -> Any:
        return None if value is _MISSING else deepcopy(value)

    def _empty_number_if_missing(self, value: Any) -> Any:
        return 0 if value is _MISSING or value is None else deepcopy(value)

    def _as_list(self, value: Any) -> list[Any]:
        if value is _MISSING or value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, Mapping):
            return [deepcopy(value)]
        if isinstance(value, Sequence):
            return [deepcopy(item) for item in value if item is not None and item != ""]
        return [deepcopy(value)]

    def _build_line_range(self, line_start: Any, line_end: Any) -> list[Any]:
        line_range: list[Any] = []
        if line_start is not _MISSING and line_start is not None:
            line_range.append(deepcopy(line_start))
        if line_end is not _MISSING and line_end is not None:
            line_range.append(deepcopy(line_end))
        return line_range


__all__ = [
    "ChunkNormalizerService",
    "UnifiedChunk",
    "UnifiedChunkMetadata",
    "UnifiedDocumentMetadata",
    "UnifiedStructureMetadata",
    "UnifiedSourceMetadata",
    "UnifiedContentMetadata",
    "UnifiedProcessingMetadata",
]
