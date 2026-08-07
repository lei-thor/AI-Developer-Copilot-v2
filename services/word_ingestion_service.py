from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from services.milvus_service import MilvusConnectionService, get_milvus_service


WORD_FILE_TYPE = "word"
DEFAULT_WORD_COLLECTION = "developer_knowledge_chunks"


class EmbeddingProvider(Protocol):
    """Adapter contract for an existing embedding service."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass(frozen=True)
class WordElement:
    element_type: str
    content: str
    level: int | None = None
    heading_path: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.element_type,
            "content": self.content,
        }
        if self.level is not None:
            data["level"] = self.level
        if self.heading_path:
            data["heading_path"] = list(self.heading_path)
        if self.table_rows:
            data["table_rows"] = [list(row) for row in self.table_rows]
        return data


@dataclass(frozen=True)
class WordChunk:
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    content_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WordIngestionResult:
    source_path: str
    file_name: str
    file_type: str
    collection_name: str
    document_hash: str
    parsed_elements_count: int
    cleaned_elements_count: int
    chunks_count: int
    inserted_count: int
    chunk_ids: list[str]


class WordParser:
    """Parse .docx files while preserving paragraphs, headings, lists, and tables."""

    supported_suffixes = {".docx"}

    def parse(self, file_path: str | Path) -> list[WordElement]:
        path = self._validate_file(file_path)

        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is required to ingest Word files. Install it in law_rag."
            ) from exc

        document = Document(path)
        elements: list[WordElement] = []

        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, document)
                element = self._parse_paragraph(paragraph)
                if element is not None:
                    elements.append(element)
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                table_element = self._parse_table(table)
                if table_element is not None:
                    elements.append(table_element)

        return elements

    def _validate_file(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Word file does not exist: {path}")
        if path.suffix.lower() not in self.supported_suffixes:
            raise ValueError("Only .docx Word files are supported.")
        return path

    def _parse_paragraph(self, paragraph: Any) -> WordElement | None:
        text = paragraph.text.strip()
        if not text:
            return None

        heading_level = self._heading_level(paragraph, text)
        if heading_level is not None:
            return WordElement(
                element_type="heading",
                level=heading_level,
                content=text,
            )

        if self._is_list_item(paragraph, text):
            return WordElement(element_type="list", content=text)

        return WordElement(element_type="paragraph", content=text)

    def _parse_table(self, table: Any) -> WordElement | None:
        rows: list[tuple[str, ...]] = []
        for row in table.rows:
            cells = tuple(self._normalize_cell(cell.text) for cell in row.cells)
            if any(cells):
                rows.append(cells)

        if not rows:
            return None

        content = WordTableNormalizer().to_natural_language(rows)
        return WordElement(
            element_type="table",
            content=content,
            table_rows=tuple(rows),
        )

    def _heading_level(self, paragraph: Any, text: str) -> int | None:
        style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
        lower_style = style_name.lower()
        if lower_style.startswith("heading") or style_name.startswith("标题"):
            match = re.search(r"(\d+)", style_name)
            return int(match.group(1)) if match else 1

        outline_level = self._outline_level(paragraph)
        if outline_level is not None:
            return outline_level

        match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]+", text)
        if match:
            return min(match.group(1).count(".") + 1, 6)

        if re.match(r"^\s*第[一二三四五六七八九十百千万0-9]+[章节篇部分]", text):
            return 1

        return None

    def _outline_level(self, paragraph: Any) -> int | None:
        p_pr = getattr(paragraph._p, "pPr", None)
        outline = getattr(p_pr, "outlineLvl", None) if p_pr is not None else None
        if outline is None or outline.val is None:
            return None
        return int(outline.val) + 1

    def _is_list_item(self, paragraph: Any, text: str) -> bool:
        style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
        p_pr = getattr(paragraph._p, "pPr", None)
        has_numbering = bool(p_pr is not None and getattr(p_pr, "numPr", None) is not None)
        style_says_list = any(
            marker in style_name.lower()
            for marker in ("list", "bullet", "number")
        ) or "列表" in style_name
        text_says_list = bool(re.match(r"^\s*([-*+]|[0-9]+[.)、]|[a-zA-Z][.)])\s+", text))
        return has_numbering or style_says_list or text_says_list

    def _normalize_cell(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


class WordTableNormalizer:
    """Convert Word tables into embedding-friendly natural language."""

    explanation_headers = {"原因", "含义", "说明", "描述", "功能", "用途", "解释", "处理方式"}

    def to_natural_language(self, rows: Sequence[Sequence[str]]) -> str:
        normalized_rows = [tuple(cell.strip() for cell in row) for row in rows if any(row)]
        if not normalized_rows:
            return ""

        if len(normalized_rows) == 1:
            return self._row_to_sentence((), normalized_rows[0], 1)

        header = normalized_rows[0]
        body_rows = normalized_rows[1:]
        sentences = [
            self._row_to_sentence(header, row, row_index)
            for row_index, row in enumerate(body_rows, start=1)
            if any(row)
        ]
        return "\n".join(sentence for sentence in sentences if sentence)

    def _row_to_sentence(
        self,
        header: Sequence[str],
        row: Sequence[str],
        row_index: int,
    ) -> str:
        values = [cell for cell in row if cell]
        if not values:
            return ""

        if len(row) == 2 and len(header) >= 2:
            key_name = header[0] or "字段"
            key_value = self._sentence_part(row[0])
            value_name = header[1] or "说明"
            value = self._sentence_part(row[1])
            if value_name in self.explanation_headers:
                return f"{key_name}{key_value}表示{value}。"
            return f"{key_name}{key_value}的{value_name}是{value}。"

        if header and len(header) == len(row):
            pairs = [
                f"{name}为{self._sentence_part(value)}"
                for name, value in zip(header, row)
                if name and value
            ]
            return f"表格第{row_index}行记录: " + "，".join(pairs) + "。"

        return f"表格第{row_index}行记录: " + "，".join(
            self._sentence_part(value) for value in values
        ) + "。"

    def _sentence_part(self, value: str) -> str:
        return value.strip().rstrip("。.;；")


class WordCleaner:
    """Clean text conservatively and attach heading context to each element."""

    def clean(self, elements: Sequence[WordElement]) -> list[WordElement]:
        cleaned: list[WordElement] = []
        heading_stack: list[str] = []
        paragraph_buffer: list[str] = []

        for element in elements:
            content = self._clean_text(element.content)
            if not content:
                continue

            if element.element_type == "heading":
                self._flush_paragraph_buffer(cleaned, paragraph_buffer, heading_stack)
                paragraph_buffer.clear()
                level = max(element.level or 1, 1)
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(content)
                cleaned.append(
                    WordElement(
                        element_type="heading",
                        level=level,
                        content=content,
                        heading_path=tuple(heading_stack),
                    )
                )
                continue

            if element.element_type in {"paragraph", "list"}:
                paragraph_buffer.append(content)
                continue

            self._flush_paragraph_buffer(cleaned, paragraph_buffer, heading_stack)
            paragraph_buffer.clear()
            cleaned.append(
                WordElement(
                    element_type=element.element_type,
                    content=content,
                    heading_path=tuple(heading_stack),
                    table_rows=element.table_rows,
                )
            )

        self._flush_paragraph_buffer(cleaned, paragraph_buffer, heading_stack)
        return cleaned

    def _flush_paragraph_buffer(
        self,
        cleaned: list[WordElement],
        paragraph_buffer: list[str],
        heading_stack: Sequence[str],
    ) -> None:
        if not paragraph_buffer:
            return
        text = "\n".join(paragraph_buffer).strip()
        if text:
            cleaned.append(
                WordElement(
                    element_type="paragraph",
                    content=text,
                    heading_path=tuple(heading_stack),
                )
            )

    def _clean_text(self, text: str) -> str:
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class TokenEstimator:
    """Small dependency-free token estimator for chunk budgeting."""

    token_pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")

    def count(self, text: str) -> int:
        return len(self.token_pattern.findall(text))

    def split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？!?；;.\n])\s*", text)
        return [part.strip() for part in parts if part.strip()]


class WordChunker:
    """Split by heading hierarchy first, then by estimated token length."""

    def __init__(
        self,
        target_tokens: int = 650,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_estimator = token_estimator or TokenEstimator()

    def chunk(self, elements: Sequence[WordElement], file_path: str | Path) -> list[WordChunk]:
        path = Path(file_path)
        chunks: list[WordChunk] = []
        current_heading: tuple[str, ...] = ()
        section_items: list[WordElement] = []

        for element in elements:
            if element.element_type == "heading":
                self._flush_section(chunks, section_items, current_heading, path)
                current_heading = element.heading_path or (element.content,)
                section_items = []
                continue

            section_items.append(element)

        self._flush_section(chunks, section_items, current_heading, path)
        return chunks

    def _flush_section(
        self,
        chunks: list[WordChunk],
        section_items: list[WordElement],
        heading_path: tuple[str, ...],
        path: Path,
    ) -> None:
        if not section_items:
            return

        prefix = self._heading_prefix(heading_path)
        current_units: list[str] = []
        current_tokens = self.token_estimator.count(prefix)

        for item in section_items:
            for unit in self._content_units(item):
                unit_tokens = self.token_estimator.count(unit)
                if current_units and current_tokens + unit_tokens > self.max_tokens:
                    self._append_chunk(chunks, prefix, current_units, heading_path, path)
                    current_units = self._overlap_units(current_units)
                    current_tokens = self.token_estimator.count(prefix + "\n" + "\n".join(current_units))

                if unit_tokens > self.max_tokens:
                    self._append_large_unit(chunks, prefix, unit, heading_path, path)
                    current_units = []
                    current_tokens = self.token_estimator.count(prefix)
                    continue

                current_units.append(unit)
                current_tokens += unit_tokens

        if current_units:
            self._append_chunk(chunks, prefix, current_units, heading_path, path)

    def _content_units(self, element: WordElement) -> list[str]:
        if self.token_estimator.count(element.content) <= self.target_tokens:
            return [element.content]
        return self.token_estimator.split_sentences(element.content)

    def _append_large_unit(
        self,
        chunks: list[WordChunk],
        prefix: str,
        unit: str,
        heading_path: tuple[str, ...],
        path: Path,
    ) -> None:
        sentences = self.token_estimator.split_sentences(unit)
        buffer: list[str] = []
        buffer_tokens = self.token_estimator.count(prefix)

        for sentence in sentences:
            sentence_tokens = self.token_estimator.count(sentence)
            if buffer and buffer_tokens + sentence_tokens > self.max_tokens:
                self._append_chunk(chunks, prefix, buffer, heading_path, path)
                buffer = self._overlap_units(buffer)
                buffer_tokens = self.token_estimator.count(prefix + "\n" + "\n".join(buffer))

            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            self._append_chunk(chunks, prefix, buffer, heading_path, path)

    def _append_chunk(
        self,
        chunks: list[WordChunk],
        prefix: str,
        body_units: Sequence[str],
        heading_path: tuple[str, ...],
        path: Path,
    ) -> None:
        body = "\n".join(unit.strip() for unit in body_units if unit.strip()).strip()
        if not body:
            return

        text = f"{prefix}\n\n{body}".strip() if prefix else body
        chunk_index = len(chunks)
        chapter, section = self._chapter_and_section(heading_path)
        metadata = {
            "file_name": path.name,
            "file_type": WORD_FILE_TYPE,
            "chapter": chapter,
            "section": section,
            "chunk_index": chunk_index,
            "heading_path": " > ".join(heading_path),
        }
        content_hash = self._hash_text(text)
        chunks.append(
            WordChunk(
                chunk_id=self._chunk_id(path, chunk_index, content_hash),
                chunk_index=chunk_index,
                text=text,
                token_count=self.token_estimator.count(text),
                content_hash=content_hash,
                metadata=metadata,
            )
        )

    def _overlap_units(self, units: Sequence[str]) -> list[str]:
        overlap: list[str] = []
        token_count = 0
        for unit in reversed(units):
            unit_tokens = self.token_estimator.count(unit)
            if overlap and token_count + unit_tokens > self.overlap_tokens:
                break
            overlap.insert(0, unit)
            token_count += unit_tokens
        return overlap

    def _heading_prefix(self, heading_path: Sequence[str]) -> str:
        return "\n".join(heading_path)

    def _chapter_and_section(self, heading_path: Sequence[str]) -> tuple[str, str]:
        chapter = heading_path[0] if heading_path else ""
        section = heading_path[-1] if len(heading_path) > 1 else ""
        return chapter, section

    def _chunk_id(self, path: Path, chunk_index: int, content_hash: str) -> str:
        stable_value = f"{path.name}:{chunk_index}:{content_hash}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_value))

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class WordMetadataBuilder:
    """Finalize metadata before vector-store insertion."""

    def build(self, chunk: WordChunk, source_path: str | Path) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "source_path": str(Path(source_path).resolve()),
                "content_hash": chunk.content_hash,
                "token_count": chunk.token_count,
            }
        )
        return metadata


class WordEmbeddingGenerator:
    """Call an injected embedding provider or the shared embedding service."""

    def __init__(self, embedding_provider: Any) -> None:
        self.embedding_provider = embedding_provider

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        provider = self._provider()
        if hasattr(provider, "embed_documents"):
            vectors = provider.embed_documents(texts)
        elif hasattr(provider, "encode"):
            vectors = provider.encode(list(texts))
        elif callable(provider):
            vectors = provider(list(texts))
        else:
            raise TypeError(
                "embedding_provider must expose embed_documents(texts), encode(texts), "
                "or be callable."
            )

        normalized = [list(vector) for vector in vectors]
        if len(normalized) != len(texts):
            raise RuntimeError("Embedding result count does not match chunk count.")
        if any(not vector for vector in normalized):
            raise RuntimeError("Embedding provider returned an empty vector.")
        return normalized

    def _provider(self) -> Any:
        if self.embedding_provider is not None:
            return self.embedding_provider

        from services.embedding_service import get_embedding_service

        return get_embedding_service()


class WordMilvusWriter:
    """Write Word chunks and vectors by reusing the existing Milvus connection service."""

    def __init__(
        self,
        collection_name: str | None = None,
        milvus_service: MilvusConnectionService | None = None,
        auto_create_collection: bool = True,
    ) -> None:
        self.collection_name = (
            collection_name
            or os.getenv("MILVUS_COLLECTION_NAME")
            or os.getenv("MILVUS_WORD_COLLECTION")
            or os.getenv("WORD_COLLECTION_NAME")
            or DEFAULT_WORD_COLLECTION
        )
        self.milvus_service = milvus_service or get_milvus_service()
        self.auto_create_collection = auto_create_collection

    def write(self, chunks: Sequence[WordChunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")

        vector_dim = len(vectors[0])
        self._ensure_collection(vector_dim)

        rows = [
            self._build_row(chunk, vector)
            for chunk, vector in zip(chunks, vectors)
        ]
        result = self.milvus_service.client.upsert(
            collection_name=self.collection_name,
            data=rows,
        )
        return int(result.get("upsert_count") or result.get("insert_count") or len(rows))

    def _ensure_collection(self, vector_dim: int) -> None:
        client = self.milvus_service.client
        if client.has_collection(self.collection_name):
            return
        if not self.auto_create_collection:
            raise RuntimeError(f"Milvus collection does not exist: {self.collection_name}")

        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("pymilvus is required to create Milvus collections.") from exc

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("file_name", DataType.VARCHAR, max_length=512)
        schema.add_field("file_type", DataType.VARCHAR, max_length=32)
        schema.add_field("chapter", DataType.VARCHAR, max_length=512)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _build_row(self, chunk: WordChunk, vector: Sequence[float]) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        return {
            "id": chunk.chunk_id,
            "vector": list(vector),
            "text": chunk.text,
            "file_name": metadata.get("file_name", ""),
            "file_type": metadata.get("file_type", WORD_FILE_TYPE),
            "chapter": metadata.get("chapter", ""),
            "section": metadata.get("section", ""),
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "heading_path": metadata.get("heading_path", ""),
            "token_count": chunk.token_count,
            "source_path": metadata.get("source_path", ""),
        }


class WordIngestionService:
    """End-to-end Word ingestion pipeline: parse, clean, chunk, embed, and index."""

    def __init__(
        self,
        embedding_provider: Any | None = None,
        collection_name: str | None = None,
        milvus_service: MilvusConnectionService | None = None,
        parser: WordParser | None = None,
        cleaner: WordCleaner | None = None,
        chunker: WordChunker | None = None,
        metadata_builder: WordMetadataBuilder | None = None,
        auto_create_collection: bool = True,
    ) -> None:
        self.parser = parser or WordParser()
        self.cleaner = cleaner or WordCleaner()
        self.chunker = chunker or WordChunker()
        self.metadata_builder = metadata_builder or WordMetadataBuilder()
        self.embedding_generator = WordEmbeddingGenerator(embedding_provider)
        self.writer = WordMilvusWriter(
            collection_name=collection_name,
            milvus_service=milvus_service,
            auto_create_collection=auto_create_collection,
        )

    def ingest(self, file_path: str | Path) -> WordIngestionResult:
        path = Path(file_path).expanduser().resolve()
        parsed_elements = self.parser.parse(path)
        cleaned_elements = self.cleaner.clean(parsed_elements)
        chunks = self.chunker.chunk(cleaned_elements, path)

        if not chunks:
            raise RuntimeError(f"No valid chunks were generated from Word file: {path}")

        chunks = self._attach_metadata(chunks, path)
        vectors = self.embedding_generator.embed([chunk.text for chunk in chunks])
        inserted_count = self.writer.write(chunks, vectors)

        return WordIngestionResult(
            source_path=str(path),
            file_name=path.name,
            file_type=WORD_FILE_TYPE,
            collection_name=self.writer.collection_name,
            document_hash=self._file_hash(path),
            parsed_elements_count=len(parsed_elements),
            cleaned_elements_count=len(cleaned_elements),
            chunks_count=len(chunks),
            inserted_count=inserted_count,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )

    def prepare_chunks(self, file_path: str | Path) -> list[WordChunk]:
        """Parse, clean, and chunk without embedding or Milvus writes."""
        path = Path(file_path).expanduser().resolve()
        parsed_elements = self.parser.parse(path)
        cleaned_elements = self.cleaner.clean(parsed_elements)
        chunks = self.chunker.chunk(cleaned_elements, path)
        return self._attach_metadata(chunks, path)

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Expose the parser output for debugging or API previews."""
        return [element.to_dict() for element in self.parser.parse(file_path)]

    def _attach_metadata(self, chunks: Sequence[WordChunk], path: Path) -> list[WordChunk]:
        enriched: list[WordChunk] = []
        for chunk in chunks:
            metadata = self.metadata_builder.build(chunk, path)
            enriched.append(
                WordChunk(
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    content_hash=chunk.content_hash,
                    metadata=metadata,
                )
            )
        return enriched

    def _file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
