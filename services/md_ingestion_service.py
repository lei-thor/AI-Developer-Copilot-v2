from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

try:
    from markdown_it import MarkdownIt
except ImportError:  # pragma: no cover - handled when Markdown ingestion is used.
    MarkdownIt = None  # type: ignore[assignment]

from services.milvus_service import MilvusConnectionService, get_milvus_service


MARKDOWN_FILE_TYPE = "markdown"
DEFAULT_MARKDOWN_COLLECTION = "developer_knowledge_chunks"


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass(frozen=True)
class MarkdownElement:
    """One semantic Markdown block produced by the parser."""

    element_type: str
    content: str
    level: int | None = None
    heading_path: tuple[str, ...] = ()
    heading_levels: tuple[int, ...] = ()
    language: str | None = None
    ordered: bool = False
    table_rows: tuple[tuple[str, ...], ...] = ()
    line_start: int | None = None
    line_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.element_type,
            "content": self.content,
            "level": self.level,
            "heading_path": list(self.heading_path),
            "heading_levels": list(self.heading_levels),
            "language": self.language,
            "ordered": self.ordered,
            "table_rows": [list(row) for row in self.table_rows],
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass(frozen=True)
class MarkdownDocument:
    file_id: str
    file_name: str
    file_type: str
    source_path: str
    title: str
    language: str
    content_hash: str
    elements: list[MarkdownElement]


@dataclass(frozen=True)
class MarkdownChunk:
    """Backward-compatible chunk DTO with full ingestion metadata."""

    chunk_index: int
    heading_path: str
    text: str
    content_hash: str
    chunk_id: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarkdownIngestionResult:
    source_path: str
    file_type: str
    title: str
    content_hash: str
    chunks: list[MarkdownChunk]
    collection_name: str = DEFAULT_MARKDOWN_COLLECTION
    parsed_elements_count: int = 0
    cleaned_elements_count: int = 0
    chunks_count: int = 0
    embedding_count: int = 0
    inserted_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)


class MarkdownParser:
    """Parse Markdown into headings, paragraphs, lists, code, tables, and quotes."""

    supported_suffixes = {".md", ".markdown"}

    def __init__(self, markdown_engine: Any | None = None) -> None:
        self.markdown_engine = markdown_engine

    def parse(self, file_path: str | Path) -> MarkdownDocument:
        path = self._validate_file(file_path)
        source = path.read_text(encoding="utf-8-sig")
        content_hash = self._hash_text(source)
        tokens = self._engine().parse(source)
        elements = self._tokens_to_elements(tokens)
        elements = self._attach_neighboring_table_captions(elements)
        title = self._extract_title(elements, path.stem)

        return MarkdownDocument(
            file_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{content_hash}")),
            file_name=path.name,
            file_type=MARKDOWN_FILE_TYPE,
            source_path=str(path),
            title=title,
            language=self._detect_language(source),
            content_hash=content_hash,
            elements=elements,
        )

    def _engine(self) -> Any:
        if self.markdown_engine is not None:
            return self.markdown_engine
        if MarkdownIt is None:
            raise RuntimeError(
                "markdown-it-py is required for Markdown ingestion. "
                "Install it in the law_rag environment."
            )

        engine = MarkdownIt("commonmark", {"html": True})
        for rule in ("table", "strikethrough"):
            try:
                engine.enable(rule)
            except Exception:
                continue
        return engine

    def _tokens_to_elements(self, tokens: Sequence[Any]) -> list[MarkdownElement]:
        elements: list[MarkdownElement] = []
        heading_stack: list[tuple[int, str]] = []
        index = 0

        while index < len(tokens):
            token = tokens[index]

            if token.type == "heading_open":
                inline = self._next_token(tokens, index, "inline")
                content = self._inline_content(inline)
                level = self._heading_level(token)
                heading_stack = self._push_heading(heading_stack, level, content)
                line_start, line_end = self._line_range(token)
                elements.append(
                    MarkdownElement(
                        element_type="heading",
                        content=content,
                        level=level,
                        heading_path=tuple(item[1] for item in heading_stack),
                        heading_levels=tuple(item[0] for item in heading_stack),
                        line_start=line_start,
                        line_end=line_end,
                    )
                )
                index = self._find_closing_token(tokens, index, "heading_open", "heading_close") + 1
                continue

            if token.type == "paragraph_open":
                inline = self._next_token(tokens, index, "inline")
                content = self._inline_content(inline)
                if content:
                    line_start, line_end = self._line_range(token)
                    element_type = "image" if self._is_standalone_image(content) else "paragraph"
                    elements.append(
                        MarkdownElement(
                            element_type=element_type,
                            content=content,
                            heading_path=tuple(item[1] for item in heading_stack),
                            heading_levels=tuple(item[0] for item in heading_stack),
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                index = self._find_closing_token(tokens, index, "paragraph_open", "paragraph_close") + 1
                continue

            if token.type in {"bullet_list_open", "ordered_list_open"}:
                close_index = self._find_matching_list_close(tokens, index)
                content = self._render_list(tokens[index : close_index + 1])
                if content:
                    line_start, line_end = self._line_range(token)
                    elements.append(
                        MarkdownElement(
                            element_type="list",
                            content=content,
                            ordered=token.type == "ordered_list_open",
                            heading_path=tuple(item[1] for item in heading_stack),
                            heading_levels=tuple(item[0] for item in heading_stack),
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                index = close_index + 1
                continue

            if token.type in {"fence", "code_block"}:
                language = self._fence_language(token)
                code = token.content.rstrip("\n")
                fence = f"```{language}\n{code}\n```"
                line_start, line_end = self._line_range(token)
                elements.append(
                    MarkdownElement(
                        element_type="code",
                        content=fence,
                        language=language or None,
                        heading_path=tuple(item[1] for item in heading_stack),
                        heading_levels=tuple(item[0] for item in heading_stack),
                        line_start=line_start,
                        line_end=line_end,
                    )
                )
                index += 1
                continue

            if token.type == "table_open":
                close_index = self._find_closing_token(tokens, index, "table_open", "table_close")
                rows = self._parse_table_rows(tokens[index : close_index + 1])
                content = self._render_table(rows)
                if content:
                    line_start, line_end = self._line_range(token)
                    elements.append(
                        MarkdownElement(
                            element_type="table",
                            content=content,
                            table_rows=tuple(rows),
                            heading_path=tuple(item[1] for item in heading_stack),
                            heading_levels=tuple(item[0] for item in heading_stack),
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                index = close_index + 1
                continue

            if token.type == "blockquote_open":
                close_index = self._find_closing_token(
                    tokens,
                    index,
                    "blockquote_open",
                    "blockquote_close",
                )
                content = self._render_blockquote(tokens[index : close_index + 1])
                if content:
                    line_start, line_end = self._line_range(token)
                    elements.append(
                        MarkdownElement(
                            element_type="blockquote",
                            content=content,
                            heading_path=tuple(item[1] for item in heading_stack),
                            heading_levels=tuple(item[0] for item in heading_stack),
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                index = close_index + 1
                continue

            if token.type in {"html_block", "html_inline"}:
                content = html_lib.unescape(str(token.content).strip())
                if content:
                    line_start, line_end = self._line_range(token)
                    elements.append(
                        MarkdownElement(
                            element_type="html",
                            content=content,
                            heading_path=tuple(item[1] for item in heading_stack),
                            heading_levels=tuple(item[0] for item in heading_stack),
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )

            index += 1

        return elements

    def _attach_neighboring_table_captions(
        self,
        elements: Sequence[MarkdownElement],
    ) -> list[MarkdownElement]:
        result: list[MarkdownElement] = []
        consumed: set[int] = set()

        for index, element in enumerate(elements):
            if index in consumed:
                continue

            if element.element_type == "table":
                previous = self._nearest_caption(elements, index, -1)
                following = self._nearest_caption(elements, index, 1)
                caption_index, caption = previous or following or (None, None)
                if caption_index is not None and caption is not None:
                    content = f"Table caption: {caption}\n\n{element.content}"
                    element = replace(element, content=content)
                    consumed.add(caption_index)

            result.append(element)

        return result

    def _nearest_caption(
        self,
        elements: Sequence[MarkdownElement],
        table_index: int,
        direction: int,
    ) -> tuple[int | None, str | None]:
        candidate_index = table_index + direction
        if candidate_index < 0 or candidate_index >= len(elements):
            return None, None

        candidate = elements[candidate_index]
        table = elements[table_index]
        if candidate.element_type != "paragraph":
            return None, None
        if candidate.heading_path != table.heading_path:
            return None, None
        if not self._looks_like_table_caption(candidate.content):
            return None, None
        return candidate_index, candidate.content.strip()

    def _looks_like_table_caption(self, content: str) -> bool:
        compact = " ".join(content.split())
        if not compact or len(compact) > 220 or compact.count("|") >= 2:
            return False
        return bool(
            re.match(
                r"^(table|tab\.?|tbl\.?)\s*[\w.-]+[\s:：.-]+.+",
                compact,
                flags=re.IGNORECASE,
            )
            or re.match(
                r"^[\u8868\u9644][\u8868]?\s*[\w.-]+[\s:：、.-]+.+",
                compact,
            )
            or re.match(r"^\([a-zA-Z0-9]+\)\s+\S+", compact)
        )

    def _parse_table_rows(self, tokens: Sequence[Any]) -> list[tuple[str, ...]]:
        rows: list[tuple[str, ...]] = []
        current_row: list[str] | None = None
        current_cell: list[str] | None = None

        for token in tokens:
            if token.type == "tr_open":
                current_row = []
            elif token.type in {"th_open", "td_open"}:
                current_cell = []
            elif token.type in {"inline", "code_inline"} and current_cell is not None:
                current_cell.append(str(token.content).strip())
            elif token.type in {"th_close", "td_close"} and current_row is not None:
                current_row.append(" ".join(current_cell or []).strip())
                current_cell = None
            elif token.type == "tr_close" and current_row is not None:
                if any(cell for cell in current_row):
                    rows.append(tuple(current_row))
                current_row = None

        return rows

    def _render_table(self, rows: Sequence[Sequence[str]]) -> str:
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [list(row) + [""] * (width - len(row)) for row in rows]
        lines = [
            "| " + " | ".join(normalized[0]) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
        return "\n".join(lines)

    def _render_list(self, tokens: Sequence[Any]) -> str:
        lines: list[str] = []
        stack: list[dict[str, Any]] = []
        pending_prefix: str | None = None

        for token in tokens:
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                ordered = token.type == "ordered_list_open"
                start = self._token_attr(token, "start", 1)
                stack.append({"ordered": ordered, "next": int(start)})
            elif token.type == "list_item_open" and stack:
                current = stack[-1]
                if current["ordered"]:
                    pending_prefix = f"{current['next']}."
                    current["next"] += 1
                else:
                    pending_prefix = "-"
            elif token.type == "inline" and pending_prefix is not None:
                indent = "  " * (len(stack) - 1)
                lines.append(f"{indent}{pending_prefix} {token.content.strip()}")
                pending_prefix = None
            elif token.type in {"bullet_list_close", "ordered_list_close"} and stack:
                stack.pop()

        return "\n".join(lines).strip()

    def _render_blockquote(self, tokens: Sequence[Any]) -> str:
        lines: list[str] = []
        for token in tokens:
            if token.type == "inline":
                lines.extend(f"> {line}" for line in token.content.splitlines() if line.strip())
        return "\n".join(lines).strip()

    def _next_token(self, tokens: Sequence[Any], index: int, token_type: str) -> Any | None:
        for candidate in tokens[index + 1 : index + 4]:
            if candidate.type == token_type:
                return candidate
        return None

    def _find_closing_token(
        self,
        tokens: Sequence[Any],
        start: int,
        open_type: str,
        close_type: str,
    ) -> int:
        depth = 0
        for index in range(start, len(tokens)):
            if tokens[index].type == open_type:
                depth += 1
            elif tokens[index].type == close_type:
                depth -= 1
                if depth == 0:
                    return index
        return len(tokens) - 1

    def _find_matching_list_close(self, tokens: Sequence[Any], start: int) -> int:
        depth = 0
        for index in range(start, len(tokens)):
            if tokens[index].type in {"bullet_list_open", "ordered_list_open"}:
                depth += 1
            elif tokens[index].type in {"bullet_list_close", "ordered_list_close"}:
                depth -= 1
                if depth == 0:
                    return index
        return len(tokens) - 1

    def _inline_content(self, token: Any | None) -> str:
        return str(token.content).strip() if token is not None else ""

    def _heading_level(self, token: Any) -> int:
        tag = str(getattr(token, "tag", "h1"))
        match = re.search(r"(\d+)$", tag)
        return int(match.group(1)) if match else 1

    def _push_heading(
        self,
        stack: list[tuple[int, str]],
        level: int,
        content: str,
    ) -> list[tuple[int, str]]:
        stack = [item for item in stack if item[0] < level]
        stack.append((level, content))
        return stack

    def _fence_language(self, token: Any) -> str:
        info = str(getattr(token, "info", "") or "").strip()
        return info.split()[0] if info else ""

    def _token_attr(self, token: Any, name: str, default: Any = None) -> Any:
        attrs = getattr(token, "attrs", None) or {}
        if isinstance(attrs, dict):
            return attrs.get(name, default)
        try:
            return dict(attrs).get(name, default)
        except (TypeError, ValueError):
            return default

    def _line_range(self, token: Any) -> tuple[int | None, int | None]:
        token_map = getattr(token, "map", None)
        if not token_map:
            return None, None
        return int(token_map[0]) + 1, int(token_map[1])

    def _is_standalone_image(self, content: str) -> bool:
        return bool(re.fullmatch(r"\s*!\[[^\]]*\]\([^)]+\)\s*", content))

    def _extract_title(self, elements: Sequence[MarkdownElement], fallback: str) -> str:
        for element in elements:
            if element.element_type == "heading" and element.level == 1:
                return element.content
        return fallback

    def _detect_language(self, source: str) -> str:
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", source))
        latin_count = len(re.findall(r"[A-Za-z]", source))
        return "zh" if cjk_count > latin_count else "en"

    def _validate_file(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Markdown file does not exist: {path}")
        if path.suffix.lower() not in self.supported_suffixes:
            raise ValueError("Only .md and .markdown files are supported.")
        return path

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MarkdownCleaner:
    """Clean Markdown conservatively while keeping semantic blocks intact."""

    protected_types = {"code", "table", "list", "blockquote", "html"}

    def clean(self, document: MarkdownDocument) -> MarkdownDocument:
        cleaned: list[MarkdownElement] = []
        for element in document.elements:
            content = self._clean_content(element)
            if not content:
                continue

            current = replace(element, content=content)
            if (
                cleaned
                and current.element_type == "paragraph"
                and cleaned[-1].element_type == "paragraph"
                and current.heading_path == cleaned[-1].heading_path
            ):
                cleaned[-1] = replace(
                    cleaned[-1],
                    content=f"{cleaned[-1].content}\n\n{current.content}",
                    line_end=current.line_end,
                )
            else:
                cleaned.append(current)

        return replace(document, elements=cleaned)

    def _clean_content(self, element: MarkdownElement) -> str:
        content = element.content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if element.element_type in self.protected_types:
            return re.sub(r"\n{4,}", "\n\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        return content.strip()


class MarkdownTokenEstimator:
    """Approximate token counting without adding a tokenizer dependency."""

    pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")

    def count(self, text: str) -> int:
        return len(self.pattern.findall(text))

    def split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？!?；;.])\s+|(?<=\n)\s*", text)
        return [part.strip() for part in parts if part.strip()]


class MarkdownMetadataBuilder:
    def build(
        self,
        document: MarkdownDocument,
        chunk_index: int,
        text: str,
        elements: Sequence[MarkdownElement],
        token_count: int,
    ) -> dict[str, Any]:
        heading_path = self._heading_path(elements)
        heading_levels = self._heading_levels(elements)
        page_types = sorted({element.element_type for element in elements})
        line_start = min(
            (element.line_start for element in elements if element.line_start is not None),
            default=None,
        )
        line_end = max(
            (element.line_end for element in elements if element.line_end is not None),
            default=None,
        )
        return {
            "file_id": document.file_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "chapter": heading_path[0] if heading_path else document.title,
            "section": heading_path[-1] if len(heading_path) > 1 else "",
            "heading": " > ".join(heading_path),
            "heading_path": " > ".join(heading_path),
            "heading_levels": list(heading_levels),
            "chunk_index": chunk_index,
            "block_types": page_types,
            "block_type": ",".join(page_types),
            "line_start": line_start,
            "line_end": line_end,
            "language": document.language,
            "parser": "markdown-it-py",
            "source_path": document.source_path,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count": token_count,
        }

    def _heading_path(self, elements: Sequence[MarkdownElement]) -> tuple[str, ...]:
        for element in elements:
            if element.heading_path:
                return element.heading_path
        return ()

    def _heading_levels(self, elements: Sequence[MarkdownElement]) -> tuple[int, ...]:
        for element in elements:
            if element.heading_levels:
                return element.heading_levels
        return tuple(range(1, len(self._heading_path(elements)) + 1))


class MarkdownChunkBuilder:
    """Build chunks by heading context and semantic block boundaries."""

    atomic_types = {"code", "table", "list", "blockquote", "html", "image"}

    def __init__(
        self,
        target_tokens: int = 650,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        token_estimator: MarkdownTokenEstimator | None = None,
        metadata_builder: MarkdownMetadataBuilder | None = None,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_estimator = token_estimator or MarkdownTokenEstimator()
        self.metadata_builder = metadata_builder or MarkdownMetadataBuilder()

    def build(self, document: MarkdownDocument) -> list[MarkdownChunk]:
        chunks: list[MarkdownChunk] = []
        current: list[MarkdownElement] = []
        current_path: tuple[str, ...] = ()

        for element in document.elements:
            units = self._split_element(element)
            for unit in units:
                if current and unit.heading_path != current_path:
                    chunks.append(self._build_chunk(document, len(chunks), current))
                    current = []

                if not current:
                    current_path = unit.heading_path

                unit_tokens = self.token_estimator.count(unit.content)
                current_tokens = self._elements_tokens(current)
                should_flush = (
                    bool(current)
                    and current_tokens >= self.target_tokens
                    or bool(current)
                    and current_tokens + unit_tokens > self.max_tokens
                )
                if should_flush:
                    chunks.append(self._build_chunk(document, len(chunks), current))
                    overlap = self._overlap_elements(current, unit.heading_path)
                    current = overlap
                    current_path = unit.heading_path

                current.append(unit)

        if current:
            chunks.append(self._build_chunk(document, len(chunks), current))
        return chunks

    def _split_element(self, element: MarkdownElement) -> list[MarkdownElement]:
        if element.element_type in self.atomic_types:
            return [element]
        if self.token_estimator.count(element.content) <= self.max_tokens:
            return [element]

        parts: list[MarkdownElement] = []
        sentences = self.token_estimator.split_sentences(element.content)
        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in sentences:
            sentence_tokens = self.token_estimator.count(sentence)
            if buffer and buffer_tokens + sentence_tokens > self.max_tokens:
                parts.append(replace(element, content="\n".join(buffer)))
                buffer = []
                buffer_tokens = 0
            if sentence_tokens > self.max_tokens:
                for offset in range(0, len(sentence), max(self.max_tokens * 2, 1)):
                    part = sentence[offset : offset + self.max_tokens * 2].strip()
                    if part:
                        parts.append(replace(element, content=part))
                continue
            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        if buffer:
            parts.append(replace(element, content="\n".join(buffer)))
        return parts or [element]

    def _build_chunk(
        self,
        document: MarkdownDocument,
        chunk_index: int,
        elements: Sequence[MarkdownElement],
    ) -> MarkdownChunk:
        heading_path = self._heading_path(elements)
        heading_levels = self._heading_levels(elements)
        heading_prefix = self._render_heading_prefix(heading_path, heading_levels)
        body = "\n\n".join(
            element.content
            for element in elements
            if element.element_type != "heading" and element.content.strip()
        ).strip()
        text = f"{heading_prefix}\n\n{body}".strip() if body and heading_prefix else body or heading_prefix
        token_count = self.token_estimator.count(text)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata = self.metadata_builder.build(
            document=document,
            chunk_index=chunk_index,
            text=text,
            elements=elements,
            token_count=token_count,
        )
        chunk_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document.file_id}:{chunk_index}:{content_hash}",
            )
        )
        return MarkdownChunk(
            chunk_index=chunk_index,
            heading_path=" > ".join(heading_path),
            text=text,
            content_hash=content_hash,
            chunk_id=chunk_id,
            token_count=token_count,
            metadata=metadata,
        )

    def _overlap_elements(
        self,
        elements: Sequence[MarkdownElement],
        heading_path: tuple[str, ...],
    ) -> list[MarkdownElement]:
        overlap: list[MarkdownElement] = []
        count = 0
        for element in reversed(elements):
            if element.heading_path != heading_path or element.element_type != "paragraph":
                continue
            element_tokens = self.token_estimator.count(element.content)
            if overlap and count + element_tokens > self.overlap_tokens:
                break
            overlap.insert(0, element)
            count += element_tokens
            if count >= self.overlap_tokens:
                break
        return overlap

    def _elements_tokens(self, elements: Sequence[MarkdownElement]) -> int:
        return self.token_estimator.count(
            "\n\n".join(element.content for element in elements)
        )

    def _heading_path(self, elements: Sequence[MarkdownElement]) -> tuple[str, ...]:
        for element in elements:
            if element.heading_path:
                return element.heading_path
        return ()

    def _heading_levels(self, elements: Sequence[MarkdownElement]) -> tuple[int, ...]:
        for element in elements:
            if element.heading_levels:
                return element.heading_levels
        return tuple(range(1, len(self._heading_path(elements)) + 1))

    def _render_heading_prefix(
        self,
        heading_path: Sequence[str],
        heading_levels: Sequence[int],
    ) -> str:
        lines: list[str] = []
        for index, title in enumerate(heading_path):
            level = heading_levels[index] if index < len(heading_levels) else index + 1
            lines.append(f"{'#' * max(level, 1)} {title}")
        return "\n".join(lines)


class MarkdownEmbeddingGenerator:
    def __init__(self, embedding_provider: Any | None = None) -> None:
        self.embedding_provider = embedding_provider

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        provider = self.embedding_provider
        if provider is None:
            from services.embedding_service import get_embedding_service

            provider = get_embedding_service()

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


class MarkdownMilvusWriter:
    """Persist Markdown chunks through the existing Milvus connection service."""

    def __init__(
        self,
        collection_name: str | None = None,
        milvus_service: MilvusConnectionService | None = None,
        auto_create_collection: bool = True,
    ) -> None:
        self.collection_name = (
            collection_name
            or os.getenv("MILVUS_COLLECTION_NAME")
            or os.getenv("MILVUS_MARKDOWN_COLLECTION")
            or DEFAULT_MARKDOWN_COLLECTION
        )
        self.milvus_service = milvus_service or get_milvus_service()
        self.auto_create_collection = auto_create_collection

    def write(
        self,
        chunks: Sequence[MarkdownChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")

        self._ensure_collection(len(vectors[0]))
        rows = [self._build_row(chunk, vector) for chunk, vector in zip(chunks, vectors)]
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

    def _build_row(
        self,
        chunk: MarkdownChunk,
        vector: Sequence[float],
    ) -> dict[str, Any]:
        metadata = dict(chunk.metadata)
        return {
            "id": chunk.chunk_id,
            "vector": list(vector),
            "text": chunk.text,
            "file_name": metadata.get("file_name", ""),
            "file_type": metadata.get("file_type", MARKDOWN_FILE_TYPE),
            "chapter": metadata.get("chapter", ""),
            "section": metadata.get("section", ""),
            "chunk_index": chunk.chunk_index,
            "heading_path": metadata.get("heading_path", chunk.heading_path),
            "block_type": metadata.get("block_type", ""),
            "token_count": chunk.token_count,
            "line_start": metadata.get("line_start"),
            "line_end": metadata.get("line_end"),
            "parser": metadata.get("parser", "markdown-it-py"),
            "language": metadata.get("language", ""),
            "source_path": metadata.get("source_path", ""),
            "content_hash": chunk.content_hash,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }


class MarkdownIngestionService:
    """End-to-end Markdown pipeline: parse, clean, chunk, embed, and index."""

    supported_suffixes = {".md", ".markdown"}

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        target_tokens: int = 650,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        embedding_provider: Any | None = None,
        collection_name: str | None = None,
        milvus_service: MilvusConnectionService | None = None,
        parser: MarkdownParser | None = None,
        cleaner: MarkdownCleaner | None = None,
        chunker: MarkdownChunkBuilder | None = None,
        auto_create_collection: bool = True,
    ) -> None:
        self.parser = parser or MarkdownParser()
        self.cleaner = cleaner or MarkdownCleaner()
        self.chunker = chunker or MarkdownChunkBuilder(
            target_tokens=target_tokens,
            max_tokens=chunk_size or max_tokens,
            overlap_tokens=chunk_overlap or overlap_tokens,
        )
        self.embedding_generator = MarkdownEmbeddingGenerator(embedding_provider)
        self.writer = MarkdownMilvusWriter(
            collection_name=collection_name,
            milvus_service=milvus_service,
            auto_create_collection=auto_create_collection,
        )

    def ingest(self, file_path: str | Path) -> MarkdownIngestionResult:
        document = self.parse_document(file_path)
        cleaned = self.clean_document(document)
        chunks = self.build_chunks(cleaned)
        vectors = self.generate_embeddings(chunks)
        inserted_count = self.save_to_milvus(chunks, vectors)
        return MarkdownIngestionResult(
            source_path=document.source_path,
            file_type=document.file_type,
            title=document.title,
            content_hash=document.content_hash,
            chunks=chunks,
            collection_name=self.writer.collection_name,
            parsed_elements_count=len(document.elements),
            cleaned_elements_count=len(cleaned.elements),
            chunks_count=len(chunks),
            embedding_count=len(vectors),
            inserted_count=inserted_count,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )

    def prepare_chunks(self, file_path: str | Path) -> list[MarkdownChunk]:
        """Parse, clean, and chunk without embedding or Milvus writes."""
        document = self.parse_document(file_path)
        return self.build_chunks(self.clean_document(document))

    def parse_document(self, file_path: str | Path) -> MarkdownDocument:
        return self.parser.parse(file_path)

    def clean_document(self, document: MarkdownDocument) -> MarkdownDocument:
        return self.cleaner.clean(document)

    def build_chunks(self, document: MarkdownDocument) -> list[MarkdownChunk]:
        chunks = self.chunker.build(document)
        if not chunks:
            raise RuntimeError(f"No valid chunks were generated from Markdown: {document.source_path}")
        return chunks

    def generate_embeddings(self, chunks: Sequence[MarkdownChunk]) -> list[list[float]]:
        return self.embedding_generator.embed([chunk.text for chunk in chunks])

    def save_to_milvus(
        self,
        chunks: Sequence[MarkdownChunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        return self.writer.write(chunks, vectors)

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Expose structured parser output for debugging and review."""
        return [element.to_dict() for element in self.parse_document(file_path).elements]

    def _validate_file(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Markdown file does not exist: {path}")
        if path.suffix.lower() not in self.supported_suffixes:
            raise ValueError("Only .md and .markdown files are supported.")
        return path


MarkdownIngestionService = MarkdownIngestionService
