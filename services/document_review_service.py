from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence


logger = logging.getLogger(__name__)

ReviewStatus = Literal["PENDING", "APPROVED", "REJECTED", "MODIFIED"]
IssueSeverity = Literal["INFO", "WARNING", "ERROR"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES_ROOT = PROJECT_ROOT / "files"
DEFAULT_REVIEW_STATE_PATH = FILES_ROOT / "document_review_state.json"


@dataclass(frozen=True)
class ReviewBlock:
    id: str
    type: str
    level: int | None
    content: str
    page_number: int | None
    bbox: list[float] | None
    chunk_id: str | None


@dataclass(frozen=True)
class ReviewPage:
    page_number: int
    blocks: list[ReviewBlock]


@dataclass(frozen=True)
class ReviewTreeNode:
    id: str
    label: str
    node_type: str
    page_number: int | None
    chunk_id: str | None
    children: list["ReviewTreeNode"]


@dataclass(frozen=True)
class ReviewChunk:
    chunk_id: str
    chunk_index: int
    token_count: int
    text: str
    metadata: dict[str, Any]
    review_status: ReviewStatus
    review_comment: str
    updated_time: str | None


@dataclass(frozen=True)
class QualityIssue:
    id: str
    severity: IssueSeverity
    code: str
    message: str
    chunk_id: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class QualityReport:
    document_id: str
    quality_score: int
    issue_count: int
    error_count: int
    warning_count: int
    issues: list[QualityIssue]


@dataclass(frozen=True)
class DocumentReview:
    document_id: str
    file_name: str
    file_type: str
    page_count: int
    pdf_url: str | None
    source_text_url: str | None
    pages: list[ReviewPage]
    document_tree: ReviewTreeNode
    chunks: list[ReviewChunk]
    quality_report: QualityReport


@dataclass(frozen=True)
class ChunkReviewInput:
    review_status: ReviewStatus
    review_comment: str = ""
    text: str | None = None
    metadata: dict[str, Any] | None = None


class TableContentNormalizer:
    """Normalize parser table output into reviewable caption + body text."""

    caption_keys = (
        "table_caption",
        "caption",
        "table_title",
        "title",
    )
    footnote_keys = (
        "table_footnote",
        "footnote",
        "note",
    )

    def from_mineru_item(self, item: dict[str, Any]) -> str:
        captions = self._collect_lines(item, self.caption_keys)
        footnotes = self._collect_lines(item, self.footnote_keys)
        raw_body = (
            item.get("table_body")
            or item.get("table")
            or item.get("html")
            or item.get("content")
            or ""
        )
        body_text = self.to_text(raw_body)
        leading_captions, body_text = self._split_leading_captions(body_text)
        captions = self._dedupe_lines([*captions, *leading_captions])

        parts: list[str] = []
        if captions:
            parts.extend(f"Table caption: {caption}" for caption in captions)
        if body_text:
            parts.append(body_text)
        if footnotes:
            parts.extend(f"Table note: {footnote}" for footnote in self._dedupe_lines(footnotes))
        return "\n\n".join(parts).strip()

    def to_text(self, table: Any) -> str:
        if isinstance(table, str):
            if table.strip().startswith("|"):
                return table.strip()
            return self._html_to_text(table)
        if isinstance(table, list):
            return self._rows_to_text(table)
        return str(table).strip()

    def _rows_to_text(self, rows: Sequence[Any]) -> str:
        lines: list[str] = []
        for row in rows:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                cells = [str(cell).strip() for cell in row if str(cell).strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)

    def _html_to_text(self, html_table: str) -> str:
        text = html_lib.unescape(html_table)
        text = re.sub(r"<\s*/\s*t[dh]\s*>\s*<\s*t[dh][^>]*>", " | ", text, flags=re.IGNORECASE)
        text = re.sub(r"<\s*/\s*tr\s*>\s*<\s*tr[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        lines = [re.sub(r"\s+", " ", line).strip(" |") for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def _split_leading_captions(self, table_text: str) -> tuple[list[str], str]:
        lines = [line.strip() for line in table_text.splitlines() if line.strip()]
        captions: list[str] = []
        while lines and self._looks_like_caption_line(lines[0]):
            captions.append(lines.pop(0))
        return captions, "\n".join(lines).strip()

    def _looks_like_caption_line(self, line: str) -> bool:
        compact = line.strip()
        if not compact or len(compact) > 220:
            return False
        if compact.count("|") >= 2:
            return False
        return bool(
            re.match(r"^(table|tab\.?|tbl\.?)\s*[\w.-]+[\s:：.-]+.+", compact, flags=re.IGNORECASE)
            or re.match(r"^[\u8868\u9644][\u8868]?\s*[\w.-]+[\s:：、.-]+.+", compact)
            or re.match(r"^\([a-zA-Z0-9]+\)\s+\S+", compact)
        )

    def _collect_lines(self, item: dict[str, Any], keys: Sequence[str]) -> list[str]:
        lines: list[str] = []
        for key in keys:
            lines.extend(self._lines_from_value(item.get(key)))
        return self._dedupe_lines(lines)

    def _lines_from_value(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, list):
            lines: list[str] = []
            for item in value:
                lines.extend(self._lines_from_value(item))
            return lines
        return []

    def _dedupe_lines(self, lines: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for line in lines:
            if line and line not in seen:
                result.append(line)
                seen.add(line)
        return result


class DocumentReviewRepository:
    """Filesystem-backed repository; replace with MySQL repositories later."""

    def __init__(
        self,
        files_root: Path = FILES_ROOT,
        review_state_path: Path = DEFAULT_REVIEW_STATE_PATH,
    ) -> None:
        self.files_root = files_root
        self.review_state_path = review_state_path

    def list_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for preview_path in sorted(self.files_root.glob("*_chunks_preview.json")):
            try:
                chunks = self._read_preview(preview_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load preview file %s: %s", preview_path, exc)
                continue
            if not chunks:
                continue

            first_metadata = chunks[0].get("metadata", {})
            document_id = str(first_metadata.get("file_id") or preview_path.stem)
            file_name = str(first_metadata.get("file_name") or preview_path.name)
            file_type = str(first_metadata.get("file_type") or "unknown")
            documents.append(
                {
                    "document_id": document_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "preview_path": str(preview_path),
                    "chunk_count": len(chunks),
                    "page_count": self._page_count(chunks),
                }
            )
        return documents

    def load_chunks(self, document_id: str) -> list[dict[str, Any]]:
        for document in self.list_documents():
            if document["document_id"] == document_id or document["file_name"] == document_id:
                return self._read_preview(Path(document["preview_path"]))
        raise FileNotFoundError(f"Document review data not found: {document_id}")

    def find_document(self, document_id: str) -> dict[str, Any]:
        for document in self.list_documents():
            if document["document_id"] == document_id or document["file_name"] == document_id:
                return document
        raise FileNotFoundError(f"Document review data not found: {document_id}")

    def pdf_path_for(self, file_name: str) -> Path | None:
        path = self.source_path_for(file_name)
        if path.exists() and path.suffix.lower() == ".pdf":
            return path
        return None

    def source_path_for(self, file_name: str) -> Path:
        return self.files_root / file_name

    def load_parser_blocks(self, document_id: str) -> list[ReviewBlock]:
        document = self.find_document(document_id)
        if document["file_type"] != "pdf":
            return []

        artifact_path = self._find_mineru_artifact(document["file_name"])
        if artifact_path is None:
            return []

        try:
            with zipfile.ZipFile(artifact_path) as archive:
                content_name = self._content_list_name(archive)
                if content_name is None:
                    return []
                content_list = json.loads(archive.read(content_name).decode("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            logger.warning("Failed to load MinerU artifact %s: %s", artifact_path, exc)
            return []

        blocks: list[ReviewBlock] = []
        for index, item in enumerate(content_list):
            block = self._review_block_from_mineru_item(document_id, index, item)
            if block is not None:
                blocks.append(block)
        return blocks

    def load_review_state(self) -> dict[str, Any]:
        if not self.review_state_path.exists():
            return {"chunks": {}}
        try:
            return json.loads(self.review_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load review state: %s", exc)
            return {"chunks": {}}

    def save_chunk_review(self, chunk_id: str, data: ChunkReviewInput) -> dict[str, Any]:
        state = self.load_review_state()
        chunks = state.setdefault("chunks", {})
        record = {
            "chunk_id": chunk_id,
            "review_status": data.review_status,
            "review_comment": data.review_comment,
            "text": data.text,
            "metadata": data.metadata,
            "updated_time": datetime.now(timezone.utc).isoformat(),
        }
        chunks[chunk_id] = record
        self.review_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    def _read_preview(self, preview_path: Path) -> list[dict[str, Any]]:
        payload = json.loads(preview_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Preview file must contain a list: {preview_path}")
        return payload

    def _page_count(self, chunks: Sequence[dict[str, Any]]) -> int:
        pages: set[int] = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            page_numbers = metadata.get("page_numbers") or []
            if isinstance(page_numbers, list):
                pages.update(int(page) for page in page_numbers if page)
            elif metadata.get("page_number"):
                pages.add(int(metadata["page_number"]))
        return max(pages) if pages else 0

    def _find_mineru_artifact(self, file_name: str) -> Path | None:
        output_root = self.files_root / "mineru_outputs"
        candidates = sorted(
            output_root.glob("*.zip"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None

        source_path = self.pdf_path_for(file_name)
        source_hash = self._sha256_path(source_path) if source_path else None
        if source_hash:
            for candidate in candidates:
                if self._zip_origin_pdf_hash(candidate) == source_hash:
                    return candidate

        for candidate in candidates:
            try:
                with zipfile.ZipFile(candidate) as archive:
                    if self._content_list_name(archive):
                        return candidate
            except (OSError, zipfile.BadZipFile):
                continue
        return None

    def _content_list_name(self, archive: zipfile.ZipFile) -> str | None:
        names = archive.namelist()
        for suffix in ("_content_list.json", "_content_list_v2.json"):
            for name in names:
                if name.endswith(suffix):
                    return name
        return None

    def _zip_origin_pdf_hash(self, archive_path: Path) -> str | None:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                origin_name = next((name for name in archive.namelist() if name.endswith("_origin.pdf")), None)
                if origin_name is None:
                    return None
                return hashlib.sha256(archive.read(origin_name)).hexdigest()
        except (OSError, zipfile.BadZipFile):
            return None

    def _sha256_path(self, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _review_block_from_mineru_item(
        self,
        document_id: str,
        index: int,
        item: dict[str, Any],
    ) -> ReviewBlock | None:
        raw_type = str(item.get("type") or "paragraph")
        if raw_type == "page_number":
            return None

        content = self._mineru_block_content(item)
        if not content:
            return None

        page_idx = item.get("page_idx")
        page_number = int(page_idx) + 1 if page_idx is not None else None
        block_type = self._normalize_mineru_type(raw_type, item)
        level = item.get("text_level") or item.get("level")
        return ReviewBlock(
            id=f"{document_id}:block:{index}",
            type=block_type,
            level=int(level) if level else None,
            content=content,
            page_number=page_number,
            bbox=item.get("bbox"),
            chunk_id=None,
        )

    def _normalize_mineru_type(self, raw_type: str, item: dict[str, Any]) -> str:
        if raw_type in {"title", "heading"} or item.get("text_level"):
            return "heading"
        if raw_type in {"image", "chart"}:
            return "image"
        if raw_type == "equation":
            return "formula"
        if raw_type in {"table"}:
            return "table"
        if raw_type in {"ref_text", "page_footnote", "aside_text"}:
            return raw_type
        return "paragraph"

    def _mineru_block_content(self, item: dict[str, Any]) -> str:
        if item.get("type") == "table":
            return TableContentNormalizer().from_mineru_item(item)

        content_parts: list[str] = []
        for key in (
            "text",
            "content",
            "table_body",
            "image_caption",
            "image_footnote",
            "chart_caption",
            "chart_footnote",
            "table_caption",
            "table_footnote",
        ):
            value = item.get(key)
            if isinstance(value, str):
                content_parts.append(value)
            elif isinstance(value, list):
                content_parts.extend(str(part) for part in value if part)

        content = "\n".join(part.strip() for part in content_parts if str(part).strip())
        return content.strip()

    def _html_table_to_text(self, html_table: str) -> str:
        text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", html_table)
        text = re.sub(r"</tr>\s*<tr[^>]*>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


class DocumentReviewService:
    """Build review DTOs and quality reports from structured chunk outputs."""

    def __init__(self, repository: DocumentReviewRepository | None = None) -> None:
        self.repository = repository or DocumentReviewRepository()

    def list_documents(self) -> list[dict[str, Any]]:
        return self.repository.list_documents()

    def get_document_review(self, document_id: str) -> DocumentReview:
        document = self.repository.find_document(document_id)
        raw_chunks = self.repository.load_chunks(document_id)
        chunks = self._review_chunks(raw_chunks)
        pages = self._pages_from_parser_blocks(document_id) or self._pages_from_chunks(chunks)
        quality_report = self.get_quality_report(document_id, chunks=chunks)
        pdf_path = self.repository.pdf_path_for(document["file_name"])
        source_path = self.repository.source_path_for(document["file_name"])
        source_text_url = (
            f"/api/document-review/{document['document_id']}/source-text"
            if source_path.exists() and source_path.suffix.lower() in {".md", ".markdown", ".txt"}
            else None
        )

        return DocumentReview(
            document_id=document["document_id"],
            file_name=document["file_name"],
            file_type=document["file_type"],
            page_count=document["page_count"],
            pdf_url=f"/api/document-review/{document['document_id']}/source"
            if pdf_path is not None
            else None,
            source_text_url=source_text_url,
            pages=pages,
            document_tree=self._tree_from_chunks(document["file_name"], chunks),
            chunks=chunks,
            quality_report=quality_report,
        )

    def get_chunks(self, document_id: str) -> list[ReviewChunk]:
        return self._review_chunks(self.repository.load_chunks(document_id))

    def update_chunk_review(self, chunk_id: str, data: ChunkReviewInput) -> dict[str, Any]:
        if data.review_status not in {"PENDING", "APPROVED", "REJECTED", "MODIFIED"}:
            raise ValueError(f"Unsupported review_status: {data.review_status}")
        return self.repository.save_chunk_review(chunk_id, data)

    def get_quality_report(
        self,
        document_id: str,
        chunks: Sequence[ReviewChunk] | None = None,
    ) -> QualityReport:
        review_chunks = list(chunks) if chunks is not None else self.get_chunks(document_id)
        issues = QualityAnalyzer().analyze(document_id, review_chunks)
        error_count = sum(1 for issue in issues if issue.severity == "ERROR")
        warning_count = sum(1 for issue in issues if issue.severity == "WARNING")
        score = max(0, 100 - error_count * 15 - warning_count * 5)
        return QualityReport(
            document_id=document_id,
            quality_score=score,
            issue_count=len(issues),
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
        )

    def get_source_pdf_path(self, document_id: str) -> Path:
        document = self.repository.find_document(document_id)
        path = self.repository.pdf_path_for(document["file_name"])
        if path is None:
            raise FileNotFoundError(f"Source PDF not found for document: {document_id}")
        return path

    def get_source_text(self, document_id: str) -> str:
        document = self.repository.find_document(document_id)
        path = self.repository.source_path_for(document["file_name"])
        if not path.exists() or path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            raise FileNotFoundError(f"Source text file not found for document: {document_id}")
        return path.read_text(encoding="utf-8-sig", errors="replace")

    def render_pdf_page_png(
        self,
        document_id: str,
        page_number: int,
        zoom: float = 1.8,
    ) -> bytes:
        """Render one PDF page for browser review without relying on PDF plugins."""
        if page_number < 1:
            raise ValueError("page_number must be greater than 0")

        source_path = self.get_source_pdf_path(document_id)
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to render PDF pages.") from exc

        with fitz.open(source_path) as document:
            if page_number > document.page_count:
                raise ValueError(
                    f"page_number {page_number} exceeds PDF page count {document.page_count}"
                )
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return pixmap.tobytes("png")

    def _review_chunks(self, raw_chunks: Sequence[dict[str, Any]]) -> list[ReviewChunk]:
        state = self.repository.load_review_state().get("chunks", {})
        chunks: list[ReviewChunk] = []
        for raw in sorted(raw_chunks, key=lambda item: item.get("chunk_index", 0)):
            chunk_id = str(raw.get("chunk_id") or raw.get("id") or "")
            review_state = state.get(chunk_id, {})
            metadata = dict(raw.get("metadata") or {})
            text = review_state.get("text") or raw.get("text") or ""
            if isinstance(review_state.get("metadata"), dict):
                metadata.update(review_state["metadata"])

            chunks.append(
                ReviewChunk(
                    chunk_id=chunk_id,
                    chunk_index=int(raw.get("chunk_index") or metadata.get("chunk_index") or 0),
                    token_count=int(raw.get("token_count") or metadata.get("token_count") or 0),
                    text=str(text),
                    metadata=metadata,
                    review_status=review_state.get("review_status", "PENDING"),
                    review_comment=review_state.get("review_comment", ""),
                    updated_time=review_state.get("updated_time"),
                )
            )
        return chunks

    def _pages_from_chunks(self, chunks: Sequence[ReviewChunk]) -> list[ReviewPage]:
        page_blocks: dict[int, list[ReviewBlock]] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            page_numbers = metadata.get("page_numbers") or [metadata.get("page_number")]
            if not isinstance(page_numbers, list):
                page_numbers = [page_numbers]
            pages = [int(page) for page in page_numbers if page]
            if not pages:
                pages = [0]

            for page in pages:
                page_blocks.setdefault(page, []).append(
                    ReviewBlock(
                        id=f"{chunk.chunk_id}:page:{page}",
                        type=str(metadata.get("block_type") or "chunk"),
                        level=None,
                        content=chunk.text,
                        page_number=page,
                        bbox=metadata.get("bbox"),
                        chunk_id=chunk.chunk_id,
                    )
                )

        return [
            ReviewPage(page_number=page, blocks=blocks)
            for page, blocks in sorted(page_blocks.items(), key=lambda item: item[0])
        ]

    def _pages_from_parser_blocks(self, document_id: str) -> list[ReviewPage]:
        parser_blocks = self.repository.load_parser_blocks(document_id)
        if not parser_blocks:
            return []

        page_blocks: dict[int, list[ReviewBlock]] = {}
        for block in parser_blocks:
            page = block.page_number or 0
            page_blocks.setdefault(page, []).append(block)

        return [
            ReviewPage(page_number=page, blocks=blocks)
            for page, blocks in sorted(page_blocks.items(), key=lambda item: item[0])
        ]

    def _tree_from_chunks(self, file_name: str, chunks: Sequence[ReviewChunk]) -> ReviewTreeNode:
        root = MutableTreeNode(label=file_name, node_type="document", page_number=None, chunk_id=None)
        for chunk in chunks:
            heading = str(chunk.metadata.get("heading") or chunk.metadata.get("heading_path") or "")
            parts = [part.strip() for part in heading.split(">") if part.strip()]
            current = root
            for part in parts:
                current = current.child(part, "heading", chunk.metadata.get("page_number"))
            current.children.append(
                MutableTreeNode(
                    label=f"Chunk {chunk.chunk_index}",
                    node_type="chunk",
                    page_number=chunk.metadata.get("page_number"),
                    chunk_id=chunk.chunk_id,
                )
            )
        return root.freeze()


class MutableTreeNode:
    def __init__(
        self,
        label: str,
        node_type: str,
        page_number: int | None,
        chunk_id: str | None,
    ) -> None:
        self.label = label
        self.node_type = node_type
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.children: list[MutableTreeNode] = []

    def child(self, label: str, node_type: str, page_number: int | None) -> "MutableTreeNode":
        for child in self.children:
            if child.label == label and child.node_type == node_type:
                return child
        child = MutableTreeNode(label=label, node_type=node_type, page_number=page_number, chunk_id=None)
        self.children.append(child)
        return child

    def freeze(self) -> ReviewTreeNode:
        stable_id = hashlib.sha1(f"{self.node_type}:{self.label}:{self.chunk_id}".encode("utf-8")).hexdigest()
        return ReviewTreeNode(
            id=stable_id,
            label=self.label,
            node_type=self.node_type,
            page_number=self.page_number,
            chunk_id=self.chunk_id,
            children=[child.freeze() for child in self.children],
        )


class QualityAnalyzer:
    required_metadata = ("chapter", "section", "page_number")

    def analyze(self, document_id: str, chunks: Sequence[ReviewChunk]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        seen_hashes: dict[str, str] = {}

        for chunk in chunks:
            metadata = chunk.metadata
            if chunk.token_count > 1000:
                issues.append(self._issue("WARNING", "CHUNK_TOO_LARGE", "Chunk token_count > 1000.", chunk))
            if chunk.token_count < 50:
                issues.append(self._issue("WARNING", "CHUNK_TOO_SHORT", "Chunk token_count < 50.", chunk))

            for key in self.required_metadata:
                if metadata.get(key) in {None, ""}:
                    issues.append(
                        self._issue("ERROR", "METADATA_MISSING", f"Missing metadata field: {key}.", chunk)
                    )

            page_numbers = metadata.get("page_numbers") or []
            if isinstance(page_numbers, list) and len(page_numbers) >= 2:
                pages = [int(page) for page in page_numbers if page]
                if pages and max(pages) - min(pages) > 5:
                    issues.append(self._issue("WARNING", "PAGE_JUMP", "Chunk spans distant pages.", chunk))

            heading = str(metadata.get("heading") or metadata.get("heading_path") or "")
            content_without_heading = chunk.text.replace(heading.replace(" > ", "\n"), "").strip()
            if heading and not content_without_heading:
                issues.append(self._issue("WARNING", "ORPHAN_HEADING", "Heading has no body content.", chunk))

            content_hash = str(metadata.get("content_hash") or "")
            if content_hash:
                if content_hash in seen_hashes:
                    issues.append(self._issue("WARNING", "DUPLICATE_CONTENT", "Duplicate chunk content hash.", chunk))
                seen_hashes[content_hash] = chunk.chunk_id

        return issues

    def _issue(
        self,
        severity: IssueSeverity,
        code: str,
        message: str,
        chunk: ReviewChunk,
    ) -> QualityIssue:
        metadata = chunk.metadata
        return QualityIssue(
            id=hashlib.sha1(f"{chunk.chunk_id}:{code}".encode("utf-8")).hexdigest(),
            severity=severity,
            code=code,
            message=message,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=metadata.get("page_number"),
        )


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_dict(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
