from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping
from uuid import uuid4

from services.md_ingestion_service import MarkdownIngestionService
from services.pdf_ingestion_service import PDFIngestionService
from services.word_ingestion_service import WordIngestionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "files" / "uploads"
UPLOAD_CHUNK_SIZE = 1024 * 1024
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class IngestionHandler:
    file_type: str
    factory: Callable[[], Any]


@dataclass(frozen=True)
class FileUploadResult:
    file_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    ingestion_status: str
    ingestion_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "file_id": self.file_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "status": self.status,
            "ingestion_status": self.ingestion_status,
            "ingestion": self.ingestion_summary,
        }


DEFAULT_HANDLERS: dict[str, IngestionHandler] = {
    ".pdf": IngestionHandler("pdf", PDFIngestionService),
    ".docx": IngestionHandler("word", WordIngestionService),
    ".md": IngestionHandler("markdown", MarkdownIngestionService),
    ".markdown": IngestionHandler("markdown", MarkdownIngestionService),
}


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded suffix has no registered ingestion handler."""


class FileUploadService:
    """Save uploaded files and dispatch them to the registered ingestion service."""

    def __init__(
        self,
        upload_dir: str | Path | None = None,
        handlers: Mapping[str, IngestionHandler] | None = None,
    ) -> None:
        self.upload_dir = Path(upload_dir or DEFAULT_UPLOAD_DIR).expanduser().resolve()
        self.handlers = dict(handlers or DEFAULT_HANDLERS)

    def register_handler(
        self,
        extension: str,
        file_type: str,
        factory: Callable[[], Any],
    ) -> None:
        normalized_extension = extension.lower()
        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"
        if not callable(factory):
            raise TypeError("factory must be callable.")
        self.handlers[normalized_extension] = IngestionHandler(file_type, factory)

    def upload_and_ingest(
        self,
        filename: str,
        source: BinaryIO,
    ) -> FileUploadResult:
        safe_filename = self._safe_filename(filename)
        extension = Path(safe_filename).suffix.lower()
        handler = self.handlers.get(extension)
        if handler is None:
            supported = ", ".join(sorted(self.handlers))
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {extension or '[none]'}. "
                f"Supported types: {supported}"
            )

        file_id = str(uuid4())
        document_dir = self.upload_dir / file_id
        document_dir.mkdir(parents=True, exist_ok=True)
        storage_path = document_dir / safe_filename
        file_size = self._save_stream(source, storage_path)
        if file_size == 0:
            raise ValueError("Uploaded file is empty.")

        ingestion_service = handler.factory()
        ingestion_result = ingestion_service.ingest(storage_path)

        return FileUploadResult(
            file_id=file_id,
            filename=safe_filename,
            file_type=handler.file_type,
            file_size=file_size,
            status="uploaded",
            ingestion_status="indexed",
            ingestion_summary=self._summarize_ingestion_result(ingestion_result),
        )

    def _safe_filename(self, filename: str) -> str:
        original_name = Path(filename or "").name
        safe_name = INVALID_FILENAME_CHARS.sub("_", original_name).strip()
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("A valid filename is required.")
        return safe_name

    def _save_stream(self, source: BinaryIO, destination: Path) -> int:
        total_size = 0
        with destination.open("wb") as target:
            while True:
                data = source.read(UPLOAD_CHUNK_SIZE)
                if not data:
                    break
                if not isinstance(data, bytes):
                    raise TypeError("Uploaded file stream must return bytes.")
                target.write(data)
                total_size += len(data)
        return total_size

    def _summarize_ingestion_result(self, result: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for field_name in (
            "collection_name",
            "chunks_count",
            "embedding_count",
            "inserted_count",
            "parsed_blocks_count",
            "cleaned_blocks_count",
            "parsed_elements_count",
            "cleaned_elements_count",
        ):
            if hasattr(result, field_name):
                summary[field_name] = getattr(result, field_name)
        return summary


__all__ = [
    "DEFAULT_HANDLERS",
    "DEFAULT_UPLOAD_DIR",
    "FileUploadResult",
    "FileUploadService",
    "IngestionHandler",
    "UnsupportedFileTypeError",
]
