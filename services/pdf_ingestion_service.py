from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
import subprocess
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import requests

from services.chunk_normalizer_service import ChunkNormalizerService, UnifiedChunk
from services.embedding_service import get_embedding_service
from services.milvus_service import MilvusConnectionService, get_milvus_service


PDF_FILE_TYPE = "pdf"
DEFAULT_COLLECTION = "developer_knowledge_chunks"
MINERU_PARSER_NAME = "mineru"


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class MinerUSettings:
    provider: str
    token: str
    precision_base_url: str
    extract_task_url: str
    extract_task_batch_url: str
    file_urls_batch_url: str
    task_result_url_template: str
    batch_result_url_template: str
    model_version: str
    language: str
    enable_table: bool
    enable_formula: bool
    is_ocr: bool
    page_ranges: str | None
    extra_formats: list[str]
    no_cache: bool
    cache_tolerance_seconds: int
    poll_interval_seconds: float
    timeout_seconds: float
    output_dir: Path

    @classmethod
    def from_env(cls) -> "MinerUSettings":
        _load_env_file()

        base_url = os.getenv("MINERU_PRECISION_BASE_URL", "https://mineru.net/api/v4").rstrip("/")
        return cls(
            provider=os.getenv("MINERU_PROVIDER", "precision"),
            token=os.getenv("MINERU_TOKEN", ""),
            precision_base_url=base_url,
            extract_task_url=os.getenv("MINERU_EXTRACT_TASK_URL") or f"{base_url}/extract/task",
            extract_task_batch_url=os.getenv("MINERU_EXTRACT_TASK_BATCH_URL")
            or f"{base_url}/extract/task/batch",
            file_urls_batch_url=os.getenv("MINERU_FILE_URLS_BATCH_URL")
            or f"{base_url}/file-urls/batch",
            task_result_url_template=os.getenv("MINERU_TASK_RESULT_URL_TEMPLATE")
            or f"{base_url}/extract/task/{{task_id}}",
            batch_result_url_template=os.getenv("MINERU_BATCH_RESULT_URL_TEMPLATE")
            or f"{base_url}/extract-results/batch/{{batch_id}}",
            model_version=os.getenv("MINERU_MODEL_VERSION", "vlm"),
            language=os.getenv("MINERU_LANGUAGE", "ch"),
            enable_table=_env_bool("MINERU_ENABLE_TABLE", True),
            enable_formula=_env_bool("MINERU_ENABLE_FORMULA", True),
            is_ocr=_env_bool("MINERU_IS_OCR", False),
            page_ranges=os.getenv("MINERU_PAGE_RANGES") or None,
            extra_formats=[
                value.strip()
                for value in os.getenv("MINERU_EXTRA_FORMATS", "docx,html").split(",")
                if value.strip()
            ],
            no_cache=_env_bool("MINERU_NO_CACHE", False),
            cache_tolerance_seconds=int(os.getenv("MINERU_CACHE_TOLERANCE_SECONDS", "900")),
            poll_interval_seconds=float(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "2")),
            timeout_seconds=float(os.getenv("MINERU_TIMEOUT_SECONDS", "300")),
            output_dir=Path(os.getenv("MINERU_OUTPUT_DIR", "files/mineru_outputs")),
        )


@dataclass(frozen=True)
class DocumentBlock:
    block_id: str
    block_type: str
    content: str
    page_number: int | None = None
    heading_level: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredDocument:
    file_id: str
    file_name: str
    file_type: str
    source_path: str
    parser: str
    language: str
    blocks: list[DocumentBlock]
    raw_result: dict[str, Any]


@dataclass(frozen=True)
class PdfChunk:
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    content_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PdfIngestionResult:
    file_id: str
    file_name: str
    file_type: str
    source_path: str
    collection_name: str
    parsed_blocks_count: int
    cleaned_blocks_count: int
    chunks_count: int
    embedding_count: int
    inserted_count: int
    mysql_persisted: bool
    chunk_ids: list[str]


class DocumentRecordRepository(Protocol):
    def save_document_record(self, record: dict[str, Any]) -> None:
        ...


class NoopDocumentRecordRepository:
    """Placeholder until a MySQL repository service exists."""

    def save_document_record(self, record: dict[str, Any]) -> None:
        return None


class MinerUClient:
    """Precision MinerU client. Local files use signed upload URLs."""

    def __init__(self, settings: MinerUSettings | None = None) -> None:
        self.settings = settings or MinerUSettings.from_env()
        if self.settings.provider != "precision":
            raise RuntimeError("PDF ingestion requires MINERU_PROVIDER=precision.")
        if not self.settings.token:
            raise RuntimeError("MINERU_TOKEN must be configured for MinerU precision API.")

    def parse_file(self, file_path: str | Path) -> dict[str, Any]:
        path = self._validate_pdf(file_path)
        data_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{self._file_hash(path)}"))
        batch_id = self._request_upload_url(path.name, data_id)
        self._upload_file(path, batch_id.file_url)
        return self._poll_batch_result(batch_id.batch_id, data_id)

    def parse_url(self, file_url: str, file_name: str | None = None) -> dict[str, Any]:
        task_id = self._create_url_task(file_url, file_name=file_name)
        return self._poll_task_result(task_id)

    def _validate_pdf(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only .pdf files are supported.")
        return path

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.token}",
        }

    def _base_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_version": self.settings.model_version,
            "enable_table": self.settings.enable_table,
            "enable_formula": self.settings.enable_formula,
            "language": self.settings.language,
            "is_ocr": self.settings.is_ocr,
            "no_cache": self.settings.no_cache,
            "cache_tolerance": self.settings.cache_tolerance_seconds,
        }
        if self.settings.page_ranges:
            payload["page_ranges"] = self.settings.page_ranges
        if self.settings.extra_formats:
            payload["extra_formats"] = self.settings.extra_formats
        return payload

    @dataclass(frozen=True)
    class UploadPlan:
        batch_id: str
        file_url: str

    def _request_upload_url(self, file_name: str, data_id: str) -> "MinerUClient.UploadPlan":
        payload = self._base_payload()
        payload["files"] = [{"name": file_name, "data_id": data_id}]
        response = requests.post(
            self.settings.file_urls_batch_url,
            headers=self._headers(),
            json=payload,
            timeout=self.settings.timeout_seconds,
        )
        data = self._json_response(response)
        file_urls = data.get("data", {}).get("file_urls") or []
        batch_id = data.get("data", {}).get("batch_id")
        if not batch_id or not file_urls:
            raise RuntimeError(f"MinerU did not return upload URL: {data}")
        return self.UploadPlan(batch_id=batch_id, file_url=file_urls[0])

    def _upload_file(self, path: Path, file_url: str) -> None:
        with path.open("rb") as file:
            response = requests.put(
                file_url,
                data=file,
                timeout=self.settings.timeout_seconds,
            )
        if response.status_code not in {200, 201, 204}:
            raise RuntimeError(
                f"MinerU file upload failed: {response.status_code} {response.text[:300]}"
            )

    def _create_url_task(self, file_url: str, file_name: str | None = None) -> str:
        payload = self._base_payload()
        payload["url"] = file_url
        if file_name:
            payload["file_name"] = file_name
        response = requests.post(
            self.settings.extract_task_url,
            headers=self._headers(),
            json=payload,
            timeout=self.settings.timeout_seconds,
        )
        data = self._json_response(response)
        task_id = data.get("data", {}).get("task_id") or data.get("data", {}).get("id")
        if not task_id:
            raise RuntimeError(f"MinerU did not return task_id: {data}")
        return str(task_id)

    def _poll_task_result(self, task_id: str) -> dict[str, Any]:
        url = self.settings.task_result_url_template.format(task_id=task_id)
        deadline = time.time() + self.settings.timeout_seconds
        last_data: dict[str, Any] | None = None

        while time.time() < deadline:
            response = requests.get(url, headers=self._headers(), timeout=self.settings.timeout_seconds)
            data = self._json_response(response)
            last_data = data
            result = data.get("data") or {}
            if self._is_done(result):
                return result
            if self._is_failed(result):
                raise RuntimeError(f"MinerU task failed: {data}")
            time.sleep(self.settings.poll_interval_seconds)

        raise TimeoutError(f"MinerU task timed out. Last response: {last_data}")

    def _poll_batch_result(self, batch_id: str, data_id: str) -> dict[str, Any]:
        url = self.settings.batch_result_url_template.format(batch_id=batch_id)
        deadline = time.time() + self.settings.timeout_seconds
        last_data: dict[str, Any] | None = None

        while time.time() < deadline:
            response = requests.get(url, headers=self._headers(), timeout=self.settings.timeout_seconds)
            data = self._json_response(response)
            last_data = data
            result = self._extract_batch_item(data, data_id)
            if result and self._is_done(result):
                return result
            if result and self._is_failed(result):
                raise RuntimeError(f"MinerU batch task failed: {data}")
            time.sleep(self.settings.poll_interval_seconds)

        raise TimeoutError(f"MinerU batch task timed out. Last response: {last_data}")

    def _extract_batch_item(self, response_data: dict[str, Any], data_id: str) -> dict[str, Any] | None:
        data = response_data.get("data") or {}
        candidates = (
            data.get("extract_result")
            or data.get("extract_results")
            or data.get("results")
            or data.get("files")
            or []
        )
        if isinstance(candidates, dict):
            candidates = [candidates]
        for item in candidates:
            if str(item.get("data_id", "")) == data_id or item.get("data_id") is None:
                return item
        return None

    def _is_done(self, result: dict[str, Any]) -> bool:
        status = str(result.get("state") or result.get("status") or "").lower()
        has_artifact = any(
            result.get(key)
            for key in (
                "full_zip_url",
                "md_url",
                "markdown_url",
                "content_list_url",
                "middle_json_url",
                "json_url",
            )
        )
        return status in {"done", "success", "completed", "finish", "finished"} or has_artifact

    def _is_failed(self, result: dict[str, Any]) -> bool:
        status = str(result.get("state") or result.get("status") or "").lower()
        return status in {"failed", "error", "fail"}

    def _json_response(self, response: requests.Response) -> dict[str, Any]:
        response.raise_for_status()
        data = response.json()
        if data.get("code", 0) not in {0, "0"}:
            raise RuntimeError(f"MinerU API error: {data}")
        return data

    def _file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()


class MinerUResultNormalizer:
    """Convert MinerU artifacts into a common StructuredDocument."""

    def __init__(self, settings: MinerUSettings | None = None) -> None:
        self.settings = settings or MinerUSettings.from_env()

    def normalize(self, mineru_result: dict[str, Any], source_path: str | Path) -> StructuredDocument:
        path = Path(source_path).expanduser().resolve()
        artifact = self._load_artifact(mineru_result)
        blocks = self._blocks_from_artifact(artifact)
        return StructuredDocument(
            file_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{self._file_hash(path)}")),
            file_name=path.name,
            file_type=PDF_FILE_TYPE,
            source_path=str(path),
            parser=MINERU_PARSER_NAME,
            language=self.settings.language,
            blocks=blocks,
            raw_result=mineru_result,
        )

    def _load_artifact(self, result: dict[str, Any]) -> dict[str, Any]:
        artifact: dict[str, Any] = {"raw": result}
        json_payload = self._first_json_artifact(result)
        markdown_payload = self._first_markdown_artifact(result)

        if json_payload is not None:
            artifact["json"] = json_payload
        if markdown_payload:
            artifact["markdown"] = markdown_payload

        full_zip_url = result.get("full_zip_url") or result.get("zip_url")
        if full_zip_url and ("json" not in artifact or "markdown" not in artifact):
            zip_artifact = self._download_zip_artifacts(full_zip_url)
            artifact.update({key: value for key, value in zip_artifact.items() if key not in artifact})

        return artifact

    def _first_json_artifact(self, result: dict[str, Any]) -> Any | None:
        for key in ("content_list", "middle_json", "layout_dets", "json"):
            if isinstance(result.get(key), (list, dict)):
                return result[key]
        for key in ("content_list_url", "middle_json_url", "json_url"):
            if result.get(key):
                return self._fetch_json(result[key])
        return None

    def _first_markdown_artifact(self, result: dict[str, Any]) -> str | None:
        for key in ("md_content", "markdown", "md"):
            if isinstance(result.get(key), str) and result[key].strip():
                return result[key]
        for key in ("md_url", "markdown_url"):
            if result.get(key):
                return self._fetch_text(result[key])
        return None

    def _download_zip_artifacts(self, url: str) -> dict[str, Any]:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.settings.output_dir / f"mineru_{uuid.uuid4().hex}.zip"
        self._download_file(url, zip_path)

        artifact: dict[str, Any] = {"zip_path": str(zip_path)}
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                lower_name = name.lower()
                if lower_name.endswith("content_list.json") and "json" not in artifact:
                    artifact["json"] = json.loads(archive.read(name).decode("utf-8"))
                elif lower_name.endswith(".json") and "json" not in artifact:
                    artifact["json"] = json.loads(archive.read(name).decode("utf-8"))
                elif lower_name.endswith(".md") and "markdown" not in artifact:
                    artifact["markdown"] = archive.read(name).decode("utf-8")
        return artifact

    def _download_file(self, url: str, target_path: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                with requests.get(url, timeout=self.settings.timeout_seconds, stream=True) as response:
                    response.raise_for_status()
                    with target_path.open("wb") as file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                return
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(self.settings.poll_interval_seconds * attempt)
        if self._download_file_with_curl(url, target_path):
            return
        raise RuntimeError(f"Failed to download MinerU artifact after retries: {last_error}") from last_error

    def _download_file_with_curl(self, url: str, target_path: Path) -> bool:
        curl_path = self._find_curl()
        if curl_path is None:
            return False

        result = subprocess.run(
            [
                curl_path,
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--output",
                str(target_path),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=self.settings.timeout_seconds,
            check=False,
        )
        return result.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0

    def _find_curl(self) -> str | None:
        for candidate in ("curl.exe", "curl"):
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except OSError:
                continue
            if result.returncode == 0:
                return candidate
        return None

    def _blocks_from_artifact(self, artifact: dict[str, Any]) -> list[DocumentBlock]:
        if "json" in artifact:
            blocks = self._blocks_from_json(artifact["json"])
            if blocks:
                return blocks
        if artifact.get("markdown"):
            return MarkdownBlockParser().parse(artifact["markdown"])
        raise RuntimeError("MinerU result does not contain structured JSON or Markdown content.")

    def _blocks_from_json(self, payload: Any) -> list[DocumentBlock]:
        items = payload
        if isinstance(payload, dict):
            items = (
                payload.get("content_list")
                or payload.get("blocks")
                or payload.get("layout_dets")
                or payload.get("pages")
                or []
            )
        if isinstance(items, dict):
            items = [items]

        blocks: list[DocumentBlock] = []
        for item in self._flatten_items(items):
            block = self._block_from_item(item, len(blocks))
            if block is not None:
                blocks.append(block)
        return blocks

    def _flatten_items(self, items: Any) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        if not isinstance(items, list):
            return flattened
        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("blocks"), list):
                flattened.extend(self._flatten_items(item["blocks"]))
            else:
                flattened.append(item)
        return flattened

    def _block_from_item(self, item: dict[str, Any], index: int) -> DocumentBlock | None:
        raw_type = str(item.get("type") or item.get("category") or item.get("block_type") or "text")
        content = self._item_content(item)
        if not content:
            return None

        block_type = self._normalize_type(raw_type, item, content)
        page_number = self._page_number(item)
        heading_level = self._heading_level(item, block_type, content)
        return DocumentBlock(
            block_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"pdf-block:{index}:{content[:80]}")),
            block_type=block_type,
            content=content,
            page_number=page_number,
            heading_level=heading_level,
            metadata={
                "raw_type": raw_type,
                "bbox": item.get("bbox") or item.get("poly"),
                "image_path": item.get("img_path") or item.get("image_path"),
            },
        )

    def _item_content(self, item: dict[str, Any]) -> str:
        raw_type = str(item.get("type") or item.get("category") or "").lower()
        if raw_type == "table" or item.get("table_body") or item.get("table"):
            return TableProcessor().from_mineru_item(item)
        if item.get("text"):
            return str(item["text"]).strip()
        if item.get("content"):
            return str(item["content"]).strip()
        if item.get("html"):
            return TableProcessor().to_embedding_text(str(item["html"]))
        if item.get("caption"):
            return str(item["caption"]).strip()
        if item.get("img_caption"):
            return self._caption_text(item["img_caption"])
        if item.get("latex"):
            return str(item["latex"]).strip()
        return ""

    def _normalize_type(self, raw_type: str, item: dict[str, Any], content: str) -> str:
        lower_type = raw_type.lower()
        if lower_type in {"title"}:
            return "title"
        if item.get("text_level") or lower_type in {"heading", "header"}:
            return "heading"
        if lower_type in {"table"} or content.startswith("|"):
            return "table"
        if lower_type in {"image", "figure"}:
            return "image"
        if lower_type in {"caption", "figure_caption", "table_caption"}:
            return "caption"
        if lower_type in {"equation", "formula"}:
            return "formula"
        if lower_type in {"code", "code_block"}:
            return "code"
        if re.match(r"^\s*([-*+]|\d+[.)、])\s+", content):
            return "list"
        return "paragraph"

    def _page_number(self, item: dict[str, Any]) -> int | None:
        value = item.get("page_number") or item.get("page") or item.get("page_idx")
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number + 1 if item.get("page_idx") is not None else number

    def _heading_level(self, item: dict[str, Any], block_type: str, content: str) -> int | None:
        if block_type != "heading":
            return None
        match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]+", content)
        if match:
            number_depth = len(match.group(1).split("."))
            return min(number_depth + 1, 6)
        if item.get("text_level"):
            return int(item["text_level"])
        return 1

    def _caption_text(self, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def _fetch_json(self, url: str) -> Any:
        response = requests.get(url, timeout=self.settings.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _fetch_text(self, url: str) -> str:
        response = requests.get(url, timeout=self.settings.timeout_seconds)
        response.raise_for_status()
        return response.text

    def _file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()


class MarkdownBlockParser:
    """Fallback parser for MinerU markdown artifacts."""

    def parse(self, markdown: str) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        buffer: list[str] = []
        in_code = False

        for line in markdown.splitlines():
            if line.startswith("```"):
                if not in_code:
                    self._flush_text(blocks, buffer)
                    buffer = [line]
                    in_code = True
                else:
                    buffer.append(line)
                    self._append_block(blocks, "code", "\n".join(buffer), None)
                    buffer = []
                    in_code = False
                continue

            if in_code:
                buffer.append(line)
                continue

            if line.startswith("#"):
                self._flush_text(blocks, buffer)
                buffer = []
                heading_level = min(len(line) - len(line.lstrip("#")), 6)
                self._append_block(blocks, "heading", line.lstrip("#").strip(), heading_level)
            elif line.strip().startswith("|"):
                self._flush_text(blocks, buffer)
                buffer = []
                self._append_block(blocks, "table", line.strip(), None)
            else:
                buffer.append(line)

        self._flush_text(blocks, buffer)
        return blocks

    def _flush_text(self, blocks: list[DocumentBlock], buffer: list[str]) -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        block_type = "list" if re.match(r"^\s*([-*+]|\d+[.)、])\s+", text) else "paragraph"
        self._append_block(blocks, block_type, text, None)

    def _append_block(
        self,
        blocks: list[DocumentBlock],
        block_type: str,
        content: str,
        heading_level: int | None,
    ) -> None:
        blocks.append(
            DocumentBlock(
                block_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"md-block:{len(blocks)}:{content[:80]}")),
                block_type=block_type,
                content=content,
                heading_level=heading_level,
            )
        )


class DocumentCleaner:
    """Remove repeated noise while preserving the document tree."""

    page_number_pattern = re.compile(r"^\s*(第\s*)?\d+\s*(页|/\s*\d+)?\s*$")

    def clean(self, document: StructuredDocument) -> StructuredDocument:
        repeated_noise = self._repeated_short_text(document.blocks)
        cleaned_blocks: list[DocumentBlock] = []
        previous_content = ""

        for block in document.blocks:
            content = self._clean_content(block)
            if not content:
                continue
            if self._is_page_number(content):
                continue
            if block.block_type in {"paragraph", "caption"} and content in repeated_noise:
                continue
            if content == previous_content:
                continue

            cleaned_blocks.append(
                DocumentBlock(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    content=content,
                    page_number=block.page_number,
                    heading_level=block.heading_level,
                    metadata=block.metadata,
                )
            )
            previous_content = content

        return StructuredDocument(
            file_id=document.file_id,
            file_name=document.file_name,
            file_type=document.file_type,
            source_path=document.source_path,
            parser=document.parser,
            language=document.language,
            blocks=cleaned_blocks,
            raw_result=document.raw_result,
        )

    def _clean_content(self, block: DocumentBlock) -> str:
        content = block.content.replace("\u00a0", " ").replace("\u200b", "")
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.strip()
        if block.block_type in {"paragraph", "list", "caption"}:
            content = self._merge_broken_lines(content)
        return content

    def _merge_broken_lines(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        merged = [lines[0]]
        for line in lines[1:]:
            previous = merged[-1]
            if self._should_join(previous, line):
                merged[-1] = previous + line
            else:
                merged.append(line)
        return "\n".join(merged)

    def _should_join(self, previous: str, current: str) -> bool:
        if previous.endswith(("。", "！", "？", ".", ":", "：", ";", "；", ")", "）", "`")):
            return False
        if re.match(r"^([-*+]|\d+[.)、])\s+", current):
            return False
        return True

    def _is_page_number(self, text: str) -> bool:
        return bool(self.page_number_pattern.match(text))

    def _repeated_short_text(self, blocks: Sequence[DocumentBlock]) -> set[str]:
        counts: dict[str, int] = {}
        for block in blocks:
            content = block.content.strip()
            if block.block_type in {"paragraph", "caption"} and 0 < len(content) <= 120:
                counts[content] = counts.get(content, 0) + 1
        return {content for content, count in counts.items() if count >= 3}


class TokenEstimator:
    token_pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")

    def count(self, text: str) -> int:
        return len(self.token_pattern.findall(text))

    def split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？!?；;.\n])\s*", text)
        return [part.strip() for part in parts if part.strip()]


class TableProcessor:
    """Create embedding-friendly table text."""

    caption_keys = ("table_caption", "caption", "table_title", "title")
    footnote_keys = ("table_footnote", "footnote", "note")

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
        body_text = self._table_to_text(raw_body)
        leading_captions, body_text = self._split_leading_captions(body_text)
        captions = self._dedupe_lines([*captions, *leading_captions])

        parts: list[str] = []
        if captions:
            parts.extend(f"Table caption: {caption}" for caption in captions)
        if body_text:
            parts.append(body_text)
        if footnotes:
            parts.extend(f"Table note: {note}" for note in self._dedupe_lines(footnotes))
        return "\n\n".join(parts).strip()

    def to_embedding_text(self, table: Any) -> str:
        if isinstance(table, str):
            if table.strip().startswith("|"):
                return table.strip()
            return self._html_to_text(table)
        if isinstance(table, list):
            return self._rows_to_text(table)
        return str(table).strip()

    def _table_to_text(self, table: Any) -> str:
        if isinstance(table, str):
            if table.strip().startswith("|"):
                return table.strip()
            return self._html_to_text_preserving_rows(table)
        if isinstance(table, list):
            return self._rows_to_text(table)
        return str(table).strip()

    def _html_to_text_preserving_rows(self, html: str) -> str:
        text = html_lib.unescape(html)
        text = re.sub(
            r"<\s*/\s*t[dh]\s*>\s*<\s*t[dh][^>]*>",
            " | ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"<\s*/\s*tr\s*>\s*<\s*tr[^>]*>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
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

    def _rows_to_text(self, rows: Sequence[Any]) -> str:
        normalized_rows = [
            [str(cell).strip() for cell in row if str(cell).strip()]
            for row in rows
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        ]
        if not normalized_rows:
            return ""
        header = normalized_rows[0]
        sentences: list[str] = []
        for row_index, row in enumerate(normalized_rows[1:], start=1):
            if len(row) == len(header):
                pairs = [f"{name}为{value.rstrip('。.;；')}" for name, value in zip(header, row)]
                sentences.append(f"表格第{row_index}行记录: " + "，".join(pairs) + "。")
            else:
                sentences.append(f"表格第{row_index}行记录: " + "，".join(row) + "。")
        return "\n".join(sentences)

    def _html_to_text(self, html: str) -> str:
        text = re.sub(r"<\s*/\s*tr\s*>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<\s*/\s*(td|th)\s*>", " | ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class SemanticChunkBuilder:
    """Build chunks from document structure rather than raw strings."""

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

    def build(self, document: StructuredDocument) -> list[PdfChunk]:
        chunks: list[PdfChunk] = []
        heading_stack: list[str] = []
        section_blocks: list[DocumentBlock] = []

        for block in document.blocks:
            if block.block_type in {"title", "heading"}:
                for heading_block in self._expand_heading_block(block):
                    self._flush_section(chunks, document, heading_stack, section_blocks)
                    section_blocks = []
                    level = self._effective_heading_level(heading_block)
                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(heading_block.content)
            else:
                section_blocks.append(block)

        self._flush_section(chunks, document, heading_stack, section_blocks)
        return chunks

    def _expand_heading_block(self, block: DocumentBlock) -> list[DocumentBlock]:
        content = block.content.strip()
        if not content:
            return []

        matches = list(self._heading_marker_pattern().finditer(content))
        if len(matches) <= 1:
            return [block]

        expanded: list[DocumentBlock] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            heading_text = content[start:end].strip()
            if not heading_text:
                continue
            expanded.append(
                DocumentBlock(
                    block_id=f"{block.block_id}:heading:{index}",
                    block_type=block.block_type,
                    content=heading_text,
                    page_number=block.page_number,
                    heading_level=self._level_from_numbered_heading(heading_text)
                    or block.heading_level,
                    metadata=block.metadata,
                )
            )
        return expanded or [block]

    def _effective_heading_level(self, block: DocumentBlock) -> int:
        return max(self._level_from_numbered_heading(block.content) or block.heading_level or 1, 1)

    def _level_from_numbered_heading(self, content: str) -> int | None:
        match = self._heading_marker_pattern().match(content.strip())
        if not match:
            return None

        marker = match.group("marker").lower()
        marker = marker.rstrip(".;；、")
        if marker.startswith("a"):
            marker = marker[1:]
        depth = len([part for part in marker.split(".") if part])
        return min(depth + 1, 6)

    def _heading_marker_pattern(self) -> re.Pattern[str]:
        return re.compile(
            r"(?<!\S)(?P<marker>[Aa]?\d+(?:\.\d+)*)(?:\.)?\s*[;；、.]?\s*"
        )

    def _flush_section(
        self,
        chunks: list[PdfChunk],
        document: StructuredDocument,
        heading_stack: Sequence[str],
        blocks: Sequence[DocumentBlock],
    ) -> None:
        if not blocks:
            return

        prefix = "\n".join(heading_stack)
        current_blocks: list[DocumentBlock] = []
        current_tokens = self.token_estimator.count(prefix)

        for block in blocks:
            units = self._block_units(block)
            for unit in units:
                unit_tokens = self.token_estimator.count(unit.content)
                if current_blocks and current_tokens + unit_tokens > self.max_tokens:
                    self._append_chunk(chunks, document, heading_stack, current_blocks)
                    current_blocks = self._overlap_blocks(current_blocks)
                    current_tokens = self.token_estimator.count(
                        prefix + "\n" + "\n".join(block.content for block in current_blocks)
                    )

                if unit_tokens > self.max_tokens and unit.block_type == "paragraph":
                    self._append_large_paragraph(chunks, document, heading_stack, unit)
                    current_blocks = []
                    current_tokens = self.token_estimator.count(prefix)
                    continue

                current_blocks.append(unit)
                current_tokens += unit_tokens

        if current_blocks:
            self._append_chunk(chunks, document, heading_stack, current_blocks)

    def _block_units(self, block: DocumentBlock) -> list[DocumentBlock]:
        if block.block_type != "paragraph":
            return [block]
        if self.token_estimator.count(block.content) <= self.target_tokens:
            return [block]
        return [
            DocumentBlock(
                block_id=f"{block.block_id}:{index}",
                block_type=block.block_type,
                content=sentence,
                page_number=block.page_number,
                heading_level=block.heading_level,
                metadata=block.metadata,
            )
            for index, sentence in enumerate(self.token_estimator.split_sentences(block.content))
        ]

    def _append_large_paragraph(
        self,
        chunks: list[PdfChunk],
        document: StructuredDocument,
        heading_stack: Sequence[str],
        block: DocumentBlock,
    ) -> None:
        for unit in self._block_units(block):
            self._append_chunk(chunks, document, heading_stack, [unit])

    def _append_chunk(
        self,
        chunks: list[PdfChunk],
        document: StructuredDocument,
        heading_stack: Sequence[str],
        blocks: Sequence[DocumentBlock],
    ) -> None:
        body = "\n".join(block.content.strip() for block in blocks if block.content.strip()).strip()
        if not body:
            return

        prefix = "\n".join(heading_stack)
        text = f"{prefix}\n\n{body}".strip() if prefix else body
        chunk_index = len(chunks)
        chapter = heading_stack[0] if heading_stack else ""
        section = heading_stack[-1] if len(heading_stack) > 1 else ""
        page_numbers = sorted({block.page_number for block in blocks if block.page_number})
        block_types = sorted({block.block_type for block in blocks})
        content_hash = self._hash_text(text)
        metadata = {
            "file_id": document.file_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "page_number": page_numbers[0] if page_numbers else None,
            "page_numbers": page_numbers,
            "chapter": chapter,
            "section": section,
            "heading": " > ".join(heading_stack),
            "chunk_index": chunk_index,
            "block_type": ",".join(block_types),
            "parser": document.parser,
            "language": document.language,
            "create_time": datetime.now(timezone.utc).isoformat(),
            "source_path": document.source_path,
        }
        chunks.append(
            PdfChunk(
                chunk_id=self._chunk_id(document.file_id, chunk_index, content_hash),
                chunk_index=chunk_index,
                text=text,
                token_count=self.token_estimator.count(text),
                content_hash=content_hash,
                metadata=metadata,
            )
        )

    def _overlap_blocks(self, blocks: Sequence[DocumentBlock]) -> list[DocumentBlock]:
        overlap: list[DocumentBlock] = []
        token_count = 0
        for block in reversed(blocks):
            if block.block_type in {"table", "image", "formula", "code"}:
                continue
            block_tokens = self.token_estimator.count(block.content)
            if overlap and token_count + block_tokens > self.overlap_tokens:
                break
            overlap.insert(0, block)
            token_count += block_tokens
        return overlap

    def _chunk_id(self, file_id: str, chunk_index: int, content_hash: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}:{chunk_index}:{content_hash}"))

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingGenerator:
    def __init__(self, embedding_service: Any | None = None) -> None:
        self.embedding_service = embedding_service

    def generate(self, chunks: Sequence[UnifiedChunk]) -> list[list[float]]:
        service = self.embedding_service or get_embedding_service()
        vectors = service.embed_documents([chunk.text for chunk in chunks])
        normalized = [list(vector) for vector in vectors]
        if len(normalized) != len(chunks):
            raise RuntimeError("Embedding result count does not match chunk count.")
        return normalized


class MilvusChunkWriter:
    def __init__(
        self,
        collection_name: str | None = None,
        milvus_service: MilvusConnectionService | None = None,
        auto_create_collection: bool = True,
    ) -> None:
        self.collection_name = (
            collection_name
            or os.getenv("MILVUS_COLLECTION_NAME")
            or DEFAULT_COLLECTION
        )
        self.milvus_service = milvus_service or get_milvus_service()
        self.auto_create_collection = auto_create_collection

    def save(self, chunks: Sequence[UnifiedChunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")

        self._ensure_collection(len(vectors[0]))
        rows = [self._row(chunk, vector) for chunk, vector in zip(chunks, vectors)]
        result = self.milvus_service.client.upsert(collection_name=self.collection_name, data=rows)
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
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _row(self, chunk: UnifiedChunk, vector: Sequence[float]) -> dict[str, Any]:
        metadata = chunk.metadata.to_dict()
        document = metadata["document"]
        structure = metadata["structure"]
        source = metadata["source"]
        content = metadata["content"]
        processing = metadata["processing"]
        page_numbers = source.get("page_numbers") or []
        content_types = content.get("types") or []
        return {
            "id": chunk.chunk_id,
            "vector": list(vector),
            "text": chunk.text,
            "file_name": document.get("file_name", ""),
            "file_type": document.get("file_type", PDF_FILE_TYPE),
            "chapter": structure.get("chapter", ""),
            "section": structure.get("section", ""),
            "heading": structure.get("heading", ""),
            "heading_path": " > ".join(str(value) for value in structure.get("heading_path", [])),
            "chunk_index": chunk.chunk_index,
            "page_number": page_numbers[0] if page_numbers else None,
            "page_numbers": ",".join(str(value) for value in page_numbers),
            "block_type": ",".join(str(value) for value in content_types),
            "parser": processing.get("parser", MINERU_PARSER_NAME),
            "language": processing.get("language", ""),
            "create_time": processing.get("create_time", ""),
            "source_path": source.get("source_path", ""),
            "token_count": chunk.token_count,
            "content_hash": chunk.content_hash,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }


class PDFIngestionService:
    """Enterprise PDF document understanding pipeline."""

    def __init__(
        self,
        parser: MinerUClient | None = None,
        normalizer: MinerUResultNormalizer | None = None,
        cleaner: DocumentCleaner | None = None,
        chunk_builder: SemanticChunkBuilder | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_writer: MilvusChunkWriter | None = None,
        document_repository: DocumentRecordRepository | None = None,
        chunk_normalizer: ChunkNormalizerService | None = None,
    ) -> None:
        self.parser = parser or MinerUClient()
        self.normalizer = normalizer or MinerUResultNormalizer()
        self.cleaner = cleaner or DocumentCleaner()
        self.chunk_builder = chunk_builder or SemanticChunkBuilder()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_writer = vector_writer or MilvusChunkWriter()
        self.document_repository = document_repository or NoopDocumentRecordRepository()
        self.chunk_normalizer = chunk_normalizer or ChunkNormalizerService()

    def ingest(self, file_path: str | Path) -> PdfIngestionResult:
        started_at = datetime.now(timezone.utc)
        structured_document = self.parse_document(file_path)
        cleaned_document = self.clean_document(structured_document)
        raw_chunks = self.build_chunks(cleaned_document)
        chunks = self.normalize_chunks(raw_chunks)
        vectors = self.generate_embeddings(chunks)
        inserted_count = self.save_to_milvus(chunks, vectors)
        mysql_persisted = self.save_document_record(
            cleaned_document,
            chunks,
            vectors,
            inserted_count,
            started_at,
        )

        return PdfIngestionResult(
            file_id=cleaned_document.file_id,
            file_name=cleaned_document.file_name,
            file_type=cleaned_document.file_type,
            source_path=cleaned_document.source_path,
            collection_name=self.vector_writer.collection_name,
            parsed_blocks_count=len(structured_document.blocks),
            cleaned_blocks_count=len(cleaned_document.blocks),
            chunks_count=len(chunks),
            embedding_count=len(vectors),
            inserted_count=inserted_count,
            mysql_persisted=mysql_persisted,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )

    def parse_document(self, file_path: str | Path) -> StructuredDocument:
        mineru_result = self.parser.parse_file(file_path)
        return self.normalizer.normalize(mineru_result, file_path)

    def clean_document(self, document: StructuredDocument) -> StructuredDocument:
        return self.cleaner.clean(document)

    def build_chunks(self, document: StructuredDocument) -> list[PdfChunk]:
        chunks = self.chunk_builder.build(document)
        if not chunks:
            raise RuntimeError(f"No chunks were generated for PDF: {document.source_path}")
        return chunks

    def normalize_chunks(self, chunks: Sequence[PdfChunk]) -> list[UnifiedChunk]:
        return self.chunk_normalizer.normalize_many(chunks, file_type=PDF_FILE_TYPE)

    def generate_embeddings(self, chunks: Sequence[UnifiedChunk]) -> list[list[float]]:
        return self.embedding_generator.generate(chunks)

    def save_to_milvus(self, chunks: Sequence[UnifiedChunk], vectors: Sequence[Sequence[float]]) -> int:
        return self.vector_writer.save(chunks, vectors)

    def save_document_record(
        self,
        document: StructuredDocument,
        chunks: Sequence[UnifiedChunk],
        vectors: Sequence[Sequence[float]],
        inserted_count: int,
        started_at: datetime,
    ) -> bool:
        record = {
            "file_id": document.file_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "source_path": document.source_path,
            "parser": document.parser,
            "language": document.language,
            "document_status": "indexed",
            "parse_status": "success",
            "chunk_count": len(chunks),
            "embedding_count": len(vectors),
            "milvus_collection": self.vector_writer.collection_name,
            "inserted_count": inserted_count,
            "parse_time_seconds": round((datetime.now(timezone.utc) - started_at).total_seconds(), 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.document_repository.save_document_record(record)
        return not isinstance(self.document_repository, NoopDocumentRecordRepository)


PdfIngestionService = PDFIngestionService
