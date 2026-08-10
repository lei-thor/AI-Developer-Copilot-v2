from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.embedding_service import EmbeddingService, get_embedding_service
from services.milvus_service import MilvusConnectionService, get_milvus_service


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_COLLECTION_NAME = "developer_knowledge_chunks"
DEFAULT_OUTPUT_FIELDS = [
    "text",
    "file_name",
    "file_type",
    "chapter",
    "section",
    "$meta",
]


def _load_env_file(env_path: Path = ENV_PATH) -> None:
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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer env value for %s: %s", name, value)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env value for %s: %s", name, value)
        return default


class RetrievalError(RuntimeError):
    """Base error for retrieval failures."""


class EmbeddingUnavailableError(RetrievalError):
    """Raised when query embedding generation fails."""


class VectorSearchUnavailableError(RetrievalError):
    """Raised when Milvus vector search fails."""


@dataclass(frozen=True)
class RetrievalSettings:
    collection_name: str
    default_top_k: int = 5
    max_top_k: int = 20
    min_score: float = 0.0

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        _load_env_file()
        default_top_k = max(_env_int("RETRIEVAL_DEFAULT_TOP_K", 5), 1)
        max_top_k = max(_env_int("RETRIEVAL_MAX_TOP_K", 20), 1)

        return cls(
            collection_name=(
                os.getenv("RETRIEVAL_COLLECTION_NAME")
                or os.getenv("MILVUS_COLLECTION_NAME")
                or DEFAULT_COLLECTION_NAME
            ),
            default_top_k=default_top_k,
            max_top_k=max(max_top_k, default_top_k),
            min_score=max(_env_float("RETRIEVAL_MIN_SCORE", 0.0), 0.0),
        )


class MilvusVectorSearchService:
    """Thin Milvus data-access wrapper used by RetrievalService."""

    def __init__(
        self,
        milvus_service: MilvusConnectionService | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.milvus_service = milvus_service or get_milvus_service()
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        output_fields: Sequence[str] | None = None,
    ) -> list[Any]:
        fields = list(output_fields or DEFAULT_OUTPUT_FIELDS)
        try:
            return self._search(vector=vector, top_k=top_k, output_fields=fields)
        except Exception as exc:
            if "$meta" in fields:
                fallback_fields = [field for field in fields if field != "$meta"]
                logger.debug("Retry Milvus search without $meta output field.", exc_info=exc)
                try:
                    return self._search(
                        vector=vector,
                        top_k=top_k,
                        output_fields=fallback_fields,
                    )
                except Exception:
                    pass
            raise VectorSearchUnavailableError("Milvus vector search failed.") from exc

    def _search(
        self,
        vector: Sequence[float],
        top_k: int,
        output_fields: Sequence[str],
    ) -> list[Any]:
        if not vector:
            raise ValueError("Search vector cannot be empty.")

        client = self.milvus_service.client
        if not client.has_collection(self.collection_name):
            raise VectorSearchUnavailableError(
                f"Milvus collection does not exist: {self.collection_name}"
            )

        self._load_collection()
        hits = client.search(
            collection_name=self.collection_name,
            data=[list(vector)],
            anns_field="vector",
            limit=top_k,
            output_fields=list(output_fields),
            search_params={"metric_type": "COSINE", "params": {}},
        )
        if not hits:
            return []
        return list(hits[0] or [])

    def _load_collection(self) -> None:
        load_collection = getattr(self.milvus_service.client, "load_collection", None)
        if not callable(load_collection):
            return
        try:
            load_collection(collection_name=self.collection_name)
        except Exception:
            logger.debug("Milvus collection load skipped or already loaded.", exc_info=True)


class RetrievalService:
    """RAG Retrieval V1: query embedding, vector search, and result formatting."""

    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        embedding_service: EmbeddingService | None = None,
        milvus_search_service: MilvusVectorSearchService | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.embedding_service = embedding_service or get_embedding_service()
        self.milvus_search_service = milvus_search_service or MilvusVectorSearchService(
            collection_name=self.settings.collection_name
        )

    def search(self, query: str, top_k: Any = None) -> dict[str, Any]:
        clean_query = self._validate_query(query)
        safe_top_k = self._normalize_top_k(top_k)
        vector = self._embed_query(clean_query)
        raw_hits = self.milvus_search_service.search(
            vector=vector,
            top_k=safe_top_k,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
        results = self._format_results(raw_hits)

        return {
            "success": True,
            "query": clean_query,
            "results": results,
        }

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty.")
        return query.strip()

    def _normalize_top_k(self, top_k: Any) -> int:
        if top_k is None:
            return self.settings.default_top_k
        try:
            value = int(top_k)
        except (TypeError, ValueError):
            return self.settings.default_top_k

        if value <= 0:
            return self.settings.default_top_k
        return min(value, self.settings.max_top_k)

    def _embed_query(self, query: str) -> list[float]:
        try:
            vector = self.embedding_service.embed_query(query)
        except Exception as exc:
            raise EmbeddingUnavailableError("Embedding service is unavailable.") from exc

        if not vector:
            raise EmbeddingUnavailableError("Embedding service returned an empty vector.")
        return [float(value) for value in vector]

    def _format_results(self, hits: Sequence[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hit in hits:
            entity = self._entity_from_hit(hit)
            score = self._score_from_hit(hit)
            if score < self.settings.min_score:
                continue

            text = self._string_value(entity.get("text"))
            if not text:
                continue

            results.append(
                {
                    "score": score,
                    "text": text,
                    "metadata": self._metadata_from_entity(entity),
                }
            )
        return results

    def _entity_from_hit(self, hit: Any) -> dict[str, Any]:
        hit_mapping = self._as_mapping(hit)
        entity = self._as_mapping(hit_mapping.get("entity"))
        if entity:
            return entity
        return {
            key: value
            for key, value in hit_mapping.items()
            if key not in {"distance", "score", "id"}
        }

    def _score_from_hit(self, hit: Any) -> float:
        hit_mapping = self._as_mapping(hit)
        raw_score = self._first_present(
            hit_mapping.get("score"),
            hit_mapping.get("distance"),
            getattr(hit, "score", None),
            getattr(hit, "distance", None),
        )
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            return 0.0

    def _metadata_from_entity(self, entity: Mapping[str, Any]) -> dict[str, Any]:
        dynamic_meta = self._as_mapping(entity.get("$meta"))
        metadata_json = self._first_present(
            entity.get("metadata_json"),
            dynamic_meta.get("metadata_json"),
        )
        unified_metadata = self._load_metadata_json(metadata_json)
        document = self._as_mapping(unified_metadata.get("document"))
        structure = self._as_mapping(unified_metadata.get("structure"))
        source = self._as_mapping(unified_metadata.get("source"))

        metadata: dict[str, Any] = {
            "file_name": self._string_value(
                self._first_present(entity.get("file_name"), document.get("file_name"))
            ),
            "file_type": self._string_value(
                self._first_present(entity.get("file_type"), document.get("file_type"))
            ),
            "chapter": self._string_value(
                self._first_present(entity.get("chapter"), structure.get("chapter"))
            ),
            "section": self._string_value(
                self._first_present(entity.get("section"), structure.get("section"))
            ),
        }

        chunk_index = self._first_present(
            entity.get("chunk_index"),
            dynamic_meta.get("chunk_index"),
        )
        if chunk_index is not None:
            metadata["chunk_index"] = chunk_index

        page_numbers = self._page_numbers(entity, dynamic_meta, source)
        if page_numbers:
            metadata["page_numbers"] = page_numbers

        line_range = self._line_range(entity, dynamic_meta, source)
        if line_range:
            metadata["line_range"] = line_range

        return metadata

    def _load_metadata_json(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return self._as_mapping(parsed)

    def _page_numbers(
        self,
        entity: Mapping[str, Any],
        dynamic_meta: Mapping[str, Any],
        source: Mapping[str, Any],
    ) -> list[Any]:
        value = self._first_present(
            source.get("page_numbers"),
            entity.get("page_numbers"),
            dynamic_meta.get("page_numbers"),
        )
        page_numbers = self._as_list(value)
        if page_numbers:
            return page_numbers

        page_number = self._first_present(
            entity.get("page_number"),
            dynamic_meta.get("page_number"),
        )
        return [] if page_number in {None, ""} else [page_number]

    def _line_range(
        self,
        entity: Mapping[str, Any],
        dynamic_meta: Mapping[str, Any],
        source: Mapping[str, Any],
    ) -> list[Any]:
        value = self._first_present(
            source.get("line_range"),
            entity.get("line_range"),
            dynamic_meta.get("line_range"),
        )
        line_range = self._as_list(value)
        if line_range:
            return line_range

        start = self._first_present(entity.get("line_start"), dynamic_meta.get("line_start"))
        end = self._first_present(entity.get("line_end"), dynamic_meta.get("line_end"))
        if start in {None, ""} or end in {None, ""}:
            return []
        return [start, end]

    def _as_mapping(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            converted = value.to_dict()
            return dict(converted) if isinstance(converted, Mapping) else {}
        return {}

    def _as_list(self, value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value]

    def _first_present(self, *values: Any) -> Any:
        for value in values:
            if value is not None and value != "":
                return value
        return None

    def _string_value(self, value: Any) -> str:
        return "" if value is None else str(value).strip()


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()
