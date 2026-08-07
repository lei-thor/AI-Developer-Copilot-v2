from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    api_base: str | None
    api_url: str | None
    api_key: str | None
    batch_size: int
    timeout_seconds: float
    normalize: bool
    device: str

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        _load_env_file()

        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai_compatible"),
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            api_base=os.getenv("EMBEDDING_API_BASE") or None,
            api_url=os.getenv("EMBEDDING_API_URL") or None,
            api_key=os.getenv("EMBEDDING_API_KEY") or None,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60")),
            normalize=_env_bool("EMBEDDING_NORMALIZE", True),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )


class OpenAICompatibleEmbeddingProvider:
    """Embedding client for OpenAI-compatible /embeddings endpoints."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        clean_texts = self._validate_texts(texts)
        vectors: list[list[float]] = []
        for batch in self._batches(clean_texts):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        endpoint = self._endpoint()
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": list(texts),
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return self._extract_vectors(data, expected_count=len(texts))

    def _endpoint(self) -> str:
        if self.settings.api_url:
            return self.settings.api_url
        if not self.settings.api_base:
            raise RuntimeError(
                "EMBEDDING_API_BASE or EMBEDDING_API_URL must be configured in .env "
                "when EMBEDDING_PROVIDER=openai_compatible."
            )
        return f"{self.settings.api_base.rstrip('/')}/embeddings"

    def _extract_vectors(self, data: Any, expected_count: int) -> list[list[float]]:
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors = [item.get("embedding") for item in items]
        elif isinstance(data, dict) and isinstance(data.get("embeddings"), list):
            vectors = data["embeddings"]
        elif isinstance(data, list):
            vectors = data
        else:
            raise RuntimeError("Embedding API response does not contain vectors.")

        normalized = [self._normalize_vector(vector) for vector in vectors]
        if len(normalized) != expected_count:
            raise RuntimeError(
                f"Embedding API returned {len(normalized)} vectors, expected {expected_count}."
            )
        return normalized

    def _normalize_vector(self, vector: Any) -> list[float]:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise RuntimeError("Embedding vector must be a numeric sequence.")
        values = [float(value) for value in vector]
        if not values:
            raise RuntimeError("Embedding vector is empty.")
        return values

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if len(clean_texts) != len(texts):
            raise ValueError("Embedding texts cannot contain empty values.")
        return clean_texts

    def _batches(self, texts: Sequence[str]) -> list[list[str]]:
        batch_size = max(self.settings.batch_size, 1)
        return [
            list(texts[index : index + batch_size])
            for index in range(0, len(texts), batch_size)
        ]


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider for sentence-transformers compatible models."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self._model: Any | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if len(clean_texts) != len(texts):
            raise ValueError("Embedding texts cannot contain empty values.")

        vectors = self.model.encode(
            clean_texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local embedding. "
                    "Install it in law_rag or use EMBEDDING_PROVIDER=openai_compatible."
                ) from exc

            self._model = SentenceTransformer(
                self.settings.model,
                device=self.settings.device,
            )
        return self._model


class EmbeddingService:
    """Facade used by ingestion and retrieval services."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or EmbeddingSettings.from_env()
        self._provider = self._create_provider(self.settings)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._provider.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._provider.embed_query(text)

    def _create_provider(self, settings: EmbeddingSettings) -> Any:
        provider = settings.provider.strip().lower()
        if provider in {"openai", "openai_compatible", "api"}:
            return OpenAICompatibleEmbeddingProvider(settings)
        if provider in {"sentence_transformers", "sentence-transformer", "local"}:
            return SentenceTransformerEmbeddingProvider(settings)
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.provider}")


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
