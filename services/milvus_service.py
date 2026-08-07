from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover - handled when the service is used.
    MilvusClient = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env_file(env_path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
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


@dataclass(frozen=True)
class MilvusSettings:
    uri: str
    host: str
    port: int
    db_name: str
    token: str | None
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "MilvusSettings":
        _load_env_file()

        host = os.getenv("MILVUS_HOST")
        if not host:
            raise RuntimeError("MILVUS_HOST must be configured in .env.")

        port = int(os.getenv("MILVUS_PORT", "19530"))
        uri = os.getenv("MILVUS_URI")
        if not uri:
            raise RuntimeError("MILVUS_URI must be configured in .env.")

        token = os.getenv("MILVUS_TOKEN") or None

        return cls(
            uri=uri,
            host=host,
            port=port,
            db_name=os.getenv("MILVUS_DB_NAME", "default"),
            token=token,
            timeout_seconds=float(os.getenv("MILVUS_TIMEOUT_SECONDS", "10")),
        )


class MilvusConnectionService:
    """Lazy Milvus client factory for the service layer."""

    def __init__(self, settings: MilvusSettings | None = None) -> None:
        self.settings = settings or MilvusSettings.from_env()
        self._client: MilvusClient | None = None

    @property
    def client(self) -> MilvusClient:
        if MilvusClient is None:
            raise RuntimeError(
                "pymilvus is not installed. Install it before using MilvusConnectionService."
            )

        if self._client is None:
            client_kwargs: dict[str, Any] = {
                "uri": self.settings.uri,
                "db_name": self.settings.db_name,
                "timeout": self.settings.timeout_seconds,
            }
            if self.settings.token:
                client_kwargs["token"] = self.settings.token

            self._client = MilvusClient(**client_kwargs)

        return self._client

    def health_check(self) -> dict[str, Any]:
        collections = self.client.list_collections()
        return {
            "status": "ok",
            "uri": self.settings.uri,
            "db_name": self.settings.db_name,
            "collections_count": len(collections),
        }

    def list_collections(self) -> list[str]:
        return list(self.client.list_collections())

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None


@lru_cache(maxsize=1)
def get_milvus_service() -> MilvusConnectionService:
    return MilvusConnectionService()
