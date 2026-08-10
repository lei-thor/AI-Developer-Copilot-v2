from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - handled when the service is used.
    ChatOpenAI = None  # type: ignore[assignment]


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


@dataclass(frozen=True)
class ChatModelSettings:
    provider: str
    model: str | None
    api_base: str | None
    api_key: str | None
    temperature: float
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "ChatModelSettings":
        _load_env_file()
        return cls(
            provider=os.getenv("CHAT_MODEL_PROVIDER", "openai_compatible"),
            model=os.getenv("CHAT_MODEL_NAME") or None,
            api_base=os.getenv("CHAT_MODEL_API_BASE") or None,
            api_key=os.getenv("CHAT_MODEL_API_KEY") or None,
            temperature=float(os.getenv("CHAT_MODEL_TEMPERATURE", "0.2")),
            timeout_seconds=float(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "60")),
        )


class ChatModelService:
    """Expose one invoke() interface for OpenAI-compatible chat models."""

    def __init__(
        self,
        settings: ChatModelSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or ChatModelSettings.from_env()
        self._client = client

    def invoke(self, messages: Sequence[Any]) -> str:
        response = self.client.invoke(list(messages))
        content = getattr(response, "content", response)
        text = self._content_to_text(content)
        if not text:
            raise RuntimeError("Chat model returned an empty response.")
        return text

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        provider = self.settings.provider.strip().lower()
        if provider not in {"openai", "openai_compatible", "api"}:
            raise ValueError(f"Unsupported CHAT_MODEL_PROVIDER: {self.settings.provider}")
        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required for the configured chat model."
            )
        if not self.settings.model:
            raise RuntimeError("CHAT_MODEL_NAME must be configured in .env.")
        if not self.settings.api_base:
            raise RuntimeError("CHAT_MODEL_API_BASE must be configured in .env.")
        if not self.settings.api_key:
            raise RuntimeError("CHAT_MODEL_API_KEY must be configured in .env.")

        return ChatOpenAI(
            model=self.settings.model,
            api_key=self.settings.api_key,
            base_url=self.settings.api_base,
            temperature=self.settings.temperature,
            timeout=self.settings.timeout_seconds,
        )

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return str(content).strip()


@lru_cache(maxsize=1)
def get_chat_model_service() -> ChatModelService:
    return ChatModelService()
