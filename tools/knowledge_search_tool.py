from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

from services.retrieval_service import RetrievalService, get_retrieval_service


@dataclass
class KnowledgeSearchTool:
    """Future Agent tool that exposes the existing knowledge search capability."""

    retrieval_service: RetrievalService = field(default_factory=get_retrieval_service)
    name: str = "knowledge_search"
    description: str = "Search the developer knowledge base for relevant chunks."

    def run(self, query: str, top_k: Any = 5) -> list[dict[str, Any]]:
        response = self.retrieval_service.search(query=query, top_k=top_k)
        return list(response.get("results", []))

    def invoke(self, tool_input: str | Mapping[str, Any]) -> list[dict[str, Any]]:
        if isinstance(tool_input, str):
            return self.run(query=tool_input)
        if not isinstance(tool_input, Mapping):
            raise ValueError("tool_input must be a query string or mapping.")
        return self.run(
            query=str(tool_input.get("query") or ""),
            top_k=tool_input.get("top_k", 5),
        )


@lru_cache(maxsize=1)
def get_knowledge_search_tool() -> KnowledgeSearchTool:
    return KnowledgeSearchTool()
