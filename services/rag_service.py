from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from chat_model.chat_model_service import ChatModelService, get_chat_model_service
from services.retrieval_service import RetrievalService, get_retrieval_service


logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE_ID = "default"
NO_KNOWLEDGE_ANSWER = "知识库中没有找到相关内容。"

RAG_PROMPT = PromptTemplate.from_template(
    """你是一个专业知识库助手。
请严格根据下面提供的参考资料回答用户问题。

回答规则：
1. 优先使用参考资料中的内容回答。
2. 不要编造参考资料中不存在的事实。
3. 如果参考资料不足以回答问题，请明确说明。
4. 回答应清晰、准确，并尽量结合资料中的技术术语。

参考资料：
{context}

用户问题：
{question}
"""
)


class RAGServiceError(RuntimeError):
    """Base error for RAG application-layer failures."""


class RAGModelInvocationError(RAGServiceError):
    """Raised when the configured chat model cannot answer a request."""


class RetrievalServiceLangChainRetriever(BaseRetriever):
    """Adapt RetrievalService results into LangChain Document objects."""

    retrieval_service: Any
    top_k: Any = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        response = self.retrieval_service.search(query=query, top_k=self.top_k)
        results = response.get("results", [])
        return [
            self._to_document(result)
            for result in results
            if isinstance(result, Mapping) and str(result.get("text", "")).strip()
        ]

    def _to_document(self, result: Mapping[str, Any]) -> Document:
        raw_metadata = result.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        page_numbers = self._as_list(metadata.get("page_numbers"))

        document_metadata: dict[str, Any] = {
            "file_name": str(metadata.get("file_name") or ""),
            "file_type": str(metadata.get("file_type") or ""),
            "chapter": str(metadata.get("chapter") or ""),
            "section": str(metadata.get("section") or ""),
            "chunk_index": metadata.get("chunk_index"),
            "score": self._score(result.get("score")),
        }
        if page_numbers:
            document_metadata["page"] = page_numbers[0]
            document_metadata["page_numbers"] = page_numbers
        if metadata.get("line_range"):
            document_metadata["line_range"] = self._as_list(metadata["line_range"])

        return Document(
            page_content=str(result["text"]).strip(),
            metadata=document_metadata,
        )

    def _as_list(self, value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [value]

    def _score(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


@dataclass(frozen=True)
class RAGChatResult:
    success: bool
    answer: str
    sources: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "answer": self.answer,
            "sources": self.sources,
        }


class RAGService:
    """Coordinate retrieval, LangChain prompt building, and chat model output."""

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        chat_model: ChatModelService | None = None,
        prompt_template: PromptTemplate | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service or get_retrieval_service()
        self.chat_model = chat_model or get_chat_model_service()
        self.prompt_template = prompt_template or RAG_PROMPT

    def chat(
        self,
        query: str,
        knowledge_base_id: str = DEFAULT_KNOWLEDGE_BASE_ID,
        top_k: Any = 5,
    ) -> RAGChatResult:
        clean_query = self._validate_query(query)
        self._validate_knowledge_base_id(knowledge_base_id)

        retriever = RetrievalServiceLangChainRetriever(
            retrieval_service=self.retrieval_service,
            top_k=top_k,
        )
        documents = retriever.invoke(clean_query)
        if not documents:
            return RAGChatResult(
                success=True,
                answer=NO_KNOWLEDGE_ANSWER,
                sources=[],
            )

        context = self._build_context(documents)
        prompt = self.prompt_template.format(context=context, question=clean_query)
        try:
            answer = self.chat_model.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            logger.exception("RAG chat model invocation failed.")
            raise RAGModelInvocationError("LLM invocation failed.") from exc

        return RAGChatResult(
            success=True,
            answer=answer,
            sources=self._build_sources(documents),
        )

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty.")
        return query.strip()

    def _validate_knowledge_base_id(self, knowledge_base_id: str) -> None:
        normalized = (knowledge_base_id or DEFAULT_KNOWLEDGE_BASE_ID).strip()
        if normalized != DEFAULT_KNOWLEDGE_BASE_ID:
            raise ValueError(
                "Only knowledge_base_id='default' is available in the current version."
            )

    def _build_context(self, documents: Sequence[Document]) -> str:
        sections: list[str] = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            source = " | ".join(
                value
                for value in [
                    metadata.get("file_name", ""),
                    metadata.get("chapter", ""),
                    metadata.get("section", ""),
                    self._page_label(metadata.get("page")),
                ]
                if value
            )
            header = f"[参考资料 {index}]"
            if source:
                header = f"{header} 来源：{source}"
            sections.append(f"{header}\n{document.page_content}")
        return "\n\n".join(sections)

    def _build_sources(self, documents: Sequence[Document]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for document in documents:
            metadata = document.metadata
            source = {
                "file_name": metadata.get("file_name", ""),
                "file_type": metadata.get("file_type", ""),
                "chapter": metadata.get("chapter", ""),
                "section": metadata.get("section", ""),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
            }
            identity = (
                source["file_name"],
                source["chapter"],
                source["section"],
                source["page"],
                source["chunk_index"],
            )
            if identity in seen:
                continue
            seen.add(identity)
            sources.append(source)
        return sources

    def _page_label(self, page: Any) -> str:
        return f"第{page}页" if page not in {None, ""} else ""


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()
