from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.rag_service import (
    RAGModelInvocationError,
    RAGServiceError,
    get_rag_service,
)
from services.retrieval_service import RetrievalError


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RAGChatRequest(BaseModel):
    query: str = Field(..., description="User question.")
    knowledge_base_id: str = Field(default="default")
    top_k: Any = Field(default=5, description="Maximum number of retrieved chunks.")


@router.post("/chat", response_model=None)
def rag_chat(payload: RAGChatRequest) -> Any:
    """Answer a question from the indexed knowledge base."""
    try:
        result = get_rag_service().chat(
            query=payload.query,
            knowledge_base_id=payload.knowledge_base_id,
            top_k=payload.top_k,
        )
        return result.to_dict()
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc)},
        )
    except RAGModelInvocationError as exc:
        logger.exception("RAG chat model request failed.")
        return JSONResponse(
            status_code=502,
            content={"success": False, "message": str(exc)},
        )
    except RetrievalError as exc:
        logger.exception("RAG retrieval failed.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": str(exc)},
        )
    except RAGServiceError as exc:
        logger.exception("RAG service failed.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(exc)},
        )
    except Exception:
        logger.exception("Unexpected RAG chat failure.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "RAG chat failed."},
        )
