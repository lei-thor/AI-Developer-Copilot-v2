from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.retrieval_service import (
    EmbeddingUnavailableError,
    RetrievalService,
    VectorSearchUnavailableError,
    get_retrieval_service,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge-search"])


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., description="User query text.")
    top_k: Any = Field(default=5, description="Maximum number of chunks to return.")


@router.post("/search", response_model=None)
def search_knowledge(payload: KnowledgeSearchRequest) -> Any:
    """Search indexed knowledge chunks with Retrieval V1."""
    try:
        service: RetrievalService = get_retrieval_service()
        return service.search(query=payload.query, top_k=payload.top_k)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc)},
        )
    except EmbeddingUnavailableError as exc:
        logger.exception("Query embedding failed.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": str(exc)},
        )
    except VectorSearchUnavailableError as exc:
        logger.exception("Milvus vector search failed.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": str(exc)},
        )
    except Exception:
        logger.exception("Knowledge search failed.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Knowledge search failed."},
        )
