from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from services.document_review_service import (
    ChunkReviewInput,
    DocumentReviewService,
    to_dict,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["document-review"])
document_review_service = DocumentReviewService()

ReviewStatus = Literal["PENDING", "APPROVED", "REJECTED", "MODIFIED"]


class ChunkReviewRequest(BaseModel):
    review_status: ReviewStatus = Field(..., description="Manual review status.")
    review_comment: str = Field(default="", description="Reviewer comment.")
    text: str | None = Field(default=None, description="Optional modified chunk text.")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata patch.")


@router.get("/document-review")
def list_document_reviews() -> dict[str, Any]:
    """List parsed documents that can be reviewed."""
    return {"documents": document_review_service.list_documents()}


@router.get("/document-review/{document_id}")
def get_document_review(document_id: str) -> dict[str, Any]:
    """Return the document review DTO for a parsed document."""
    try:
        return to_dict(document_review_service.get_document_review(document_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/document-review/{document_id}/chunks")
def get_document_chunks(document_id: str) -> dict[str, Any]:
    """Return chunks for manual review."""
    try:
        chunks = document_review_service.get_chunks(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"chunks": to_dict(chunks)}


@router.get("/document-review/{document_id}/quality")
def get_document_quality(document_id: str) -> dict[str, Any]:
    """Return automatic quality analysis results."""
    try:
        return to_dict(document_review_service.get_quality_report(document_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/document-review/{document_id}/source")
def get_document_source(document_id: str) -> FileResponse:
    """Stream the original PDF for side-by-side review."""
    try:
        source_path = document_review_service.get_source_pdf_path(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        source_path,
        media_type="application/pdf",
        filename=source_path.name,
        content_disposition_type="inline",
    )


@router.get("/document-review/{document_id}/source-text")
def get_document_source_text(document_id: str) -> PlainTextResponse:
    """Return original Markdown/text source for non-PDF review."""
    try:
        text = document_review_service.get_source_text(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PlainTextResponse(
        text,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/document-review/{document_id}/pages/{page_number}.png")
def get_document_page_image(document_id: str, page_number: int) -> Response:
    """Render a PDF page as PNG so the review UI never triggers PDF downloads."""
    try:
        image_bytes = document_review_service.render_pdf_page_png(document_id, page_number)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.put("/chunk-review/{chunk_id}")
def update_chunk_review(chunk_id: str, payload: ChunkReviewRequest) -> dict[str, Any]:
    """Persist manual review status, comments, text, and metadata patches."""
    try:
        record = document_review_service.update_chunk_review(
            chunk_id,
            ChunkReviewInput(
                review_status=payload.review_status,
                review_comment=payload.review_comment,
                text=payload.text,
                metadata=payload.metadata,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Chunk review updated: chunk_id=%s status=%s", chunk_id, payload.review_status)
    return {"review": record}
