from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from services.file_upload_service import (
    FileUploadService,
    UnsupportedFileTypeError,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge/files", tags=["knowledge-upload"])
file_upload_service = FileUploadService()


@router.post("/upload", response_model=None)
def upload_file(file: UploadFile = File(...)) -> Any:
    """Save a document and dispatch it to the matching ingestion service."""
    filename = file.filename or ""
    try:
        result = file_upload_service.upload_and_ingest(filename, file.file)
        return result.to_dict()
    except UnsupportedFileTypeError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc)},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc)},
        )
    except Exception:
        logger.exception("Document upload and ingestion failed: %s", filename)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "File upload or ingestion failed."},
        )
