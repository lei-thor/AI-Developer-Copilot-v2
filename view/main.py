from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from view.document_review_api import router as document_review_router
from view.upload_api import router as upload_router


VIEW_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = VIEW_ROOT / "static"


app = FastAPI(title="AI Developer Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_review_router)
app.include_router(upload_router)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/document-review")
def document_review_page() -> FileResponse:
    return FileResponse(
        STATIC_ROOT / "document_review.html",
        headers={"Cache-Control": "no-store"},
    )
