"""
Clearpath RAG Backend — FastAPI application entry point.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.chat import router as chat_router
from app.rag.retriever import retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = FastAPI(
    title="Clearpath RAG API",
    version="1.0.0",
    description="Trust-calibrated RAG customer support system for Clearpath.",
)
@app.on_event("startup")
async def warmup_models() -> None:
    try:
        retriever._load()
    except Exception:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(chat_router, prefix="")  # exposes POST /query at root


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}