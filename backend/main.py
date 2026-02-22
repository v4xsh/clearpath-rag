"""
Clearpath RAG Backend — FastAPI application entry point.
"""

from __future__ import annotations
import logging
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from backend.app.api.chat import router as chat_router
from backend.app.rag.retriever import retriever

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
    except Exception as e:
        logging.error(f"Warmup failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(chat_router, prefix="")  

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}