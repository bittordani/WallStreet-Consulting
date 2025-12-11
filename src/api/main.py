from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.rag_query import ask
from src.ingest.ingest_djia import ingest as ingest_djia, cleanup as cleanup_djia
from src.ingest.tickers import DJIA_TICKERS

app = FastAPI(title="WallStreet Consulting - RAG DJIA")


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    tickers: Optional[List[str]] = None  # si es None -> todo el DJIA
    days: int = 30                       # días hacia atrás a descargar
    cleanup_days: int = 30               # borrar datos anteriores a hoy - cleanup_days


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    """
    Endpoint principal de consulta RAG.
    Usa la base de datos vectorial (Chroma) + LLM (Gemini/OpenAI o modo free).
    """
    return ask(req.question)


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    """
    Lanza la ingesta de datos de cotizaciones en la BBDD vectorial (Chroma).
    - Si no se pasan tickers -> ingesta de todo el Dow Jones (DJIA_TICKERS).
    - days: cuántos días hacia atrás descargar por ticker.
    - cleanup_days: elimina de Chroma los documentos con fecha anterior a hoy - cleanup_days.
    """
    if req.tickers:
        tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    else:
        tickers = list(DJIA_TICKERS)

    n_docs = ingest_djia(tickers, days=req.days)
    cleanup_djia(days=req.cleanup_days)

    return {
        "message": "Ingesta completada",
        "tickers": tickers,
        "ingested_docs": n_docs,
        "days": req.days,
        "cleanup_days": req.cleanup_days,
    }
