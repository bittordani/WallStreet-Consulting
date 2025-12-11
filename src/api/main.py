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

    # Normalizamos tickers recibidos
    raw_tickers = (req.tickers or [])
    tickers_clean = [t.strip().upper() for t in raw_tickers if t and t.strip()]

    # Caso especial: Swagger pone por defecto ["string"] como ejemplo.
    # Si el usuario no lo cambia, lo ignoramos y usamos todo el DJIA.
    if not tickers_clean or tickers_clean == ["STRING"]:
        tickers = list(DJIA_TICKERS)
    else:
        tickers = tickers_clean

    n_docs = ingest_djia(tickers, days=req.days)
    cleanup_djia(days=req.cleanup_days)

    return {
        "message": "Ingesta completada",
        "tickers": tickers,
        "ingested_docs": n_docs,
        "days": req.days,
        "cleanup_days": req.cleanup_days,
    }
