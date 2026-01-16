from __future__ import annotations

from typing import List, Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.rag_query import ask
from src.ingest.ingest_djia import ingest as ingest_djia, cleanup as cleanup_djia
from src.ingest.tickers import DJIA_TICKERS
from src.ingest.ingest_news import ingest as ingest_news, cleanup as cleanup_news

app = FastAPI(title="WallStreet Consulting - RAG DJIA")


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    tickers: Optional[List[Any]] = None

    ingest_prices: bool = True
    days: int = 30
    cleanup_days: int = 30

    ingest_docs: bool = False
    news_limit: int = 20
    news_cleanup_days: int = 30


@app.get("/health")
def health():
    return {"ok": True, "main": "src/api/main.py"}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.question)


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    # normalización robusta de tickers
    raw_tickers = req.tickers or []
    tickers_clean: List[str] = []
    for t in raw_tickers:
        if isinstance(t, str):
            s = t.strip().upper()
            if s:
                tickers_clean.append(s)

    if not tickers_clean or tickers_clean == ["STRING"]:
        tickers = list(DJIA_TICKERS)
    else:
        tickers = tickers_clean

    out = {"message": "Ingesta completada", "tickers": tickers, "prices": None, "docs": None}

    # 1) precios
    if req.ingest_prices:
        n_price_docs = ingest_djia(tickers, days=req.days)
        cleanup_djia(days=req.cleanup_days)
        out["prices"] = {"ingested_docs": n_price_docs, "days": req.days, "cleanup_days": req.cleanup_days}

    # 2) docs/noticias
    if req.ingest_docs:
        n_doc_docs = ingest_news(tickers, limit=req.news_limit)

        # ✅ desactivado para no borrar por published_num=0
        # cleanup_news(days=req.news_cleanup_days)

        out["docs"] = {
            "ingested_docs": n_doc_docs,
            "news_limit": req.news_limit,
            "cleanup_days": req.news_cleanup_days,
            "cleanup_enabled": False,
        }

    if not req.ingest_prices and not req.ingest_docs:
        out["message"] = "Nada que ingerir: activa ingest_prices y/o ingest_docs."

    return out
