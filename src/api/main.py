# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# carga .env por si lo usas para LLM/chroma
load_dotenv()

# --- importa tu lógica existente ---
from src.rag.rag_query import ask as rag_ask
from src.ingest.ingest_djia import make_metric_doc, make_news_docs, upsert_docs
from src.ingest.tickers import DJIA_TICKERS

app = FastAPI(title="RAG Dow Jones con Chroma y FastAPI", version="1.0.0")

# CORS para front local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # reduce a http://localhost:3000 si haces front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    question: str

class IngestIn(BaseModel):
    tickers: list[str] | None = None   # si None, ingesta DJIA completo

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/tickers")
def tickers():
    return {"tickers": DJIA_TICKERS}

@app.post("/ask")
def ask(payload: AskIn):
    try:
        res = rag_ask(payload.question)
        return {"answer": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest(payload: IngestIn):
    """
    Ingesta rápida:
    - si envías tickers: solo esos
    - si no: DJIA completo
    """
    tks = payload.tickers or DJIA_TICKERS
    batch = []
    ok, warn = 0, []

    for tk in tks:
        try:
            mtxt, mmeta, mid = make_metric_doc(tk)
            batch.append((mtxt, mmeta, mid))
            # titulares (opcional; comenta si no quieres)
            batch.extend(make_news_docs(tk))
            ok += 1
        except Exception as e:
            warn.append(f"{tk}: {e}")

    if batch:
        upsert_docs(batch)

    return {"ingested": ok, "warnings": warn}
