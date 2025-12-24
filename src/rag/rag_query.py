# src/rag/rag_query.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.ingest.chroma_client import col
from src.ingest.embeddings import encode
from src.ingest.tickers import DJIA_TICKERS
from src.llm.llm_client import generate_answer
from datetime import datetime, timedelta, timezone


# --------- detección del ticker ---------

_TICKERS_SET = set(DJIA_TICKERS)

_NAME_ALIASES = {
    "microsoft": "MSFT",
    "apple": "AAPL",
    "ibm": "IBM",
    "visa": "V",
    "mcdonald": "MCD",
    "mcdonalds": "MCD",
    "boeing": "BA",
    "tesla": "TSLA",
}


def _detect_ticker(question: str) -> Optional[str]:
    """
    Devuelve el ticker DJIA detectado en la pregunta, o None si no encuentra.
    """
    if not question:
        return None

    # 1) Símbolo literal (MSFT, IBM, AAPL, etc.)
    tokens = re.findall(r"[A-Za-z]+", question)
    for tok in tokens:
        up = tok.upper()
        if up in _TICKERS_SET:
            return up

    # 2) Alias por nombre de empresa
    ql = question.lower()
    for name, tk in _NAME_ALIASES.items():
        if name in ql:
            return tk

    return None


# --------- búsqueda vectorial en Chroma ---------

def _vector_search(question: str, ticker: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica en Chroma forzando recencia:
    solo busca dentro de los últimos 10 días (date_num).
    """
    qemb = encode([question], is_query=True)

    cutoff_num = int(
        (datetime.now(timezone.utc).date() - timedelta(days=10)).strftime("%Y%m%d")
    )

    res = col.query(
        query_embeddings=qemb,
        n_results=n_results,
        where={
            "$and": [
                {"ticker": ticker},
                {"date_num": {"$gte": cutoff_num}},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for i in range(min(len(ids), len(docs), len(metas), len(dists))):
        hits.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": float(dists[i]),
            }
        )
    return hits



def _date_num_from_meta(meta: Dict[str, Any]) -> int:
    meta = meta or {}

    # 1) date_num (ingesta OHLC)
    if "date_num" in meta:
        try:
            return int(meta["date_num"])
        except Exception:
            pass

    # 2) date (YYYY-MM-DD)
    if "date" in meta:
        try:
            return int(str(meta["date"]).replace("-", ""))
        except Exception:
            pass

    return 0


# --------- función principal: RAG ---------

def ask(question: str) -> Dict[str, Any]:
    """
    Punto de entrada del sistema RAG.

    1. Detecta el ticker a partir de la pregunta.
    2. Recupera documentos relevantes de la base vectorial Chroma.
    3. Ordena los documentos por fecha (más recientes primero) y construye un CONTEXTO.
    4. Llama a generate_answer() para que el LLM (Gemini/OpenAI o modo free)
       genere una respuesta usando EXCLUSIVAMENTE ese contexto (BBDD vectorial).
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Formula una pregunta, por ejemplo: «¿Cómo va Microsoft hoy?»",
            "sources": [],
        }

    ticker = _detect_ticker(question)
    if not ticker:
        return {
            "answer": (
                "No he podido identificar el valor. "
                "Incluye el nombre de la empresa o su ticker (por ejemplo, MSFT, IBM, AAPL...)."
            ),
            "sources": [],
        }

    # 1) Recuperar desde la base vectorial
    hits = _vector_search(question, ticker, n_results=60)  # o 90
    hits_sorted = sorted(hits, key=lambda h: _date_num_from_meta(h["metadata"]), reverse=True)

    if not hits:
        return {
            "answer": (
                f"No tengo datos ingestados en la base vectorial para {ticker} todavía. "
                "Ejecuta la ingesta y vuelve a preguntar."
            ),
            "sources": [],
        }

    # 2) Ordenar por fecha (más reciente primero)
    hits_sorted = sorted(
        hits,
        key=lambda h: _date_num_from_meta(h.get("metadata", {})),
        reverse=True,
    )

    # 3) Montar el contexto: concatenamos varios documentos ordenados
    context_parts = []
    for h in hits_sorted[:3]:
        meta = h.get("metadata", {}) or {}
        fecha = meta.get("date") or meta.get("fecha") or "fecha-desconocida"
        context_parts.append(f"[DOC - {ticker} - {fecha}]\n{h['document']}")

    context = "\n\n".join(context_parts)

    # 4) Llamar al LLM / modo free para generar la respuesta final
    #    generate_answer ya limita al CONTEXTO y no añade info externa.
    result = generate_answer(question, context, sources=hits_sorted[:5])

    return result
