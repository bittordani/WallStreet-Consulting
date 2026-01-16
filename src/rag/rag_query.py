# src/rag/rag_query.py
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.ingest.chroma_client import col_prices, col_news
from src.ingest.embeddings import encode
from src.ingest.tickers import DJIA_TICKERS
from src.llm.llm_client import generate_answer


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


# --------- routing: ¿precios o documentos? ---------

_PRICE_KEYWORDS = (
    "precio", "cotiza", "cotización", "cierre", "apertura", "máximo", "minimo", "mínimo",
    "volumen", "variación", "variacion", "sube", "baja", "%", "porcentaje",
    "hoy", "ayer", "último", "ultimo", "sesión", "sesion"
)

_DOCS_KEYWORDS = (
    "noticia", "noticias", "titular", "titulares", "qué ha pasado", "que ha pasado",
    "por qué", "porque", "motivo", "causa", "razón", "razones", "riesgo", "riesgos",
    "informe", "filing", "10-k", "10q", "10-q", "8-k", "earnings", "resultados",
    "guidance", "comunicado", "press release", "rumor", "rumores"
)


def _infer_mode(question: str) -> str:
    """
    Decide si la pregunta debe responderse con:
      - mode="prices": datos estructurados (precios)
      - mode="docs": RAG de documentos (noticias/filings)
    Heurística simple y efectiva para demo/entrega.
    """
    q = (question or "").strip().lower()

    # Si pregunta explícitamente por noticias/razones, manda a docs
    if any(k in q for k in _DOCS_KEYWORDS):
        return "docs"

    # Si pregunta por precio / hoy / cierre / etc, manda a prices
    if any(k in q for k in _PRICE_KEYWORDS):
        return "prices"

    # Por defecto, docs (más defendible como RAG “real”)
    return "docs"


# --------- búsqueda vectorial ---------

def _vector_search_prices(question: str, ticker: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica en Chroma (colección precios) forzando recencia:
    últimos 10 días.
    """
    qemb = encode([question], is_query=True)

    cutoff_num = int(
        (datetime.now(timezone.utc).date() - timedelta(days=10)).strftime("%Y%m%d")
    )

    res = col_prices.query(
        query_embeddings=qemb,
        n_results=n_results,
        where={
            "$and": [
                {"ticker": ticker},
                {"date_num": {"$gte": cutoff_num}},
                # opcional si lo tienes:
                # {"doc_type": "prices"},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )

    return _normalize_hits(res)


def _vector_search_docs(question: str, ticker: str, n_results: int = 8) -> List[Dict[str, Any]]:
    """
    Búsqueda semántica en Chroma (colección docs/noticias/filings).
    Forzamos recencia suave: últimos 30 días si existe published_num.
    """
    qemb = encode([question], is_query=True)

    cutoff_num = int(
        (datetime.now(timezone.utc).date() - timedelta(days=30)).strftime("%Y%m%d")
    )

    # Algunos setups pueden no tener published_num al principio.
    # Intentamos con filtro; si falla, repetimos sin filtro.
    where_with_recency = {
        "$and": [
            {"ticker": ticker},
            {"published_num": {"$gte": cutoff_num}},
            # opcional:
            # {"doc_type": "news"},
        ]
    }

    try:
        res = col_news.query(
            query_embeddings=qemb,
            n_results=n_results,
            where=where_with_recency,
            include=["documents", "metadatas", "distances"],
        )
        hits = _normalize_hits(res)
        if hits:
            return hits
    except Exception:
        pass

    # Fallback sin recencia si aún no tienes published_num
    res = col_news.query(
        query_embeddings=qemb,
        n_results=n_results,
        where={"ticker": ticker},
        include=["documents", "metadatas", "distances"],
    )
    return _normalize_hits(res)


def _normalize_hits(res: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    """
    Para ordenar por recencia:
      - prices: date_num / date
      - docs: published_num / published_at
    """
    meta = meta or {}

    for k in ("date_num", "published_num"):
        if k in meta:
            try:
                return int(meta[k])
            except Exception:
                pass

    for k in ("date", "published_at", "published"):
        if k in meta:
            try:
                return int(str(meta[k]).replace("-", "")[:10].replace("-", ""))
            except Exception:
                pass

    return 0


# --------- función principal: RAG ---------

def ask(question: str) -> Dict[str, Any]:
    """
    Punto de entrada del sistema.

    - Detecta ticker.
    - Decide modo: prices vs docs.
    - Recupera hits de la colección correspondiente.
    - Construye contexto:
        * docs: con etiquetas [S1], [S2]... (para citas)
        * prices: por fecha reciente
    - Llama a generate_answer(question, context, sources=..., mode=...)
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Formula una pregunta, por ejemplo: «¿Qué noticias recientes hay sobre Microsoft?» o «¿Cuál fue el último cierre de MSFT?»",
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

    mode = _infer_mode(question)

    # 1) Recuperar desde la base vectorial (colección según modo)
    if mode == "prices":
        hits = _vector_search_prices(question, ticker, n_results=60)
    else:
        hits = _vector_search_docs(question, ticker, n_results=12)

    if not hits:
        tip = (
            f"No tengo documentos ingestados para {ticker} en el modo '{mode}'. "
            "Ejecuta la ingesta correspondiente y vuelve a preguntar."
        )
        return {"answer": tip, "sources": []}

    # 2) Ordenar por recencia (más reciente primero)
    hits_sorted = sorted(hits, key=lambda h: _date_num_from_meta(h.get("metadata", {})), reverse=True)

    # 3) Montar el contexto
    if mode == "docs":
        # Contexto con citas [S1], [S2]... (esto encaja con tu prompt de docs)
        context_parts: List[str] = []
        for i, h in enumerate(hits_sorted[:5], start=1):
            meta = h.get("metadata", {}) or {}
            published = meta.get("published_at") or meta.get("date") or "fecha-desconocida"
            publisher = meta.get("publisher") or meta.get("source") or "fuente-desconocida"
            url = meta.get("source_url") or meta.get("url") or ""
            header = f"[S{i}] {ticker} · {published} · {publisher}"
            if url:
                header += f" · {url}"
            context_parts.append(f"{header}\n{h.get('document','')}".strip())
        context = "\n\n".join(context_parts)

        sources_out = hits_sorted[:5]

    else:
        # Precios: context por días recientes (tu estilo anterior)
        context_parts = []
        for h in hits_sorted[:3]:
            meta = h.get("metadata", {}) or {}
            fecha = meta.get("date") or "fecha-desconocida"
            context_parts.append(f"[DOC - {ticker} - {fecha}]\n{h.get('document','')}".strip())
        context = "\n\n".join(context_parts)

        sources_out = hits_sorted[:5]

    # 4) Llamar al LLM / modo free
    result = generate_answer(question, context, sources=sources_out, mode=mode)
    return result
