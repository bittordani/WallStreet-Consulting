# src/ingest/ingest_news.py
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import yfinance as yf

from .chroma_client import col_news as col
from .tickers import DJIA_TICKERS
from .embeddings import encode


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, dict):
        for k in ("content", "text", "summary", "snippet", "title", "description"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""
    if isinstance(x, list):
        parts = [_as_text(i) for i in x]
        return " ".join(p for p in parts if p)
    return str(x)


def _safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def _iso_date_from_ts(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
    except Exception:
        return None


def _date_num(iso_date: Optional[str]) -> int:
    if not iso_date:
        return 0
    try:
        return int(iso_date.replace("-", ""))
    except Exception:
        return 0


def _build_docs_for_ticker(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    t = yf.Ticker(ticker)
    items = t.news or []
    items = items[: max(0, int(limit))]

    docs: List[Dict[str, Any]] = []
    today_iso = datetime.now(timezone.utc).date().isoformat()

    for i, it in enumerate(items):
        title = _as_text(it.get("title")).strip()
        url = _as_text(it.get("link") or it.get("url")).strip()
        publisher = _as_text(it.get("publisher")).strip()
        ts = it.get("providerPublishTime")

        published_at = _iso_date_from_ts(ts)

        # ✅ FIX: si no hay fecha, usa fecha de ingesta (hoy)
        # así cleanup no se lo carga por published_num=0
        published_at_missing = False
        if not published_at:
            published_at = today_iso
            published_at_missing = True

        published_num = _date_num(published_at)

        snippet_raw = it.get("summary") or it.get("snippet") or it.get("content") or it.get("description")
        snippet = _as_text(snippet_raw).strip()

        text_parts: List[str] = []
        if title:
            text_parts.append(f"Título: {title}")
        if snippet:
            text_parts.append(f"Resumen: {snippet}")
        if publisher:
            text_parts.append(f"Medio: {publisher}")
        if published_at:
            text_parts.append(f"Fecha: {published_at}")
        if url:
            text_parts.append(f"URL: {url}")

        document = "\n".join(text_parts).strip()
        if not document:
            continue

        key = (url or title or f"{ticker}_{i}").encode("utf-8", errors="ignore")
        doc_hash = str(abs(hash(key)))
        doc_id = f"{ticker}_news_{published_at}_{doc_hash}"

        docs.append(
            {
                "id": doc_id,
                "document": document,
                "metadata": {
                    "ticker": ticker,
                    "doc_type": "news",
                    "source": "yfinance_news",
                    "publisher": publisher or _safe_domain(url),
                    "source_url": url,
                    "published_at": published_at,
                    "published_num": published_num,
                    "published_at_missing": published_at_missing,
                },
            }
        )

    return docs


def ingest(tickers: Iterable[str], limit: int = 20) -> int:
    collection = col

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for ticker in tickers:
        ticker = (ticker or "").upper().strip()
        if not ticker:
            continue

        docs = _build_docs_for_ticker(ticker, limit=limit)
        if not docs:
            continue

        all_ids.extend(d["id"] for d in docs)
        all_docs.extend(d["document"] for d in docs)
        all_metadatas.extend(d["metadata"] for d in docs)

    if not all_docs:
        return 0

    embeddings = encode(all_docs, is_query=False)

    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )
    else:
        collection.add(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )

    return len(all_docs)


def cleanup(days: int = 30) -> None:
    """
    ✅ FIX: solo borra si published_num > 0 y es antiguo.
    """
    collection = col

    cutoff_date = (datetime.now(timezone.utc).date() - timedelta(days=days))
    cutoff_num = int(cutoff_date.strftime("%Y%m%d"))

    try:
        collection.delete(
            where={
                "$and": [
                    {"published_num": {"$gt": 0}},
                    {"published_num": {"$lt": cutoff_num}},
                ]
            }
        )
    except Exception:
        pass


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]

    if argv:
        tickers = [t.strip().upper() for t in argv if t.strip()]
    else:
        tickers = list(DJIA_TICKERS)

    print(f"[INGEST_NEWS] Tickers: {', '.join(tickers)}")
    n_docs = ingest(tickers, limit=20)
    print(f"[INGEST_NEWS] Documentos ingestados/actualizados: {n_docs}")

    cleanup(days=30)
    print("[INGEST_NEWS] Limpieza de documentos antiguos completada.")


if __name__ == "__main__":
    main()
