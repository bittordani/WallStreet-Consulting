# src/ingest/ingest_djia.py
# Ingesta para sistema RAG del Dow Jones:
# - Métricas del día (precio, % de hoy) solo en sesión regular (NYSE)
# - Titulares con fallbacks (Ticker.news -> yf.get_news -> índice ^DJI)
# - Embeddings E5 + almacenamiento en Chroma
# Nota: yfinance suele llevar ~15 min de retraso.

import datetime as dt
import hashlib
from typing import List, Tuple, Iterable, Optional
import pytz
import yfinance as yf

from src.ingest.embeddings import encode
from src.ingest.chroma_client import col
from src.ingest.tickers import DJIA_TICKERS

NY_TZ = pytz.timezone("America/New_York")
TODAY = dt.date.today().isoformat()
TODAY_NUM = int(TODAY.replace("-", ""))  # p.ej. 20251031


# ----------------------- utilidades -----------------------
def make_id(text: str) -> str:
    """ID estable a partir de un texto (p.ej., título de noticia)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def chunked(seq: Iterable, n: int) -> Iterable[list]:
    """Divide en trozos de tamaño n."""
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ----------------------- métricas de hoy -----------------------
def make_metric_doc(ticker: str) -> Tuple[str, dict, str]:
    """
    % HOY = (último precio en sesión regular de HOY) vs (cierre de AYER).
    Se evita mezclar pre/after-market. Fallbacks si falta dato.
    """
    # 1) Cierre de AYER (diario)
    daily = yf.download(ticker, period="5d", interval="1d",
                        auto_adjust=False, progress=False)
    if daily.empty or "Close" not in daily or daily["Close"].dropna().shape[0] < 2:
        # Fallback: fast_info (menos preciso pero estable)
        t = yf.Ticker(ticker)
        info = t.fast_info
        last_price = float(info.get("last_price") or 0.0)
        prev_close = float(info.get("previous_close") or 0.0)
        pct = ((last_price - prev_close) / prev_close * 100.0) if prev_close else 0.0
        txt = (f"{ticker} hoy {TODAY}: cambio {pct:.2f}% "
               f"(último {last_price:.2f}, cierre previo {prev_close:.2f}, tick: N/A).")
        meta = {
            "ticker": ticker, "fecha": TODAY, "fecha_num": TODAY_NUM,
            "tipo": "metric", "tick_ts": "N/A", "fuente": "Yahoo fallback",
            "last": round(last_price, 2), "prev": round(prev_close, 2),
            "pct": round(pct, 2)
        }
        return txt, meta, f"metric-{ticker}-{TODAY}"

    prev_close = daily["Close"].dropna().iloc[-2].item()

    # 2) Intradía (1m) filtrando SOLO sesión regular (09:30–16:00 NY)
    intraday = yf.download(ticker, period="2d", interval="1m",
                           auto_adjust=False, progress=False)
    if intraday.empty or "Close" not in intraday:
        # Fallback: usar último cierre diario (sin tick_ts)
        last_price = daily["Close"].dropna().iloc[-1].item()
        pct = ((last_price - prev_close) / prev_close * 100.0) if prev_close else 0.0
        txt = (f"{ticker} hoy {TODAY}: cambio {pct:.2f}% "
               f"(último {last_price:.2f}, cierre previo {prev_close:.2f}, tick: N/A).")
        meta = {
            "ticker": ticker, "fecha": TODAY, "fecha_num": TODAY_NUM,
            "tipo": "metric", "tick_ts": "N/A", "fuente": "Yahoo daily",
            "last": round(last_price, 2), "prev": round(prev_close, 2),
            "pct": round(pct, 2)
        }
        return txt, meta, f"metric-{ticker}-{TODAY}"

    intraday = intraday.tz_convert(NY_TZ).between_time("09:30", "16:00")
    today = dt.datetime.now(NY_TZ).date()
    intraday_today = intraday[intraday.index.date == today]

    if intraday_today.empty:
        # Mercado cerrado ahora mismo → toma el último punto disponible del intradía
        last_price = intraday["Close"].dropna().iloc[-1].item()
        tick_ts = intraday.index.max().strftime("%Y-%m-%d %H:%M %Z")
    else:
        last_price = intraday_today["Close"].dropna().iloc[-1].item()
        tick_ts = intraday_today.index.max().strftime("%Y-%m-%d %H:%M %Z")

    pct = ((last_price - prev_close) / prev_close * 100.0) if prev_close else 0.0

    txt = (f"{ticker} hoy {TODAY}: cambio {pct:.2f}% "
           f"(último {last_price:.2f}, cierre previo {prev_close:.2f}, tick: {tick_ts}).")
    meta = {
        "ticker": ticker, "fecha": TODAY, "fecha_num": TODAY_NUM,
        "tipo": "metric", "tick_ts": tick_ts, "fuente": "Yahoo 1m regular-hours",
        "last": round(last_price, 2), "prev": round(prev_close, 2),
        "pct": round(pct, 2)
    }
    return txt, meta, f"metric-{ticker}-{TODAY}"


# ----------------------- titulares (con fallbacks) -----------------------
def make_news_docs(ticker: str, max_items: int = 5) -> List[Tuple[str, dict, str]]:
    """
    Prioridad: Ticker(ticker).news -> yfinance.get_news(ticker) -> Ticker('^DJI').news.
    - Dedup por título
    - Filtra ruido (títulos muy cortos, p.ej. ".")
    """
    docs: List[Tuple[str, dict, str]] = []
    seen = set()

    def add_items(items):
        for item in (items or []):
            title = (item or {}).get("title") or ""
            title = title.strip()
            if len(title) < 6:  # evita ".", "Up", etc.
                continue
            if title in seen:
                continue
            seen.add(title)
            publisher = (item or {}).get("publisher") or ""
            link = (item or {}).get("link") or ""
            txt = f"Titular {ticker}: {title} (fuente: {publisher})."
            meta = {
                "ticker": ticker, "fecha": TODAY, "fecha_num": TODAY_NUM,
                "tipo": "headline", "url": link
            }
            doc_id = f"news-{ticker}-{make_id(title)}"
            docs.append((txt, meta, doc_id))

    # 1) Titulares directos del ticker
    try:
        add_items(getattr(yf.Ticker(ticker), "news", [])[:max_items])
    except Exception:
        pass

    # 2) API global get_news(ticker)
    if len(docs) < max_items:
        try:
            get_news = getattr(yf, "get_news", None)
            if callable(get_news):
                add_items(get_news(ticker)[: max_items - len(docs)])
        except Exception:
            pass

    # 3) Fallback del índice ^DJI
    if len(docs) < max_items:
        try:
            add_items(getattr(yf.Ticker("^DJI"), "news", [])[: max_items - len(docs)])
        except Exception:
            pass

    return docs[:max_items]


# ----------------------- operaciones BD -----------------------
def upsert_docs(payload: List[Tuple[str, dict, str]], batch_size: int = 64) -> None:
    """Inserta/actualiza en Chroma en lotes."""
    if not payload:
        return
    for part in chunked(payload, batch_size):
        docs = [p[0] for p in part]
        metas = [p[1] for p in part]
        ids = [p[2] for p in part]
        # Embeddings E5:
        embs = encode(docs, is_query=False)
        col.upsert(documents=docs, embeddings=embs, metadatas=metas, ids=ids)


def cleanup(days: int = 7) -> None:
    """Borra documentos con fecha_num anterior al umbral."""
    cutoff_num = int((dt.date.today() - dt.timedelta(days=days)).strftime("%Y%m%d"))
    # ⭐ forma recomendada: borrar por filtro directamente
    try:
        col.delete(where={"fecha_num": {"$lt": cutoff_num}})
        return
    except Exception:
        # Fallback: obtener ids sin include y borrar por ids
        res = col.get(where={"fecha_num": {"$lt": cutoff_num}}, include=[])
        ids_nested = res.get("ids") or []
        flat_ids = [i for group in ids_nested for i in group]
        if flat_ids:
            col.delete(ids=flat_ids)


# ----------------------- main -----------------------
def ingest(tickers: Iterable[str], max_news: int = 5) -> int:
    batch = []
    for ticker in tickers:
        try:
            metric_txt, metric_meta, metric_id = make_metric_doc(ticker)
            batch.append((metric_txt, metric_meta, metric_id))
        except Exception as e:
            print(f"[WARN] Métrica fallida {ticker}: {e}")

        try:
            batch.extend(make_news_docs(ticker, max_items=max_news))
        except Exception as e:
            print(f"[WARN] Noticias fallidas {ticker}: {e}")

    upsert_docs(batch)
    print(f"Ingestados {len(batch)} documentos para {len(list(tickers))} tickers en {TODAY}")
    return len(batch)


def main(argv: Optional[list] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingesta RAG DJIA (métricas + titulares)")
    parser.add_argument("--tickers", type=str, default=",".join(DJIA_TICKERS),
                        help="Lista separada por comas (por defecto DJIA completo)")
    parser.add_argument("--max-news", type=int, default=5, help="Titulares por ticker (def=5)")
    parser.add_argument("--days-keep", type=int, default=7, help="Días a conservar (def=7)")
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    ingest(tickers, max_news=args.max_news)
    cleanup(days=args.days_keep)


if __name__ == "__main__":
    main()
