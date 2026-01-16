# src/ingest/ingest_djia.py

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List

import yfinance as yf

from .chroma_client import col_prices as col  # 👈 colección explícita de precios
from .tickers import DJIA_TICKERS
from .embeddings import encode


def _build_docs_for_ticker(ticker: str, days: int = 30) -> List[Dict]:
    """
    Descarga las cotizaciones de los últimos `days` días para un ticker
    y las convierte en documentos listos para Chroma (colección de precios).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # auto_adjust se especifica explícitamente para evitar FutureWarning
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        return []

    docs: List[Dict] = []
    for date, row in df.iterrows():
        date_str = date.date().isoformat()          # "YYYY-MM-DD"
        date_num = int(date_str.replace("-", ""))   # 20251111

        open_ = float(row["Open"])
        close_ = float(row["Close"])
        high_ = float(row["High"])
        low_ = float(row["Low"])
        vol_ = float(row["Volume"])

        text = (
            f"Ticker: {ticker}\n"
            f"Fecha: {date_str}\n"
            f"Apertura: {open_}\n"
            f"Cierre: {close_}\n"
            f"Máximo: {high_}\n"
            f"Mínimo: {low_}\n"
            f"Volumen: {vol_}\n"
        )

        docs.append(
            {
                # 👇 ID determinista para evitar duplicados entre ejecuciones
                "id": f"{ticker}_{date_str}",
                "document": text,
                "metadata": {
                    "ticker": ticker,
                    "date": date_str,      # legible
                    "date_num": date_num,  # filtrable numéricamente
                    "source": "yfinance_prices",
                    "doc_type": "prices",
                },
            }
        )

    return docs


def ingest(tickers: Iterable[str], days: int = 30) -> int:
    """
    Ingesta en Chroma las cotizaciones de los tickers indicados.
    Usa la misma función de embeddings (encode) que el resto del sistema.

    Nota: si la colección soporta upsert, lo usamos para no duplicar.
    """
    collection = col

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metadatas: List[Dict] = []

    for ticker in tickers:
        ticker = ticker.upper().strip()
        if not ticker:
            continue

        docs = _build_docs_for_ticker(ticker, days=days)
        if not docs:
            continue

        all_ids.extend(d["id"] for d in docs)
        all_docs.extend(d["document"] for d in docs)
        all_metadatas.extend(d["metadata"] for d in docs)

    if not all_docs:
        return 0

    # Embeddings con tu modelo (misma dimensión que la colección)
    embeddings = encode(all_docs, is_query=False)

    # Preferimos upsert para que re-ingestas no dupliquen
    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )
    else:
        # Fallback: add (podría duplicar si no limpias antes)
        collection.add(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )

    return len(all_docs)


def cleanup(days: int = 30) -> None:
    """
    Elimina de Chroma los documentos con fecha anterior a hoy - `days`.
    Usa el campo numérico date_num (YYYYMMDD) para el filtro.
    """
    collection = col

    cutoff_date = (datetime.now(timezone.utc).date() - timedelta(days=days))
    cutoff_num = int(cutoff_date.strftime("%Y%m%d"))  # p.ej. 20251111

    collection.delete(
        where={
            "date_num": {"$lt": cutoff_num}
        }
    )


def main(argv=None) -> None:
    """
    Punto de entrada para usarlo como script:
    - Sin argumentos: ingesta de todo el DJIA
    - Con argumentos: lista de tickers concreta
    """
    argv = argv or sys.argv[1:]

    if argv:
        tickers = [t.strip().upper() for t in argv if t.strip()]
    else:
        tickers = list(DJIA_TICKERS)

    print(f"[INGEST] Tickers: {', '.join(tickers)}")
    n_docs = ingest(tickers, days=30)
    print(f"[INGEST] Documentos ingestados/actualizados: {n_docs}")

    cleanup(days=30)
    print("[INGEST] Limpieza de documentos antiguos completada.")


if __name__ == "__main__":
    main()
