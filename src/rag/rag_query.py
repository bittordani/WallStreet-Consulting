# src/rag/rag_query.py
# RAG financiero (DJIA) con VDB-primero + auto-upsert.

import json
import datetime as dt
from pathlib import Path
import yfinance as yf

from src.ingest.chroma_client import col
from src.ingest.embeddings import encode
from src.llm.llm_client import generate_answer


def _today_str() -> str:
    return dt.date.today().isoformat()

# --- NUEVO: util para ayer ---
from datetime import date, timedelta

def _yesterday_str() -> str:
    return (date.today() - timedelta(days=1)).isoformat()

# --- NUEVO: buscar métrica de un día concreto en la VDB ---
def _metric_from_chroma_day(tk: str, day: str):
    res = col.get(
        where={"$and": [{"ticker": tk}, {"tipo": "metric"}, {"fecha": day}]},
        include=["metadatas", "documents"],
    )
    metas = (res.get("metadatas") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    if not metas or not docs:
        return None
    # si hay varias, coge la más reciente por fecha_num
    pairs = sorted(
        zip(metas, docs),
        key=lambda x: _safe_meta(x[0]).get("fecha_num", 0),
        reverse=True,
    )
    return pairs[0][1]


# --- NUEVO: calcular % para AYER (solo diario, robusto) ---
def _metric_yesterday_yf(ticker: str):
    import yfinance as yf
    df = (
        yf.Ticker(ticker)
        .history(period="10d", interval="1d", auto_adjust=False)
        .dropna(subset=["Close"])
    )
    if len(df) < 3:
        return None, None
    # ayer y el día previo
    prev  = df["Close"].iloc[-3].item()
    last  = df["Close"].iloc[-2].item()
    day   = df.index[-2].date().isoformat()
    pct   = ((last - prev) / prev * 100.0) if prev else 0.0
    txt   = f"{ticker} (cierre {day}): cambio {pct:.2f}% (último {last:.2f}, cierre previo {prev:.2f})."
    return txt, day

# FIN ayer

def _safe_meta(meta):
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}

# ----------- mapping nombre->ticker (básico) -----------
try:
    TICKERS = [
        l.strip() for l in Path("data/djia_tickers.txt")
        .read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
except Exception:
    TICKERS = ["AAPL","MSFT","BA","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON",
               "IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","NKE","PG","TRV",
               "UNH","V","VZ","WMT","AMZN","CAT","AXP","AMGN"]

NAME2TICKER = {
    "apple":"AAPL","manzana":"AAPL","microsoft":"MSFT","boeing":"BA","coca-cola":"KO","coca cola":"KO",
    "visa":"V","nike":"NKE","ibm":"IBM","intel":"INTC","walmart":"WMT",
    "mcdonald":"MCD","mcdonalds":"MCD","mcdonald's":"MCD","macdonald":"MCD","macdonals":"MCD",
    "disney":"DIS","chevron":"CVX","salesforce":"CRM","jp morgan":"JPM","jpmorgan":"JPM",
    "goldman":"GS","johnson & johnson":"JNJ","johnson":"JNJ","merck":"MRK","3m":"MMM",
    "procter & gamble":"PG","procter":"PG","travellers":"TRV","unitedhealth":"UNH","amgen":"AMGN",
    "american express":"AXP","amex":"AXP","caterpillar":"CAT","home depot":"HD","honeywell":"HON",
    "dow":"DOW","verizon":"VZ","amazon":"AMZN","tesla":"TSLA"
}

def _detect_ticker(q: str):
    ql = (q or "").lower()
    for name, tk in NAME2TICKER.items():
        if name in ql:
            return tk
    for tok in (q or "").split():
        t = tok.strip(",.?!()")
        if t in TICKERS or (t.isupper() and 1 <= len(t) <= 5):
            return t
    return None

def _where(**conds):
    items = [{k: v} for k, v in conds.items() if v is not None]
    if not items:
        return None
    return items[0] if len(items) == 1 else {"$and": items}

# ----------- VDB helpers -----------
'''def _latest_metric_from_chroma(tk: str):
    res = col.get(
        where=_where(ticker=tk, tipo="metric"),
        include=["metadatas", "documents"],
    )
    metas = (res.get("metadatas") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    if not metas or not docs:
        return None
    pairs = sorted(
        zip(metas, docs),
        key=lambda x: _safe_meta(x[0]).get("fecha_num", 0),
        reverse=True
    )
    return pairs[0][1]  # devuelve el texto
'''
def _latest_metric_from_chroma(tk: str):
    res = col.get(
        where={"$and": [{"ticker": tk}, {"tipo": "metric"}]},
        include=["metadatas", "documents"],
    )
    metas = (res.get("metadatas") or [[]])[0]
    docs  = (res.get("documents") or [[]])[0]
    if not metas or not docs:
        return None

    pairs = [
        (m, d) for m, d in zip(metas, docs)
        if isinstance(d, str) and len(d.strip()) > 3
    ]
    if not pairs:
        return None

    pairs.sort(key=lambda x: _safe_meta(x[0]).get("fecha_num", 0), reverse=True)
    return pairs[0][1]


def _upsert_metric(doc_txt: str, tk: str, day: str):
    meta = {
        "ticker": tk,
        "fecha": day,
        "fecha_num": int(day.replace("-", "")),
        "tipo": "metric"
    }
    emb = encode([doc_txt], is_query=False)
    col.upsert(
        documents=[doc_txt],   # <<--- LISTA (clave del bug)
        embeddings=emb,
        metadatas=[meta],
        ids=[f"metric-{tk}-{day}"]
    )

# ----------- cálculo estable con yfinance (diario) -----------
def _metric_daily_yf(ticker: str):
    df = (yf.Ticker(ticker)
            .history(period="5d", interval="1d", auto_adjust=False)
            .dropna(subset=["Close"]))
    if len(df) < 2:
        return None, None
    prev = df["Close"].iloc[-2].item()
    last = df["Close"].iloc[-1].item()
    pct  = ((last - prev) / prev * 100.0) if prev else 0.0
    day  = df.index[-1].date().isoformat()
    txt  = f"{ticker} (cierre {day}): cambio {pct:.2f}% (último {last:.2f}, cierre previo {prev:.2f})."
    return txt, day


def _daily_history(ticker: str, days: int = 15):
    return (
        yf.Ticker(ticker)
        .history(period=f"{days}d", interval="1d", auto_adjust=False)
        .dropna(subset=["Close"])
    )

def _metric_for_today_logic(ticker: str):
    """
    Devuelve (texto, day_iso) para 'hoy':
    - Si hoy tiene vela diaria -> usa hoy.
    - Si hoy no hay sesión -> usa último día bursátil disponible.
    """
    df = _daily_history(ticker)
    if len(df) < 2:
        return None, None

    today = dt.date.today()
    last_day = df.index[-1].date()

    # Elegimos el día de referencia para 'hoy'
    ref_idx = -1  # por defecto, último día bursátil
    if last_day == today:
        # Hoy sí tiene vela diaria
        ref_idx = -1
        prev_idx = -2
        note = ""  # sesión de hoy
    else:
        # Hoy no hay sesión: usamos último día bursátil
        ref_idx = -1
        prev_idx = -2
        note = " (no hubo sesión hoy; se muestra el último día bursátil)"

    last = df["Close"].iloc[ref_idx].item()
    prev = df["Close"].iloc[prev_idx].item() if len(df) >= (abs(prev_idx)) else None
    if prev is None or prev == 0:
        return None, None

    day_iso = df.index[ref_idx].date().isoformat()
    pct = (last - prev) / prev * 100.0
    txt = f"{ticker} (cierre {day_iso}): cambio {pct:.2f}% (último {last:.2f}, cierre previo {prev:.2f}).{note}"
    return txt, day_iso

def _metric_for_yesterday_logic(ticker: str):
    """
    Devuelve (texto, day_iso) para 'ayer':
    - Si hoy hay sesión -> usa el día bursátil anterior (penúltima vela).
    - Si hoy no hay sesión -> 'ayer' = último día bursátil disponible (última vela).
    """
    df = _daily_history(ticker)
    if len(df) < 2:
        return None, None

    today = dt.date.today()
    last_day = df.index[-1].date()

    if last_day == today:
        # Hoy sí cotiza: 'ayer' = día bursátil anterior
        ref_idx = -2
        prev_idx = -3 if len(df) >= 3 else None
        note = ""
    else:
        # Hoy NO cotiza (sábado/festivo): 'ayer' = último día bursátil
        ref_idx = -1
        prev_idx = -2
        note = ""

    if prev_idx is None or abs(prev_idx) > len(df):
        return None, None

    last = df["Close"].iloc[ref_idx].item()
    prev = df["Close"].iloc[prev_idx].item()
    if prev == 0:
        return None, None

    day_iso = df.index[ref_idx].date().isoformat()
    pct = (last - prev) / prev * 100.0
    txt = f"{ticker} (cierre {day_iso}): cambio {pct:.2f}% (último {last:.2f}, cierre previo {prev:.2f}).{note}"
    return txt, day_iso




# ----------- API principal -----------
def ask(question: str) -> str:
    '''

    ql = (question or "").lower()
    tk = _detect_ticker(question)
    if not tk:
        return "Indica el valor (ej.: «¿Cómo va Microsoft hoy?»)."

    if "ayer" in ql:
        day = _yesterday_str()
        doc = _metric_from_chroma_day(tk, day)
        if isinstance(doc, str) and len(doc.strip()) > 3:
            return doc
        doc, day = _metric_yesterday_yf(tk)
        if doc and day:
            _upsert_metric(doc, tk, day)
            return doc
        return f"No puedo obtener la cotización de {tk} para ayer."

    doc = _latest_metric_from_chroma(tk)
    if isinstance(doc, str) and len(doc.strip()) > 3:
        return doc

    doc, day = _metric_daily_yf(tk)
    if doc and day:
        _upsert_metric(doc, tk, day)
        return doc

    return f"No puedo obtener la cotización de {tk} ahora mismo."



    ql = (question or "").lower()
    tk = _detect_ticker(question)
    if not tk:
        return "Indica el valor (ej.: «¿Cómo va Microsoft hoy?»)."

    # AYER usando último día hábil anterior al último cierre disponible
    if "ayer" in ql:
        import yfinance as yf

        # Ultimos cierres diarios (ya ajusta fines de semana y festivos solo).
        df = (
            yf.Ticker(tk)
            .history(period="7d", interval="1d", auto_adjust=False)
            .dropna(subset=["Close"])
        )

        # Necesitamos al menos 2 días hábiles
        if len(df) < 2:
            return f"No puedo obtener la cotización de {tk} para ayer."

        # -1 = último día hábil (para "hoy" en fin de semana)
        # -2 = día hábil anterior → esto es lo que queremos para "ayer"
        prev_row = df.iloc[-2]
        curr_row = df.iloc[-1]

        # Calculamos variación respecto al día anterior a prev_row
        if len(df) >= 3:
            prev_close = df["Close"].iloc[-3]
        else:
            prev_close = prev_row["Close"]

        last = prev_row["Close"]
        pct = ((last - prev_close) / prev_close * 100.0) if prev_close else 0.0
        day_str = prev_row.name.date().isoformat()

        txt = (
            f"{tk} (cierre {day_str}): cambio {pct:.2f}% "
            f"(último {last:.2f}, cierre previo {prev_close:.2f})."
        )

        # Guardamos en VDB para futura consulta
        _upsert_metric(txt, tk, day_str)

        return generate_answer(question, txt)


    # HOY / genérico
    doc = _latest_metric_from_chroma(tk)
    if doc:
        return generate_answer(question, doc)

    doc, day = _metric_daily_yf(tk)
    if doc and day:
        _upsert_metric(doc, tk, day)
        return generate_answer(question, doc)

    return f"No puedo obtener la cotización de {tk} ahora mismo."


def latest_metric(ticker: str) -> str:
    """Devuelve la última métrica almacenada para un ticker."""
    res = col.get(
        where={"$and":[{"ticker":ticker},{"tipo":"metric"}]},
        include=["metadatas","documents"]
    )
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    if not docs:
        return "(sin datos)"
    # Ordenar por fecha
    paired = sorted(zip(metas, docs), key=lambda x: x[0].get("fecha_num", 0), reverse=True)
    return paired[0][1]
'''

    ql = (question or "").lower()
    tk = _detect_ticker(question)
    if not tk:
        return "Indica el valor (ej.: «¿Cómo va Microsoft hoy?»)."

    # -------------------- AYER --------------------
    if "ayer" in ql:
        doc, day = _metric_for_yesterday_logic(tk)
        if doc and day:
            _upsert_metric(doc, tk, day)
            return generate_answer(question, doc)
        return f"No puedo obtener la cotización de {tk} para ayer."

    # -------------------- HOY --------------------
    if "hoy" in ql:
        doc, day = _metric_for_today_logic(tk)
        if doc and day:
            _upsert_metric(doc, tk, day)
            return generate_answer(question, doc)
        return f"No puedo obtener la cotización de {tk} ahora mismo."

    # -------------------- genérico (último disponible) --------------------
    doc = _latest_metric_from_chroma(tk)
    if isinstance(doc, str) and len(doc.strip()) > 3:
        return generate_answer(question, doc)

    # Si no hay en VDB, calculamos último disponible (como 'hoy')
    doc, day = _metric_for_today_logic(tk)
    if doc and day:
        _upsert_metric(doc, tk, day)
        return generate_answer(question, doc)

    return f"No puedo obtener la cotización de {tk} ahora mismo."
