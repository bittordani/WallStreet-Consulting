# src/llm/llm_client.py
from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional


def _load_env() -> Optional[Path]:
    """
    Carga .env buscando hacia arriba desde:
    - CWD (recomendado: raíz del repo)
    - carpeta src
    Devuelve la ruta del .env cargado (o None).
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return None

    candidates: List[Path] = []

    # 1) desde el directorio actual hacia arriba
    p = Path.cwd()
    for _ in range(8):
        candidates.append(p / ".env")
        p = p.parent

    # 2) desde la carpeta de este archivo hacia arriba
    p = Path(__file__).resolve().parent
    for _ in range(8):
        candidates.append(p / ".env")
        p = p.parent

    for c in candidates:
        if c.exists() and c.is_file():
            load_dotenv(dotenv_path=str(c), override=True)
            return c
    return None


_ENV_PATH = _load_env()
if _ENV_PATH:
    print(f"[ENV] .env cargado desde: {_ENV_PATH}")


USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()  # google | openai
MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "true").lower() == "true"


# -----------------------------
# Prompts
# -----------------------------
def _build_prompt_docs(question: str, context: str) -> str:
    """
    Prompt para RAG REAL sobre texto no estructurado (noticias, filings, comunicados, etc.).
    Espera que el contexto venga con etiquetas [S1], [S2]... para poder citar.
    """
    return dedent(f"""
    Eres un asistente RAG para análisis de información financiera basada en DOCUMENTOS
    (noticias, comunicados, transcripciones, filings). Responde en español de España.

    REGLAS:
    - Usa SOLO el CONTEXTO recuperado (no inventes).
    - Si el contexto no contiene evidencia suficiente, responde exactamente:
      "No disponible con la evidencia actual." y pide al usuario que acote (empresa, fecha, tema).
    - No des recomendaciones de compra/venta.
    - Responde en 1 solo párrafo (4–6 frases).
    - Añade 1–3 citas al final del párrafo usando el formato [S1], [S2]... (según corresponda).

    CONTEXTO (fragmentos recuperados del vector DB):
    ---
    {context.strip() or "(sin contexto)"}
    ---

    Pregunta del usuario: {question.strip()}
    Respuesta (un solo párrafo, con citas [S#]):
    """).strip()


def _build_prompt_prices(question: str, context: str) -> str:
    """
    Prompt para explicar datos estructurados (precios/métricas) cuando hayas recuperado
    un pequeño contexto numérico (por ejemplo OHLCV). No es el “RAG de documentos”, es
    un modo diferente para métricas.
    """
    return dedent(f"""
    Eres un asistente que explica métricas bursátiles a partir de datos ESTRUCTURADOS
    (precios históricos). Responde en español de España.

    REGLAS:
    - Usa SOLO el CONTEXTO (no inventes).
    - Si faltan datos, dilo (“no dispongo de X en el contexto”).
    - No des recomendaciones de compra/venta.
    - Responde en 1 párrafo (3–5 frases).
    - Indica claramente la fecha del último dato disponible.
    - Si el usuario pregunta "hoy" y el contexto no incluye hoy, aclara cuál es el último día disponible.

    CONTEXTO:
    ---
    {context.strip() or "(sin contexto)"}
    ---

    Pregunta del usuario: {question.strip()}
    Respuesta (un solo párrafo):
    """).strip()


# -----------------------------
# Providers
# -----------------------------
def _openai_answer(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _google_answer(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


# -----------------------------
# Fallback sin LLM
# -----------------------------
def _format_free_answer(question: str, context: str) -> str:
    # “modo gratis”: no llama a ninguna API, pero responde bien formado
    ctx = (context or "").strip()
    if not ctx:
        return "No disponible con la evidencia actual."
    lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
    if not lines:
        return "No disponible con la evidencia actual."
    if len(lines) == 1:
        return lines[0]
    # mini resumen
    return f"{lines[0]} {lines[1]}"


# -----------------------------
# API
# -----------------------------
def generate_answer(
    question: str,
    context: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    mode: str = "docs",  # "docs" | "prices"
) -> Dict[str, Any]:
    """
    Devuelve dict para API:
      { "answer": "...", "sources": [...], "use_llm": bool, "provider": str, "model": str, "mode": str }

    mode:
      - "docs": RAG sobre texto no estructurado con citas [S#]
      - "prices": explicación de métricas estructuradas (precios)
    """
    ctx = (context or "").strip()

    # Control anti-alucinación (clave para evaluación)
    if not ctx:
        return {
            "answer": "No disponible con la evidencia actual.",
            "sources": sources or [],
            "use_llm": USE_LLM,
            "provider": PROVIDER,
            "model": MODEL,
            "mode": mode,
        }

    if not USE_LLM:
        answer = _format_free_answer(question, ctx)
    else:
        prompt = _build_prompt_docs(question, ctx) if mode == "docs" else _build_prompt_prices(question, ctx)
        if PROVIDER == "google":
            answer = _google_answer(prompt)
        else:
            answer = _openai_answer(prompt)

    out = {
        "answer": answer,
        "sources": sources or [],
        "use_llm": USE_LLM,
        "provider": PROVIDER,
        "model": MODEL,
        "mode": mode,
    }

    # SHOW_SOURCES: mantenemos las fuentes en JSON, no pegadas al párrafo.
    if SHOW_SOURCES and out["sources"]:
        pass

    return out
