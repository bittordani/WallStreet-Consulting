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


def _build_prompt(question: str, context: str) -> str:
    return dedent(f"""
    Eres un analista bursátil que trabaja con un sistema RAG conectado
    a una base de datos vectorial de cotizaciones históricas del índice Dow Jones.

    Responde en español de España en 1 solo párrafo (4–6 frases).
    Usa SOLO los datos del CONTEXTO (proceden de la base de datos vectorial; no inventes nada).
    No des recomendaciones de compra/venta.

    Obligatorio en la respuesta:
    - Empieza explicando cómo está el valor en la FECHA más reciente del contexto
      (la que aparezca primero o tenga la fecha más alta).
    - Después, si ayuda, compara brevemente con 1–2 días anteriores del contexto.
    - Menciona la FECHA o rango de fechas que aparezcan en el contexto.
    - Menciona porcentajes (%) y precios (por ejemplo, cierres o variaciones) si están disponibles.
    - Si falta algún dato, dilo (“no dispongo de X en el contexto”).

    CONTEXTO (fragmentos recuperados de la BBDD vectorial del Dow Jones):
    ---
    {context.strip() or "(sin contexto)"}
    ---

    Pregunta del usuario: {question.strip()}
    Respuesta (un solo párrafo):
    """).strip()





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


def _format_free_answer(question: str, context: str) -> str:
    # “modo gratis”: no llama a ninguna API, pero responde bien formado
    ctx = (context or "").strip()
    if not ctx:
        return "No hay contexto suficiente para responder."
    # Si hay varias líneas (métrica + titulares), usa las 2 primeras como resumen
    lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
    if not lines:
        return "No hay contexto suficiente para responder."
    if len(lines) == 1:
        return lines[0]
    return f"{lines[0]} {lines[1]}"


def generate_answer(question: str, context: str, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Devuelve dict para API:
      { "answer": "...", "sources": [...], "use_llm": bool, "provider": str, "model": str }
    """
    if not USE_LLM:
        answer = _format_free_answer(question, context)
    else:
        prompt = _build_prompt(question, context)
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
    }

    # Si quieres ver fuentes también en el texto (opcional)
    if SHOW_SOURCES and out["sources"]:
        # no lo metas dentro del párrafo del LLM, lo devuelves como metadato en JSON
        pass

    return out
