# src/llm/llm_client.py
from pathlib import Path
from textwrap import dedent
from dotenv import dotenv_values

def _cfg_fileonly():
    """
    Lee SIEMPRE el .env del repositorio (raíz) y NO usa os.environ.
    Así, cambiar .env cambia el comportamiento inmediatamente.
    """
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    vals = dotenv_values(dotenv_path=str(env_path))  # dict plano del .env (o {})
    use_llm  = (vals.get("USE_LLM", "false") or "false").strip().lower() == "true"
    provider = (vals.get("LLM_PROVIDER", "google") or "google").strip().lower()
    model    = (vals.get("LLM_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash").strip()
    g_key    = (vals.get("GOOGLE_API_KEY") or "").strip()
    oai_key  = (vals.get("OPENAI_API_KEY") or "").strip()
    return use_llm, provider, model, g_key, oai_key, str(env_path)

def _build_prompt(question: str, context: str) -> str:
    return dedent(f"""
    Eres un asistente financiero. Responde en español de forma clara y breve.
    Usa SOLO estos datos (no inventes):
    ---
    {(context or "").strip() or "(sin contexto)"}
    ---
    Pregunta: {(question or "").strip()}
    Da una respuesta directa (2-4 frases).
    """)

def _openai_answer(prompt: str, model: str, api_key: str) -> str:
    from openai import OpenAI
    # Pasamos la clave explícitamente para no depender de os.environ
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def _google_answer(prompt: str, model: str, api_key: str) -> str:
    import google.generativeai as genai
    if api_key:
        genai.configure(api_key=api_key)  # clave desde .env, no desde entorno
    ans = genai.GenerativeModel(model).generate_content(prompt)
    return (getattr(ans, "text", None) or "").strip()

def generate_answer(question: str, context: str) -> str:
    USE_LLM, PROVIDER, MODEL, G_KEY, OAI_KEY, env_path = _cfg_fileonly()

    if not USE_LLM:
        ctx = (context or "").strip()
        return "No hay contexto suficiente para responder." if not ctx else f"Resumen: {ctx.splitlines()[0]}"

    prompt = _build_prompt(question, context)
    if PROVIDER == "openai":
        return _openai_answer(prompt, MODEL, OAI_KEY)
    # por defecto: google
    return _google_answer(prompt, MODEL, G_KEY)
