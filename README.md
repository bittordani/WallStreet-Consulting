# 🏦 WallStreet Consulting — Asistente Bursátil con RAG + FastAPI + Docker

WallStreet Consulting es un asistente financiero capaz de responder preguntas como:

> "¿Cómo va Microsoft hoy?"  
> "¿Qué cambio registró McDonalds ayer?"  

El sistema combina:

- **Descarga automática de datos bursátiles** (Yahoo Finance)
- **Almacenamiento en Base de Datos Vectorial** (ChromaDB)
- **Motor de Recuperación (RAG)** para encontrar el contexto correcto
- **Generación de respuesta natural** (con o sin LLM, configurable)
- **API REST** expuesta con FastAPI
- **Ejecución local o con Docker**

---

## 🧠 ¿Qué problema resuelve?

La información bursátil cambia cada día.  
Buscar manualmente datos históricos es lento y repetitivo.

Este asistente:
- Guarda automáticamente los últimos cierres
- Actualiza valores solo cuando es necesario
- Evita datos obsoletos
- Responde lenguaje natural

---

## 📂 Estructura del Proyecto
WallStreet-Consulting/
│
├── src/
│ ├── ingest/ # Scripts para descargar y actualizar datos
│ ├── rag/ # Recuperación + generación de respuesta
│ ├── llm/ # (Opcional) Conexión con modelos LLM
│ └── api/ # FastAPI (endpoints)
│
├── data/
│ └── chroma_djia/ # Base de datos vectorial persistente
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md



---

## ⚙️ Configuración

Crea tu archivo `.env` en la raíz del proyecto:

# Apagar o encender el LLM
USE_LLM=true

# --- LLM (opcional) ---
USE_LLM=false
LLM_PROVIDER=google        # o: openai
LLM_MODEL=gemini-2.5-flash # o: gpt-4o-mini
GOOGLE_API_KEY=pon-tu-clave
# OPENAI_API_KEY=tu_clave


---

## 🚀 Ejecutar en Local (sin Docker)

```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload

Ir a:
👉 http://127.0.0.1:8000/docs


🐳 Ejecutar con Docker (recomendado)
1️⃣ Construir
docker compose build

2️⃣ Levantar
docker compose up -d

3️⃣ Probar
curl "http://127.0.0.1:8000/ask?question=Como%20va%20Microsoft%20hoy"


🧩 Arquitectura

Usuario → FastAPI → RAG Query → ChromaDB → (Opcional) LLM → Respuesta natural
                  ↑
            Datos diarios (ingestión automática)


Cómo ejecutarlo (checklist diario)
    1. source .venv/bin/activate
    2. export PYTHONPATH=.
    3. (Opcional) python src/ingest/ingest_djia.py
    4. Probar:
       python - << 'PY'
       from src.rag.rag_query import ask
       print(ask("¿Cómo va Microsoft hoy?"))
       print(ask("¿Cómo va Visa hoy?"))
       print(ask("¿Cómo va McDonalds hoy?"))
       PY
