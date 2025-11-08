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
```bash

WallStreet-Consulting/
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── ingest/
│   │   ├── ingest_djia.py
│   │   └── chroma_client.py
│   ├── rag/
│   │   ├── rag_query.py
│   ├── llm/
│   │   └── llm_client.py
│   └── __init__.py
│
├── scripts/
│   ├── health_check.py
│   ├── rag.py
│   └── run_djia.sh
│
├── data/                  ← **se sube solo la carpeta vacía**
│   └── (vacío)            ← se creará automáticamente al ejecutar
│
├── .env.example
├── .gitignore
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md

```

---

## ⚙️ Instalación

Para configurar el entorno y ejecutar el proyecto, sigue estos pasos:

  1️⃣ Clonar el repositorio:
  ```bash
  git clone git@github.com:bittordani/WallStreet-Consulting.git
  cd WallStreet-Consulting
  ```
  2️⃣ Crear y activar un entorno virtual:
  ```bash
  python -m venv .venv
  # En Windows:
  .\.venv\Scripts\activate
  # En Linux/macOS:
  source .venv/bin/activate
  ```
  3️⃣ Instalar las dependencias:
  ```bash
  pip install -r requirements.txt
  ```
  4️⃣ Crea tu archivo `.env` en la raíz del proyecto (usa el que tienes de ejemplo .env.example y renómbralo):
  ```bash
  # Apagar o encender el LLM
  USE_LLM=true
  
  # --- LLM (opcional) ---
  USE_LLM=false
  LLM_PROVIDER=google        # o: openai
  LLM_MODEL=gemini-2.5-flash # o: gpt-4o-mini
  GOOGLE_API_KEY=pon-tu-clave
  # OPENAI_API_KEY=tu_clave
  ```

---

## 🚀 Ejecutar en Local (sin Docker)
   
  Opción A: Ejecución desde Consola
  Usa tu script principal de consola (el que está en scripts/):
  ```bash
  ./scripts/rag.py "¿Cómo va Microsoft hoy?"
  ```
  Opción B: Ejecución de la API (FastAPI)
  Si ya adaptaste el código y tienes el archivo src/api/main.py, inicia el servidor Uvicorn:
  
  ```Bash
  # Ejecutar la aplicación FastAPI
  uvicorn src.api.main:app --reload
  ```
  Una vez que veas el mensaje de que Uvicorn está corriendo, tu API estará disponible en la dirección especificada.

  Ir a:
  👉 http://127.0.0.1:8000/docs
  
---


## 🐳 Ejecutar con Docker (recomendado)

1️⃣ Construir
```bash
docker compose build
```
2️⃣ Levantar
```bash
docker compose up -d
```
3️⃣ Probar
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Cómo va Microsoft hoy?"}'

```
---

## ✍️ Autor

Víctor Daniel Martínez

🔗 [LinkedIn](https://www.linkedin.com/in/victor-daniel-martinez-martinez/)
