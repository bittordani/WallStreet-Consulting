# Imagen base ligera
FROM python:3.12-slim AS base

# Evita prompts y acelera pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Paquetes del sistema (compilación mínima)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de la app
WORKDIR /app

# Copiamos solo requirements primero (para cache)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Puerto FastAPI
EXPOSE 8000

# Asegura que la carpeta de la VDB exista dentro del contenedor
RUN mkdir -p /app/chroma

# Comando por defecto: arrancar la API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
