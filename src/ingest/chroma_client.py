# src/ingest/chroma_client.py
from pathlib import Path
import chromadb

# /app es el WORKDIR típico en Docker
BASE_DIR = Path(__file__).resolve().parents[2]  # /app
CHROMA_PATH = BASE_DIR / "data" / "chroma_djia"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

col_prices = client.get_or_create_collection(
    name="djia_prices",
    metadata={"hnsw:space": "cosine"}
)

col_news = client.get_or_create_collection(
    name="djia_docs",
    metadata={"hnsw:space": "cosine"}
)
