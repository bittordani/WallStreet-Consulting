import chromadb

# Creamos o reutilizamos una base de datos en la carpeta data/
client = chromadb.PersistentClient(path="data/chroma_djia")

# Creamos una colección llamada "djia" donde guardaremos los vectores
col = client.get_or_create_collection(
    name="djia",
    metadata={"hnsw:space": "cosine"}  # tipo de similitud recomendado con e5
)
