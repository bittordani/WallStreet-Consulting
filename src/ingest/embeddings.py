from sentence_transformers import SentenceTransformer

# Cargamos el modelo 1 sola vez
_model = SentenceTransformer("intfloat/multilingual-e5-base")

def encode(texts, is_query=False):
    """
    Convierte una lista de textos a embeddings.
    is_query=True si el texto viene de una pregunta del usuario.
    """
    prefix = "query: " if is_query else "passage: "
    return _model.encode([prefix + t for t in texts], normalize_embeddings=True).tolist()
