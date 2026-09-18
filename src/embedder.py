from sentence_transformers import SentenceTransformer

_model = None  


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def create_embeddings(texts):
    """Converts a list of text strings into a list of vectors (numbers)."""
    model = get_model()
    return model.encode(texts)
