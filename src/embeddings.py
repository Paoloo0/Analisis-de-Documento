import os
from langchain_community.embeddings import FastEmbedEmbeddings

def get_embeddings(google_api_key=None):
    """
    Retorna el modelo de embeddings local FastEmbed utilizando EXACTAMENTE el mismo
    modelo original del proyecto: 'sentence-transformers/all-MiniLM-L6-v2'.
    Esto mantiene intacto el comportamiento de búsqueda y precisión original, pero
    ejecutándose de forma ligera bajo ONNX Runtime para evitar caídas de RAM (OOM) en la nube.
    """
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
