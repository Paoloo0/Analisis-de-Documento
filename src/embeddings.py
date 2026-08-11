import os
from langchain_community.embeddings import FastEmbedEmbeddings

def get_embeddings(google_api_key=None):
    """
    Retorna el modelo de embeddings local y ultra-ligero FastEmbed.
    No consume API Keys ni cuota de Google, corre de forma 100% local y gratuita
    utilizando ONNX Runtime (sin PyTorch), evitando caídas de RAM (OOM) en la nube.
    Soporta español a través del modelo paraphrase-multilingual-MiniLM-L12-v2.
    """
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
