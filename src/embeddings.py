import os
import time
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class RateLimitedEmbeddings(Embeddings):
    """
    Decorador para el modelo de embeddings de LangChain que fragmenta las solicitudes
    en lotes pequeños e inyecta un retardo controlado (y reintentos) para evitar
    el error 429 RESOURCE_EXHAUSTED del nivel gratuito de la API de Gemini (límite de 15 RPM).
    """
    def __init__(self, base_embeddings, batch_size=20, sleep_seconds=5):
        self.base_embeddings = base_embeddings
        self.batch_size = batch_size
        self.sleep_seconds = sleep_seconds

    def embed_documents(self, texts):
        all_embeddings = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_texts + self.batch_size - 1) // self.batch_size
            
            print(f"[*] Procesando lote de embeddings {batch_num}/{total_batches} (tamaño: {len(batch)})...")
            
            # Reintentos con tiempo de espera progresivo
            for attempt in range(5):
                try:
                    batch_embeds = self.base_embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeds)
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"[-] Error definitivo al procesar lote {batch_num}: {e}")
                        raise e
                    # Espera exponencial progresiva
                    wait_time = self.sleep_seconds * (attempt + 1) + 2
                    print(f"[!] Límite de cuota alcanzado en lote {batch_num}. Esperando {wait_time}s para reintentar...")
                    time.sleep(wait_time)
            
            # Retardo entre lotes sucesivos para no superar el límite de 15 RPM
            if i + self.batch_size < total_texts:
                time.sleep(self.sleep_seconds)
                
        return all_embeddings

    def embed_query(self, text):
        # Para consultas individuales no hay problema de límite de velocidad
        for attempt in range(3):
            try:
                return self.base_embeddings.embed_query(text)
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(3)

def get_embeddings(google_api_key):
    """
    Crea e inicializa el modelo de embeddings Gemini con limitador de tasa incorporado.
    """
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
        max_retries=10
    )
    return RateLimitedEmbeddings(base_embeddings)
