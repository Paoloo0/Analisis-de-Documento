import os
import time
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.quota import get_quota, use_quota, set_quota_to_zero

class SafeGeminiEmbeddings(Embeddings):
    """
    Wrapper de seguridad para GoogleGenerativeAIEmbeddings que maneja de forma manual
    el fraccionar las solicitudes en lotes y reintenta activamente con esperas controladas
    (sleep) si se alcanza el límite de cuota (error 429 / RESOURCE_EXHAUSTED).
    Evita caídas de RAM (OOM) al no cargar modelos locales y elude bloqueos por cuota.
    """
    def __init__(self, google_api_key, batch_size=15, sleep_seconds=4):
        self.base = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        )
        self.google_api_key = google_api_key
        self.batch_size = batch_size
        self.sleep_seconds = sleep_seconds

    def embed_documents(self, texts):
        all_embeddings = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, self.batch_size):
            # Verificar si se agotaron los tokens de la API
            if get_quota(self.google_api_key)["remaining"] <= 0:
                raise ValueError("Se agotaron los tokens de la api")
                
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_texts + self.batch_size - 1) // self.batch_size
            
            print(f"[*] Enviando lote de embeddings {batch_num}/{total_batches} (tamaño: {len(batch)}) a Gemini...")
            
            # Reintentos activos con backoff exponencial progresivo si hay 429
            for attempt in range(6):
                try:
                    batch_embeds = self.base.embed_documents(batch)
                    all_embeddings.extend(batch_embeds)
                    use_quota(1, self.google_api_key)  # Restar 1 intento tras éxito
                    break
                except Exception as e:
                    error_msg = str(e)
                    # Si es error de cuota (429) o agotado (RESOURCE_EXHAUSTED)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        if attempt == 5:
                            print(f"[-] Error definitivo por cuota en lote {batch_num}: {e}")
                            set_quota_to_zero(self.google_api_key)  # Ajustar cuota a 0 inmediatamente
                            raise e
                        wait_time = 15 * (attempt + 1)
                        print(f"[!] Límite de cuota Gemini alcanzado (429). Detalles: {error_msg}. Esperando {wait_time}s para reintentar...")
                        time.sleep(wait_time)
                    else:
                        if attempt == 5:
                            print(f"[-] Error definitivo en lote {batch_num}: {e}")
                            raise e
                        time.sleep(5)
            
            # Retardo preventivo entre lotes sucesivos para no saturar la API
            if i + self.batch_size < total_texts:
                time.sleep(self.sleep_seconds)
                
        return all_embeddings

    def embed_query(self, text):
        # Verificar si se agotaron los tokens de la API
        if get_quota(self.google_api_key)["remaining"] <= 0:
            raise ValueError("Se agotaron los tokens de la api")
            
        # Para consultas individuales no hay problema de límite de velocidad,
        # pero reintentamos de forma segura si la API está muy saturada.
        for attempt in range(5):
            try:
                res = self.base.embed_query(text)
                use_quota(1, self.google_api_key)  # Restar 1 intento tras éxito
                return res
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt == 4:
                        set_quota_to_zero(self.google_api_key)  # Ajustar cuota a 0 inmediatamente
                        raise e
                    time.sleep(10)
                else:
                    if attempt == 4:
                        raise e
                    time.sleep(3)

def get_embeddings(google_api_key):
    """
    Retorna el modelo de embeddings Gemini con nuestro wrapper de seguridad incorporado.
    """
    return SafeGeminiEmbeddings(google_api_key)
