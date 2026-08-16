import os
import json
import hashlib
from pathlib import Path

# Directorio base y de datos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QUOTA_FILE = DATA_DIR / "api_quota.json"

def get_key_hash(api_key: str) -> str:
    """
    Retorna el hash SHA-256 de la clave API para guardarlo de forma anónima y segura.
    """
    if not api_key:
        return "default"
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()

def get_active_api_key() -> str:
    """
    Intenta obtener la clave API actual del entorno.
    """
    try:
        from src.config import GOOGLE_API_KEY
        return GOOGLE_API_KEY if GOOGLE_API_KEY else ""
    except ImportError:
        return ""

def get_quota(api_key: str = None):
    """
    Retorna la cuota actual para la clave especificada o activa.
    Estructura del archivo:
    {
       "keys": {
          "hash": {"limit": 1000, "remaining": 1000}
       }
    }
    """
    if not api_key:
        api_key = get_active_api_key()
    
    key_hash = get_key_hash(api_key)
    default_quota = {"limit": 1000, "remaining": 1000}
    
    if not QUOTA_FILE.exists():
        return default_quota
        
    try:
        with open(QUOTA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "keys" in data:
                keys_data = data["keys"]
                if key_hash in keys_data:
                    return keys_data[key_hash]
    except Exception:
        pass
        
    return default_quota

def set_quota(limit: int, remaining: int, api_key: str = None):
    """
    Guarda la cuota para la clave especificada o activa en disco.
    """
    if not api_key:
        api_key = get_active_api_key()
        
    key_hash = get_key_hash(api_key)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Leer datos existentes
    quota_data = {"keys": {}}
    if QUOTA_FILE.exists():
        try:
            with open(QUOTA_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, dict) and "keys" in existing:
                    quota_data = existing
        except Exception:
            pass
            
    # Actualizar para esta clave
    quota_data["keys"][key_hash] = {"limit": limit, "remaining": remaining}
    
    try:
        with open(QUOTA_FILE, "w", encoding="utf-8") as f:
            json.dump(quota_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Error al guardar cuota en disco: {e}")

def use_quota(amount: int = 1, api_key: str = None):
    """
    Resta una cantidad específica de la cuota restante de la clave activa.
    """
    if not api_key:
        api_key = get_active_api_key()
        
    quota = get_quota(api_key)
    new_remaining = max(0, quota["remaining"] - amount)
    set_quota(quota["limit"], new_remaining, api_key)
    return new_remaining

def set_quota_to_zero(api_key: str = None):
    """
    Fuerza la cuota restante de la clave activa a 0 (por ejemplo, ante un error 429 real).
    """
    if not api_key:
        api_key = get_active_api_key()
    quota = get_quota(api_key)
    set_quota(quota["limit"], 0, api_key)

def test_and_initialize_quota(api_key: str):
    """
    Analiza la clave API de Gemini realizando una llamada mínima de embeddings.
    - Si tiene éxito, establece la cuota siempre a 1000/1000 (para que el usuario sepa que tiene 1000 intentos).
    - Si falla con 429 (RESOURCE_EXHAUSTED), es válida pero está agotada; establece cuota a 0/1000.
    - Si falla por otra causa, lanza una excepción (Clave API inválida).
    """
    from google import genai
    
    try:
        client = genai.Client(api_key=api_key)
        # Intentar llamada de prueba con el nuevo SDK de google-genai
        client.models.embed_content(
            model="models/gemini-embedding-001",
            contents="test_quota_check"
        )
        
        # Guardar como 1000/1000
        set_quota(1000, 1000, api_key)
        return True, 1000, "Clave API válida y activa. Cuota de 1000 intentos inicializada."
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            set_quota(1000, 0, api_key)
            return True, 0, "Clave API válida, pero se agotaron los tokens de la api (Límite 429 alcanzado)."
        else:
            raise ValueError(f"La Clave API no es válida: {error_msg}")

def get_gauge_html(api_key: str = None):
    """
    Genera el HTML premium del APILife Gauge.
    """
    if not api_key:
        api_key = get_active_api_key()
        
    quota = get_quota(api_key)
    limit = quota["limit"]
    remaining = quota["remaining"]
    
    percentage = int((remaining / limit) * 100) if limit > 0 else 0
    
    # Determinar color del gauge
    if percentage > 50:
        color = "#10b981"  # Verde esmeralda
        shadow = "rgba(16, 185, 129, 0.2)"
    elif percentage > 20:
        color = "#f59e0b"  # Naranja ámbar
        shadow = "rgba(245, 158, 11, 0.2)"
    else:
        color = "#ef4444"  # Rojo
        shadow = "rgba(239, 68, 68, 0.2)"
        
    status_text = "⚡ Conexión estable con la API" if remaining > 0 else "🔴 Se agotaron los tokens de la api"
    
    html_content = f"""
    <div class="api-life-container" style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.9rem;
        box-shadow: 0 4px 12px {shadow};
        text-align: left;
        font-family: 'Outfit', 'Inter', sans-serif;
        margin-top: 5px;
        min-width: 220px;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 700; color: #f8fafc; font-size: 0.9rem; letter-spacing: 0.5px;">❤️ APILife Gauge</span>
            <span style="font-weight: 800; color: {color}; font-size: 0.95rem;">{remaining} / {limit}</span>
        </div>
        <div style="background-color: #334155; border-radius: 6px; height: 10px; width: 100%; overflow: hidden; margin-bottom: 8px;">
            <div style="background-color: {color}; height: 100%; width: {percentage}%; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 8px {color};"></div>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500; display: flex; align-items: center; gap: 4px;">
            {status_text}
        </div>
    </div>
    """
    return html_content
