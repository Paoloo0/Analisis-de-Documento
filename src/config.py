import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env si existe
load_dotenv()

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos y base vectorial
DATA_DIR = BASE_DIR / "data"

# En entornos cloud basados en Linux (como Streamlit Cloud), la carpeta del proyecto puede tener
# restricciones de montaje para SQLite (bloqueos y WAL). Usar /tmp garantiza permisos de escritura.
if os.name == "nt":
    DB_DIR = BASE_DIR / "chromadb_store"
else:
    DB_DIR = Path("/tmp/chromadb_store")

# Asegurar que los directorios existan
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Archivo JSON para almacenar dinámicamente las preguntas sugeridas del documento activo
METADATA_FILE = DATA_DIR / "metadata.json"

# Configuración del Modelo de Embeddings (local y ligero)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración del LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Nombre del modelo de lenguaje a usar
LLM_MODEL_NAME = "gemini-flash-latest"
