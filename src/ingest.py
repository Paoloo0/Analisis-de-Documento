import os
import sys
import json
import shutil
from pathlib import Path

# Agregar el directorio raíz al path de Python para poder importar desde src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DB_DIR, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY, LLM_MODEL_NAME, METADATA_FILE
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from src.quota import get_quota, use_quota, set_quota_to_zero

def generate_suggested_questions(document_sample: str, doc_name: str) -> list:
    """
    Usa el LLM (Gemini) para leer una muestra del documento y generar 3 preguntas sugeridas
    apropiadas y específicas sobre su contenido.
    """
    if not GOOGLE_API_KEY:
        print("[!] Advertencia: No hay API Key para generar preguntas dinámicas. Se usarán preguntas por defecto.")
        if "transito" in doc_name.lower() or "reglamento" in doc_name.lower():
            return [
                "¿Cuál es el límite de velocidad en avenidas y calles?",
                "¿Es obligatorio el chaleco reflectivo en motocicleta?",
                "¿Qué documentos debe portar obligatoriamente un conductor?"
            ]
        return [
            "¿De qué trata principalmente este documento?",
            "¿Cuáles son los 3 puntos más importantes o conclusiones del texto?",
            "¿Qué directivas o conceptos clave se explican en el documento?"
        ]
        
    try:
        # Verificar si se agotaron los tokens de la API
        if get_quota(GOOGLE_API_KEY)["remaining"] <= 0:
            raise ValueError("Se agotaron los tokens de la api")
            
        print("[*] Consultando a Gemini para generar preguntas sugeridas dinámicas...")
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        
        prompt = (
            "Analiza el siguiente extracto de un documento. "
            "Genera exactamente 3 preguntas diferentes, claras, concretas y de interés general en español "
            "que puedan responderse leyendo este documento.\n\n"
            "Responde ÚNICAMENTE con una lista en formato JSON de strings. Ejemplo de formato:\n"
            '["¿Pregunta 1?", "¿Pregunta 2?", "¿Pregunta 3?"]\n\n'
            "No incluyas formateo markdown de código (como ```json) ni comentarios extras, solo el JSON puro.\n\n"
            f"Extracto del documento:\n{document_sample}"
        )
        
        # Llamar al modelo
        response = llm.invoke(prompt)
        use_quota(1, GOOGLE_API_KEY)  # Descontar 1 intento
        response_text = response.content
        if isinstance(response_text, list):
            response_text = "".join(part if isinstance(part, str) else part.get("text", "") for part in response_text)
        
        # Limpieza por si el modelo incluye formato markdown ```json
        clean_response = response_text.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        questions = json.loads(clean_response)
        
        if isinstance(questions, list) and len(questions) >= 3:
            return questions[:3]
            
        raise ValueError("El formato retornado no es una lista válida.")
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            set_quota_to_zero(GOOGLE_API_KEY)
        print(f"[!] Error al generar preguntas automáticas ({e}). Se usarán preguntas de respaldo.")
        if "transito" in doc_name.lower() or "reglamento" in doc_name.lower():
            return [
                "¿Cuál es el límite de velocidad en avenidas y calles?",
                "¿Es obligatorio el chaleco reflectivo en motocicleta?",
                "¿Qué documentos debe portar obligatoriamente un conductor?"
            ]
        return [
            "¿De qué trata principalmente este documento?",
            "¿Cuáles son los 3 puntos más importantes o conclusiones del texto?",
            "¿Qué directivas o conceptos clave se explican en el documento?"
        ]

def save_document_metadata(file_path: str, chunks: list, db_dir_path: str):
    """
    Extrae información del archivo subido, genera las preguntas sugeridas dinámicas
    y guarda los metadatos en un archivo JSON.
    """
    path_obj = Path(file_path)
    file_name = path_obj.name
    
    # Crear un título amigable a partir del nombre del archivo
    doc_title = path_obj.stem.replace("_", " ").replace("-", " ").capitalize()
    
    # Extraer los primeros fragmentos como muestra
    sample_chunks = chunks[:3]
    sample_text = "\n\n".join(chunk.page_content for chunk in sample_chunks)
    
    # Generar las 3 preguntas sugeridas
    suggested_questions = generate_suggested_questions(sample_text, doc_title)
    
    metadata = {
        "file_name": file_name,
        "document_title": doc_title,
        "suggested_questions": suggested_questions,
        "db_dir": db_dir_path
    }
    
    # Escribir metadatos a archivo JSON
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Metadatos del documento guardados en: {METADATA_FILE}")
    print(f"    - Título: {doc_title}")
    print(f"    - Preguntas sugeridas: {suggested_questions}")
    print(f"    - Directorio DB: {db_dir_path}")

def ingest_file(file_path: str):
    """
    Función que lee un archivo (PDF o TXT),
    lo divide en fragmentos, genera sus embeddings, los guarda en ChromaDB,
    y escribe los metadatos correspondientes del documento activo.
    """
    print(f"[*] Iniciando ingesta del archivo: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {file_path}")
        
    path_obj = Path(file_path)
    extension = path_obj.suffix.lower()
    
    # PASO 1: Cargar el documento según su extensión
    if extension == ".pdf":
        print("[*] Cargando archivo PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif extension == ".txt":
        print("[*] Cargando archivo de texto plano...")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
    else:
        raise ValueError(f"Extensión de archivo no soportada: {extension}. Use .pdf o .txt")
        
    print(f"[+] Documento cargado exitosamente. Total de páginas/documentos: {len(documents)}")
    
    # PASO 2: Dividir el texto en fragmentos (Chunks)
    chunk_size = 2000
    chunk_overlap = 400
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"[+] Texto dividido en {len(chunks)} fragmentos (chunks).")
    
    # PASO 3: Inicializar el modelo de embeddings (usando la API de Gemini con limitador de velocidad personalizado para evitar errores 429)
    from src.embeddings import get_embeddings
    print("[*] Inicializando modelo de embeddings de Gemini con limitador de tasa...")
    embeddings = get_embeddings(GOOGLE_API_KEY)
    print(f"[DEBUG] GOOGLE_API_KEY configured: {GOOGLE_API_KEY is not None}")
    print(f"[DEBUG] embeddings class: {type(embeddings)}")
    print("[+] Modelo de embeddings listo.")
    
    # PASO 4: Limpiar la persistencia anterior de ChromaDB
    # Esto asegura que el asistente responda SÓLO sobre el archivo cargado actualmente
    import time
    safe_name = "".join([c if c.isalnum() else "_" for c in path_obj.stem])
    unique_db_dir = DB_DIR / f"db_{safe_name}_{int(time.time())}"
    print(f"[*] Carpeta de base de datos asignada: {unique_db_dir}")
    
    # Limpiar session state si existe
    import streamlit as st
    if "vector_store" in st.session_state:
        st.session_state["vector_store"] = None
        
    import gc
    gc.collect()

    # Intentamos borrar bases de datos viejas en DB_DIR para ahorrar espacio.
    # Si están bloqueadas por el SO, se ignora y se continúa.
    if DB_DIR.exists():
        for old_dir in DB_DIR.glob("db_*"):
            if old_dir.is_dir() and old_dir != unique_db_dir:
                try:
                    shutil.rmtree(old_dir)
                    print(f"[+] Carpeta antigua eliminada: {old_dir.name}")
                except Exception:
                    pass

    try:
        unique_db_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[!] Advertencia al crear directorio de ChromaDB: {e}")
    
    # PASO 5: Guardar los fragmentos en ChromaDB
    print("[*] Indexando fragmentos en ChromaDB...")
    
    import streamlit as st
    if os.name != "nt" and st.runtime.exists():
        # En la nube (Linux) bajo Streamlit, usamos base de datos en memoria
        # almacenada en st.session_state para evitar problemas de permisos de SQLite.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        st.session_state["vector_store"] = vector_store
        print("[+] Fragmentos guardados en ChromaDB (En Memoria - Streamlit Session State).")
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(unique_db_dir)
        )
        try:
            vector_store.persist()
        except Exception:
            pass
        print("[+] Fragmentos guardados en ChromaDB (Persistente en disco).")
        
    # PASO 6: Generar y guardar las preguntas sugeridas dinámicas
    save_document_metadata(file_path, chunks, str(unique_db_dir))
    
    print("[+] ¡Indexación y adaptación completadas con éxito!")
    return vector_store

if __name__ == "__main__":
    # Buscar archivos .pdf o .txt en la carpeta data/
    supported_files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt"))
    supported_files = [f for f in supported_files if f.name != "metadata.json"]
    
    if not supported_files:
        print(f"[!] No se encontró ningún archivo (.pdf o .txt) en la carpeta: {DATA_DIR}")
        print("[!] Por favor, coloca un archivo en la carpeta 'data' y vuelve a ejecutar.")
        sys.exit(1)
        
    # Usamos el primer archivo que encontremos (priorizando PDFs si los hay)
    pdfs = [f for f in supported_files if f.suffix.lower() == ".pdf"]
    target_file = pdfs[0] if pdfs else supported_files[0]
    
    try:
        ingest_file(str(target_file))
    except Exception as e:
        print(f"[!] Error durante la ingesta: {e}")
