import os
import sys
from pathlib import Path

# HACK de compatibilidad para ChromaDB y SQLite en Streamlit Community Cloud (Linux)
if os.name != "nt":
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

import json
import time
import streamlit as st

# Agregar la ruta del proyecto al path de Python para permitir importaciones correctas
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import DATA_DIR, DB_DIR, GOOGLE_API_KEY, METADATA_FILE
from src.ingest import ingest_file
from src.query import query_assistant

# --- CARGAR METADATOS DEL DOCUMENTO ACTIVO ---
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "sidebar_uploader_0"
if "main_uploader_key" not in st.session_state:
    st.session_state["main_uploader_key"] = "main_uploader_0"
if "api_input_key" not in st.session_state:
    st.session_state["api_input_key"] = "new_api_key_input_0"

doc_title = "Ningún documento cargado"
file_name = ""
suggested_questions = []

# Verificar si la base vectorial está creada y si existe el archivo de metadatos del documento activo
db_exists = False
active_db_dir = DB_DIR

if METADATA_FILE.exists():
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            file_name = metadata.get("file_name", "").strip()
            if file_name:  # Solo si hay un archivo activo asignado
                doc_title = metadata.get("document_title", "Documento Indexado")
                suggested_questions = metadata.get("suggested_questions", [])
                db_dir_str = metadata.get("db_dir")
                if db_dir_str:
                    active_db_dir = Path(db_dir_str)
                    db_exists = active_db_dir.exists() and len(list(active_db_dir.glob("*"))) > 0
                else:
                    # Para compatibilidad con metadatos antiguos
                    active_db_dir = DB_DIR
                    db_exists = DB_DIR.exists() and len(list(DB_DIR.glob("*"))) > 0
    except Exception as e:
        print(f"Error cargando metadatos: {e}")

# Configuración de la página en Streamlit
st.set_page_config(
    page_title="Analizador Inteligente de Documentos (RAG)",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DISEÑO ESTÉTICO PERSONALIZADO (CSS) ---
st.markdown("""
<style>
    /* Ocultar elementos nativos de Streamlit (menú de 3 puntos, botón Deploy y pie de página) */
    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    .stAppDeployButton {display: none !important;}
    #stDecoration {display: none !important;}

    /* Estilos globales y paleta de colores premium (azul institucional y gris oscuro) */
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* Títulos y fuentes */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #f8fafc;
        font-weight: 700;
    }
    
    .main-title {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Contenedor de respuesta principal */
    .answer-box {
        background-color: #1e293b;
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    /* Tarjetas de fuentes citadas */
    .source-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .source-header {
        font-weight: bold;
        color: #60a5fa;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }
    
    .source-body {
        font-style: italic;
        color: #cbd5e1;
        font-size: 0.85rem;
    }
    
    /* Indicador de estado */
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .status-active {
        background-color: #065f46;
        color: #34d399;
    }
    .status-inactive {
        background-color: #7f1d1d;
        color: #fca5a5;
    }
    
    /* Ocultar widgets flotantes de accesibilidad de terceros (UserWay, AccessiBe, etc.) */
    iframe[id*="userway"],
    iframe[src*="userway"],
    iframe[title*="Accessibility"],
    #userwayAccessibilityIcon,
    .userway-accessibility-icon,
    #uaw-widget,
    .uaw-button,
    #accessibe,
    .accessibe,
    #equalweb-accessibility,
    .equalweb-accessibility,
    #accessibility-widget,
    .accessibility-widget,
    #accessibility-icon,
    .accessibility-icon,
    div[id*="userway"],
    div[class*="userway"],
    div[id*="accessibe"],
    div[class*="accessibe"],
    div[id*="equalweb"],
    div[class*="equalweb"],
    div[class*="accessibility"],
    div[id*="accessibility"],
    [class*="userway"],
    [id*="userway"],
    [class*="accessibe"],
    [id*="accessibe"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Centrar contenido (como emojis) de los botones en la barra lateral */
    [data-testid="stSidebar"] .stButton button {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 0px !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL (ESTADOS Y CONFIGURACIÓN) ---
with st.sidebar:
    st.markdown("### ⚙️ Configuración RAG")
    
    # Validar API Key (Verificar que exista y no sea el valor por defecto)
    is_key_configured = GOOGLE_API_KEY and "tu_clave" not in GOOGLE_API_KEY and GOOGLE_API_KEY.strip() != ""
    
    if is_key_configured:
        st.markdown(
            '<div class="status-badge status-active">● API Key de Gemini Activa</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge status-inactive">● API Key de Gemini Faltante</div>', 
            unsafe_allow_html=True
        )
        st.warning("⚠️ Configura una clave válida para habilitar el motor RAG.")
        
    # Estado de la base de datos
    if db_exists:
        st.success("✅ Base Vectorial ChromaDB Lista")
        # Mostrar archivo activo y botón de eliminar (tacho de basura)
        col_file, col_delete = st.columns([0.85, 0.15])
        with col_file:
            st.info(f"📄 **Archivo activo:**\n`{file_name}`")
        with col_delete:
            st.write("")  # Espaciado para alinear verticalmente el botón
            st.write("")
            if st.button("🗑️", help="Eliminar este documento y limpiar la base de datos"):
                try:
                    import shutil
                    import gc
                    # 1. Eliminar archivo físico de la carpeta data/ si el nombre es válido y es un archivo
                    if file_name:
                        active_file_path = DATA_DIR / file_name
                        if active_file_path.exists() and active_file_path.is_file():
                            try:
                                os.remove(active_file_path)
                            except Exception as file_err:
                                print(f"Advertencia al borrar archivo físico: {file_err}")
                            
                    # 2. Eliminar el JSON de metadatos
                    if METADATA_FILE.exists():
                        try:
                            os.remove(METADATA_FILE)
                        except Exception as meta_err:
                            print(f"Advertencia al borrar metadatos: {meta_err}")
                        
                    # 3. Intentar limpiar base vectorial de ChromaDB
                    gc.collect()
                    if active_db_dir.exists() and active_db_dir != DB_DIR:
                        shutil.rmtree(active_db_dir, ignore_errors=True)
                    if DB_DIR.exists():
                        shutil.rmtree(DB_DIR, ignore_errors=True)
                    if "vector_store" in st.session_state:
                        st.session_state["vector_store"] = None
                        
                    st.success("¡Documento eliminado exitosamente!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al eliminar: {e}")
    else:
        st.error("❌ Ningún documento indexado")
        
    st.markdown("---")
    
    # Menú desplegable para gestionar clave API (⋯)
    with st.sidebar.expander("🔑 Gestionar Clave API ⋯"):
        st.write("**Clave de API actual:**")
        if GOOGLE_API_KEY:
            col_key, col_del_key = st.columns([0.82, 0.18])
            with col_key:
                masked_key = f"{GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-5:]}" if len(GOOGLE_API_KEY) > 13 else GOOGLE_API_KEY
                st.code(masked_key, language="")
            with col_del_key:
                st.write("") # Alineador vertical
                if st.button("🗑️", key="delete_api_key", help="Eliminar la clave de API actual"):
                    try:
                        # 1. Eliminar .env
                        env_path = Path(__file__).resolve().parent / ".env"
                        if env_path.exists():
                            os.remove(env_path)
                        # 2. Quitar del entorno de la sesión activa
                        if "GOOGLE_API_KEY" in os.environ:
                            del os.environ["GOOGLE_API_KEY"]
                        # 3. Recargar módulos
                        import importlib
                        import src.config
                        import src.query
                        import src.ingest
                        src.config.GOOGLE_API_KEY = ""
                        src.query.GOOGLE_API_KEY = ""
                        src.ingest.GOOGLE_API_KEY = ""
                        importlib.reload(src.config)
                        importlib.reload(src.query)
                        importlib.reload(src.ingest)
                        
                        # 4. Cambiar la clave del widget de texto para resetearlo a vacío
                        st.session_state["api_input_key"] = f"new_api_key_input_{int(time.time())}"
                        
                        st.success("¡Clave de API eliminada!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No configurada.")
            
        new_key = st.text_input(
            "Nueva Clave de API:",
            type="password",
            placeholder="Pega aquí tu clave Gemini (AQ...)",
            key=st.session_state["api_input_key"]
        )
        
        if st.button("Guardar Cambios"):
            if new_key.strip() == "":
                st.error("Por favor, ingresa una clave válida.")
            else:
                try:
                    # 1. Guardar en .env
                    env_path = Path(__file__).resolve().parent / ".env"
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.write(f"GOOGLE_API_KEY={new_key.strip()}\n")
                    
                    # 2. Actualizar variables de entorno de la sesión activa
                    os.environ["GOOGLE_API_KEY"] = new_key.strip()
                    
                    # 3. Recargar módulos para asegurar consistencia
                    import importlib
                    import src.config
                    import src.query
                    import src.ingest
                    importlib.reload(src.config)
                    importlib.reload(src.query)
                    importlib.reload(src.ingest)
                    
                    # 4. Cambiar la clave del widget para forzar su recreación vacía
                    st.session_state["api_input_key"] = f"new_api_key_input_{int(time.time())}"
                    
                    st.success("¡Clave de API guardada y actualizada exitosamente!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")
                    
    st.markdown("---")
    
    # Widget de carga secundario en el sidebar (para reemplazar el documento cuando uno ya esté activo)
    if db_exists:
        st.markdown("### 📥 Cambiar de Documento")
        new_file = st.file_uploader(
            "Sube un nuevo archivo para reemplazar el anterior (.pdf, .txt)", 
            type=["pdf", "txt"],
            key=st.session_state["uploader_key"]
        )
        if new_file is not None:
            save_path = DATA_DIR / new_file.name
            with open(save_path, "wb") as f:
                f.write(new_file.getbuffer())
            if st.button("🚀 Reemplazar e Indexar"):
                with st.spinner("Procesando nuevo documento..."):
                    try:
                        ingest_file(str(save_path))
                        # Cambiar la clave del widget para forzar el reset completo del file_uploader
                        st.session_state["uploader_key"] = f"sidebar_uploader_{int(time.time())}"
                        st.success("¡Documento indexado con éxito!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
    st.markdown("""
    **¿Cómo funciona?**
    1. **Sube tu archivo:** El texto se procesa y fragmenta.
    2. **Autogeneración:** Gemini lee el documento y genera 3 preguntas sugeridas clave.
    3. **Búsqueda Vectorial:** ChromaDB busca fragmentos relevantes de tu archivo para responder tu pregunta.
    4. **Generación veraz:** Gemini redacta la respuesta usando únicamente el contexto recuperado, citando el origen exacto.
    """)
    
    import sqlite3
    with st.expander("🛠️ Diagnóstico del Sistema"):
        st.write(f"**OS:** {os.name}")
        st.write(f"**Python:** {sys.version.split()[0]}")
        st.write(f"**SQLite versión:** {sqlite3.sqlite_version}")
        st.write(f"**DB Path:** `{DB_DIR}`")
        try:
            DB_DIR.mkdir(parents=True, exist_ok=True)
            test_f = DB_DIR / "test.txt"
            test_f.write_text("ok")
            test_f.unlink()
            st.success("Escritura en DB_DIR: OK")
        except Exception as e:
            st.error(f"Escritura en DB_DIR: Error: {e}")

# --- PÁGINA PRINCIPAL ---

# Caso 1: No hay ningún documento indexado
if not db_exists:
    st.markdown('<div class="main-title">Analizador Inteligente de Documentos (RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sube cualquier archivo PDF o TXT y chatea con él con respuestas precisas libres de alucinaciones.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contenedor de carga principal y centrado
    st.markdown("### 📂 Sube tu primer documento para comenzar")
    main_file = st.file_uploader(
        "Arrastra y suelta tu archivo PDF o TXT aquí", 
        type=["pdf", "txt"],
        key=st.session_state["main_uploader_key"]
    )
    
    if main_file is not None:
        save_path = DATA_DIR / main_file.name
        with open(save_path, "wb") as f:
            f.write(main_file.getbuffer())
            
        st.success(f"Archivo guardado exitosamente: `{main_file.name}`")
        
        if not is_key_configured:
            st.warning("⚠️ **Clave de API de Gemini faltante.** Por favor, configura una clave válida en la barra lateral (*🔑 Gestionar Clave API*) para poder iniciar la ingesta y analizar documentos.")
        else:
            if st.button("🚀 Iniciar Ingesta y Análisis", type="primary"):
                with st.spinner("Indexando y analizando el contenido del documento..."):
                    try:
                        ingest_file(str(save_path))
                        # Cambiar la clave del widget para forzar la recarga limpia en la pantalla de inicio
                        st.session_state["main_uploader_key"] = f"main_uploader_{int(time.time())}"
                        st.success("¡Documento indexado con éxito! Cargando interfaz conversacional...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al procesar: {e}")
    else:
        st.info("💡 Por favor, selecciona o arrastra un archivo de texto (.txt) o PDF (.pdf) en el recuadro superior para poder chatear con él.")

# Caso 2: Ya existe un documento indexado y listo
else:
    st.markdown(f'<div class="main-title">Analizador de: {doc_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Preguntas y respuestas automáticas basadas en el documento activo <code>{file_name}</code>.</div>', unsafe_allow_html=True)
    
    # Restricción por falta de API Key
    if not is_key_configured:
        st.warning("⚠️ **Clave de API de Gemini faltante.** Por favor, ingresa una clave de API válida en la barra lateral en la sección *🔑 Gestionar Clave API ⋯* para poder realizar consultas sobre el documento.")
    else:
        # Preguntas sugeridas dinámicas
        if suggested_questions:
            st.markdown("### 💡 Preguntas sugeridas sobre este documento:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(suggested_questions) > 0 and st.button(suggested_questions[0]):
                    st.session_state.pregunta = suggested_questions[0]
            with col2:
                if len(suggested_questions) > 1 and st.button(suggested_questions[1]):
                    st.session_state.pregunta = suggested_questions[1]
            with col3:
                if len(suggested_questions) > 2 and st.button(suggested_questions[2]):
                    st.session_state.pregunta = suggested_questions[2]
                    
        if "pregunta" not in st.session_state:
            st.session_state.pregunta = ""
            
        # Input de búsqueda
        user_query = st.text_input(
            "Haz tu pregunta sobre el contenido del documento:",
            value=st.session_state.pregunta,
            placeholder="Ej: ¿Cuáles son las conclusiones o puntos principales descritos en el texto?"
        )
        
        if user_query:
            st.markdown("---")
            with st.spinner("Buscando en el documento y redactando respuesta..."):
                result = query_assistant(user_query)
                
            st.markdown("### 💬 Respuesta del Asistente")
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>', 
                unsafe_allow_html=True
            )
            
            # Mostrar los fragmentos citados
            st.markdown("### 📚 Fragmentos de Origen Citados")
            if result["sources"]:
                for i, doc in enumerate(result["sources"], 1):
                    source_name = Path(doc.metadata.get("source", file_name)).name
                    page = doc.metadata.get("page", None)
                    
                    if page is not None:
                        page_str = f" | Página {page + 1}"
                    else:
                        page_str = ""
                        
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">Fragmento {i}: {source_name}{page_str}</div>
                        <div class="source-body">"{doc.page_content}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No se recuperaron fragmentos específicos para esta consulta.")
