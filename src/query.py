import os
import sys
import json
from pathlib import Path

# Agregar el directorio raíz al path de Python para permitir importaciones
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DB_DIR, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY, LLM_MODEL_NAME, METADATA_FILE
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_active_document_title() -> str:
    """
    Retorna el título del documento activo leyendo el archivo metadata.json.
    """
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                return metadata.get("document_title", "el documento proporcionado")
        except Exception:
            pass
    return "el documento proporcionado"

def load_vector_store():
    """
    Carga la base de datos vectorial ChromaDB.
    """
    import streamlit as st
    if st.runtime.exists():
        if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
            return st.session_state["vector_store"]
            
        # Autoreparación: Si no está en session_state pero los metadatos existen, reconstruir en memoria
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                file_name = meta.get("file_name")
                if file_name:
                    from src.config import DATA_DIR
                    pdf_path = DATA_DIR / file_name
                    if pdf_path.exists():
                        from src.ingest import ingest_file
                        print(f"[*] Autoreparación: Reconstruyendo base de datos en memoria para {file_name}...")
                        vector_store = ingest_file(str(pdf_path))
                        return vector_store
            except Exception as e:
                print(f"[!] Error en autoreparación de Chroma en memoria: {e}")

    if not DB_DIR.exists() or not list(DB_DIR.glob("*")):
        raise FileNotFoundError(
            f"La base de datos vectorial no existe en: {DB_DIR}. "
            "Por favor, indexa un documento primero desde la web."
        )
        
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )
    return vector_store

def expand_query_for_retrieval(question: str) -> str:
    """
    TÉCNICA RAG AVANZADA: Expansión de Consultas (Query Expansion).
    Convierte preguntas coloquiales en español (ej. 'pasarse la luz roja') a términos legales
    formales (ej. 'semáforo rojo detención infracción') para mejorar la precisión de la búsqueda vectorial.
    """
    # Si no hay clave de API configurada, buscar usando la consulta original directamente
    if not GOOGLE_API_KEY or "tu_clave" in GOOGLE_API_KEY:
        return question
        
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        
        prompt = (
            "Eres un asistente de búsqueda legal. Tu tarea es convertir la consulta coloquial del usuario "
            "en una lista corta de palabras clave y sinónimos formales que se utilizarían en una ley "
            "o reglamento oficial para describir el mismo concepto.\n\n"
            f"Consulta del usuario: '{question}'\n\n"
            "Responde ÚNICAMENTE con los términos de búsqueda resultantes separados por espacios, sin explicaciones.\n"
            "Ejemplo: 'qué pasa si choco' -> 'colisión accidente choque obligaciones conductor daño'\n"
            "Ejemplo: 'pasarse la luz roja' -> 'semáforo luz roja detención infracción'"
        )
        
        response = llm.invoke(prompt)
        response_text = response.content
        if isinstance(response_text, list):
            response_text = "".join(part if isinstance(part, str) else part.get("text", "") for part in response_text)
        expanded_keywords = response_text.strip()
        
        # Combinamos la pregunta original con las palabras clave formales
        search_query = f"{question} {expanded_keywords}"
        print(f"[RAG] Consulta original: '{question}'")
        print(f"[RAG] Consulta expandida para búsqueda: '{search_query}'")
        return search_query
    except Exception as e:
        print(f"[!] Advertencia al expandir consulta: {e}")
        return question

def query_assistant(question: str):
    """
    Orquestación RAG completa:
    1. Carga la base de datos.
    2. Expande la pregunta para la búsqueda semántica.
    3. Recupera los chunks más afines (k=6 para mayor cobertura).
    4. Envía el contexto y la pregunta original al LLM.
    """
    try:
        vector_store = load_vector_store()
        
        # Usamos Maximum Marginal Relevance (MMR) con lambda_mult=0.7, k=10 y fetch_k=30 para ampliar el espectro de búsqueda
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7})
        
        # 1. Expandir la consulta coloquial para mejorar la coincidencia semántica
        search_query = expand_query_for_retrieval(question)
        
        # 2. Recuperar fragmentos usando la consulta expandida
        source_documents = retriever.invoke(search_query)
        
        # 3. Configurar el LLM
        if not GOOGLE_API_KEY or "tu_clave" in GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == "":
            raise ValueError(
                "La API Key de Google (GOOGLE_API_KEY) no está configurada o es inválida en el archivo .env."
            )
            
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        
        # 4. Formatear el contexto recuperado
        context_text = "\n\n".join(doc.page_content for doc in source_documents)
        
        # 5. Prompt de generación final (instrucciones genéricas y exhaustivas basadas en contexto)
        doc_title = get_active_document_title()
        system_prompt = (
            "Eres un analista experto en extracción y síntesis de información de documentos técnicos, legales o corporativos.\n\n"
            "Directiva de Extracción Directa:\n"
            "- Cuando el usuario pregunte por fases, pasos, procedimientos, listas o características, extrae la información de manera directa y estructurada utilizando los fragmentos recuperados. Si los pasos están distribuidos en diferentes secciones, sintetízalos de forma lógica y coherente sin indicar que falta un esquema estandarizado.\n\n"
            "Citas Obligatorias:\n"
            "- Mantén la rigurosidad de citar las secciones, capítulos o números de página correspondientes del documento activo en cada respuesta.\n\n"
            "Análisis de Casos Especiales:\n"
            "- Analiza con suma atención todos los fragmentos provistos. Si la pregunta del usuario aborda una excepción técnica, regla de validación o caso especial (como comportamiento de nulos, restricciones o límites), busca en las secciones avanzadas o apéndices del documento antes de concluir que la información no está presente.\n\n"
            "Instrucción de Lenguaje:\n"
            "- Si el usuario formula preguntas utilizando lenguaje coloquial, abreviado o indirecto, analiza la intención e intenta vincularla con los conceptos, términos o datos que aparezcan en el contexto.\n\n"
            "Restricción de Alucinación:\n"
            "- Si la respuesta exacta no se encuentra dentro de los fragmentos provistos del documento, indícalo educadamente al usuario y evita inventar información o asumir datos externos.\n\n"
            "Contexto recuperado del documento:\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # 6. Ejecutar generación final con la pregunta original del usuario
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return {
            "answer": answer,
            "sources": source_documents
        }
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "answer": "Se ha superado temporalmente el límite de la API gratuita. Por favor, espera unos segundos e intenta nuevamente.",
                "sources": []
            }
        return {
            "answer": f"Error al procesar la consulta: {str(e)}",
            "sources": []
        }

if __name__ == "__main__":
    test_query = "¿Qué pasa si me paso la luz roja?"
    print(f"[*] Haciendo pregunta de prueba: '{test_query}'")
    
    try:
        result = query_assistant(test_query)
        print("\n=== RESPUESTA DEL ASISTENTE ===")
        print(result["answer"])
        print("\n=== FUENTES UTILIZADAS ===")
        for i, doc in enumerate(result["sources"], 1):
            page = doc.metadata.get("page", "N/A")
            if isinstance(page, int):
                page += 1
            print(f"\n[Fuente {i}] - Página {page}:")
            print(doc.page_content[:150] + "...")
    except Exception as e:
        print(f"[!] Error: {e}")
