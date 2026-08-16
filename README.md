# 📂 Analizador Inteligente de Documentos (RAG)

Se trata de un **MVP (Producto Mínimo Viable)** de Preguntas y Respuestas (Q&A) basado en **RAG (Retrieval-Augmented Generation)**. Permite subir cualquier archivo PDF o de Texto (.txt) e interactuar directamente con él en lenguaje natural, obteniendo respuestas precisas respaldadas por las fuentes del propio documento.

---

## 🎯 Motivación y Aprendizaje
El objetivo de este proyecto fue diseñar e implementar un sistema RAG desde cero, afrontando retos reales de desarrollo como:
*   **Gestión de límites de APIs (Quota Management):** Implementación de reintentos activos con backoff exponencial y un visualizador dinámico en tiempo real (`APILife Gauge`) para monitorizar los tokens consumidos y evitar bloqueos.
*   **Búsqueda semántica eficiente:** Uso de bases de datos vectoriales locales para recuperar información con precisión matemática sin depender de servicios de pago.
*   **Experiencia de usuario fluida:** Diseño de una interfaz adaptativa que responde y se personaliza según el tema del documento ingresado.

---

## 🛠️ Tecnologías y Lenguajes de Desarrollo

Este proyecto ha sido desarrollado utilizando el siguiente stack tecnológico principal:

*   **Lenguaje de Programación:** Python 3.10+ (desarrollado y probado bajo Python 3.12).
*   **Interfaz Gráfica / Frontend:** Python (utilizando la librería gráfica de visualización interactiva Streamlit).
*   **Orquestación RAG:** LangChain & LangChain Community (para carga de documentos, división de texto y pipelines de prompts).
*   **Base de Datos Vectorial:** ChromaDB (base de datos vectorial local incrustada y optimizada para búsquedas semánticas rápidas).
*   **Modelos de Inteligencia Artificial:**
    *   **LLM (Generación y Preguntas):** Google Gemini 1.5 Flash (a través del SDK oficial de Google GenAI).
    *   **Embeddings (Búsqueda Vectorial):** Google Gemini Embeddings (modelo `models/gemini-embedding-001` de 3072 dimensiones).

---

## 🏗️ Arquitectura de Datos RAG

El flujo de información de la plataforma se compone de dos fases:

### 1. Ingesta y Adaptación
*   **Carga:** El usuario sube un archivo (.pdf o .txt). El sistema limpia la base vectorial anterior para evitar mezclas de contexto.
*   **Segmentación (Chunking):** El texto se corta en bloques de **1000 caracteres** con un solape de **200 caracteres (20%)** para mantener la coherencia y evitar cortes en mitad de oraciones.
*   **Vectorización (Embeddings):** Cada bloque se traduce a un vector numérico de 3072 dimensiones usando la API de Gemini Embeddings.
*   **Guardado en ChromaDB:** Los fragmentos vectorizados se guardan en la base local `chromadb_store/`.
*   **Análisis Dinámico:** El LLM (**Gemini**) analiza un fragmento inicial del documento y genera **3 preguntas sugeridas de ejemplo** que se guardan en `data/metadata.json` para personalizar la web.

### 2. Consulta y Respuestas
*   **Retrieval (Recuperación):** Al escribir una pregunta, ChromaDB realiza una búsqueda con **MMR (Maximum Marginal Relevance)** evaluando 30 candidatos (`fetch_k=30`) y seleccionando los **10 fragmentos** más diversos y relevantes (`k=10`) para optimizar la respuesta.
*   **Generation (Generación):** Estos fragmentos se insertan como contexto en un prompt del sistema. Gemini redacta la respuesta basándose en ese contexto y cita las fuentes correspondientes (páginas y textos originales) para asegurar veracidad total.

---

## 📁 Estructura del Proyecto

```text
analizador_documentos_rag/
├── data/
│   └── metadata.json               # Almacena dinámicamente el título y sugerencias del documento activo
├── chromadb_store/                 # Directorio local de base de datos vectorial ChromaDB
├── src/
│   ├── config.py                   # Configuración del entorno, rutas y modelos
│   ├── ingest.py                   # Lógica para indexar y autogenerar preguntas clave del documento
│   ├── query.py                    # Motor de recuperación semántica y generación con LangChain
│   └── quota.py                    # Gestor de cuotas y lógica del APILife Gauge
├── app.py                          # Interfaz interactiva de Streamlit (UI universal adaptativa)
└── requirements.txt                # Librerías y dependencias necesarias
```

---

## 🛠️ Instalación y Configuración

### 1. Requisitos
*   Python 3.10 o superior (Se recomienda 3.12).
*   Una API Key de Google Gemini (puedes crearla de forma gratuita en [Google AI Studio](https://aistudio.google.com/)).

### 2. Instalación de Dependencias
Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (en Windows)
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar API Key de Gemini
Crea un archivo llamado `.env` en la raíz del proyecto y escribe tu clave de API:

```text
GOOGLE_API_KEY=tu_clave_de_api_aquí
```

---

## 🚀 Cómo Ejecutar la Aplicación

Una vez configurado tu entorno, inicia la aplicación web interactiva:

```bash
streamlit run app.py
```

Streamlit abrirá tu navegador web por defecto en la dirección: `http://localhost:8501`.

### Primeros Pasos:
*   Si la base de datos está limpia, la aplicación te mostrará un gran **cargador de archivos** en el centro.
*   Arrastra y suelta cualquier archivo PDF o de texto.
*   Haz clic en **"Iniciar Ingesta y Análisis"**.
*   ¡Listo! Verás que la interfaz se transforma automáticamente para coincidir con tu documento y te presenta 3 botones con preguntas sugeridas clave del tema de tu archivo para empezar a chatear.
