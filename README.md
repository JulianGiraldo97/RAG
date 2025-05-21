# Sistema RAG para Información Financiera

Este sistema implementa un RAG (Retrieval Augmented Generation) para procesar y responder preguntas sobre información financiera de la empresa.

## Formatos de Archivo Soportados

El sistema puede procesar los siguientes formatos de archivo:
- **PDF**: Documentos financieros, reportes, estados financieros
- **JSON**: Datos estructurados, configuraciones, métricas financieras
- **CSV**: Datos tabulares, hojas de cálculo, registros financieros

## Arquitectura del Sistema

El sistema está compuesto por tres componentes principales:

### 1. Procesador de Documentos (`DocumentProcessor`)
- Utiliza `PyPDFLoader` para cargar documentos PDF
- Implementa `RecursiveCharacterTextSplitter` para dividir documentos en chunks
- Configuración por defecto:
  - Tamaño de chunk: 1000 caracteres
  - Solapamiento: 200 caracteres
- Procesa directorios completos de documentos PDF

### 2. Almacén de Vectores (`VectorStore`)
- Implementa FAISS (Facebook AI Similarity Search) para búsqueda de similitud
- Soporta dos modelos de embeddings:
  - OpenAI Embeddings (por defecto)
  - HuggingFace Embeddings (modelo: sentence-transformers/all-MiniLM-L6-v2)
- Funcionalidades:
  - Creación de vector store
  - Persistencia en disco
  - Búsqueda por similitud con k=4 documentos más relevantes

### 3. Sistema RAG (`RAGSystem`)
- Integra el procesamiento de documentos y el almacén de vectores
- Implementa dos modelos de lenguaje:
  - OpenAI GPT-3.5-turbo (por defecto)
  - HuggingFace Flan-T5-large
- Prompt especializado para contexto financiero
- Cadena de procesamiento:
  1. Recuperación de contexto relevante
  2. Formateo del prompt
  3. Generación de respuesta
  4. Parseo de salida

## Requisitos Técnicos

- Python 3.9+
- API key de OpenAI o HuggingFace
- Documentos PDF con información financiera
- Dependencias principales:
  - langchain==0.1.0
  - langchain-openai==0.0.2
  - langchain-community==0.0.13
  - pypdf==3.17.1
  - chromadb==0.4.22
  - sentence-transformers==2.2.2
  - faiss-cpu==1.7.4

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
3. Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:
```
OPENAI_API_KEY=tu_api_key_aqui
# o
HUGGINGFACE_API_KEY=tu_api_key_aqui
```

## Estructura de Directorios

```
.
├── data/                  # Archivos con información financiera (PDF, JSON, CSV)
├── domain_knowledge/      # Documentos de conocimiento del dominio
├── src/                   # Código fuente
│   ├── __init__.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── rag_system.py
│   └── main.py
├── vector_store/          # Almacenamiento persistente de vectores
├── requirements.txt
└── README.md
```

## Flujo de Procesamiento

1. **Carga de Documentos**:
   - Los PDFs se cargan y dividen en chunks
   - Se procesan tanto documentos financieros como de conocimiento del dominio

2. **Indexación**:
   - Los chunks se convierten en embeddings
   - Se almacenan en FAISS para búsqueda eficiente

3. **Consulta**:
   - La pregunta del usuario se procesa
   - Se recuperan los documentos más relevantes
   - Se genera una respuesta contextualizada

## Características Técnicas

- **Procesamiento de Documentos**:
  - Soporte para múltiples formatos (PDF, JSON, CSV)
  - División inteligente de texto
  - Manejo de datos estructurados y no estructurados
  - Preservación de contexto

- **Búsqueda Semántica**:
  - FAISS para búsqueda de similitud
  - Embeddings de alta calidad
  - Recuperación de contexto relevante

- **Generación de Respuestas**:
  - Prompt especializado en finanzas
  - Contexto adaptativo
  - Respuestas basadas en evidencia

- **Persistencia**:
  - Almacenamiento eficiente de vectores
  - Carga rápida de índices
  - Reutilización de embeddings

## Uso

1. Colocar los PDFs en el directorio `data/`
2. Colocar los documentos de conocimiento del dominio en `domain_knowledge/`
3. Ejecutar el sistema:
```bash
python src/main.py
```

## Consideraciones Técnicas

- **Rendimiento**:
  - FAISS optimiza la búsqueda de similitud
  - Chunks de tamaño óptimo para balancear contexto y precisión
  - Caché de embeddings para consultas repetidas

- **Escalabilidad**:
  - Arquitectura modular
  - Fácil integración de nuevos modelos
  - Soporte para múltiples formatos de documento

- **Seguridad**:
  - Manejo seguro de API keys
  - Validación de entrada
  - Manejo de errores robusto

## Limitaciones y Mejoras Futuras

- Implementar soporte para más formatos de documento
- Añadir métricas de evaluación de calidad
- Optimizar el tamaño de chunks según el dominio
- Implementar caché de respuestas frecuentes
- Añadir soporte para múltiples idiomas 