from typing import List
import logging
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, use_openai: bool = True):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(use_openai=use_openai)
        
        if use_openai:
            logger.info("Inicializando modelo OpenAI GPT-3.5-turbo")
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=1000,  # Aumentado para permitir respuestas más completas
                request_timeout=60  # Aumentado el timeout
            )
        else:
            self.llm = HuggingFaceEndpoint(
                repo_id="google/flan-ul2",
                temperature=0.7,
                max_new_tokens=512
            )

        # Template para el prompt optimizado
        template = """Eres un asistente financiero especializado en finanzas agrícolas.
        Responde la pregunta de manera clara y estructurada usando la información del contexto.
        
        Reglas:
        1. Usa la información del contexto para responder
        2. Si el contexto no es suficiente, indícalo claramente
        3. Estructura tu respuesta en puntos clave
        4. Incluye ejemplos prácticos cuando sea relevante

        Contexto:
        {context}

        Pregunta: {question}

        Proporciona una respuesta estructurada:"""

        self.prompt = ChatPromptTemplate.from_template(template)

        # Cadena de procesamiento
        self.chain = (
            {"context": self.vector_store.similarity_search, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def process_documents(self, data_dir: str, domain_knowledge_dir: str = None):
        """Procesa los documentos y crea el vector store."""
        # Procesar documentos de datos
        documents = self.document_processor.process_directory(data_dir)
        
        # Procesar documentos de conocimiento del dominio si existen
        if domain_knowledge_dir:
            domain_docs = self.document_processor.process_directory(domain_knowledge_dir)
            documents.extend(domain_docs)
        
        # Crear vector store
        self.vector_store.create_vector_store(documents)

    def save_vector_store(self, directory: str):
        """Guarda el vector store en disco."""
        self.vector_store.save_vector_store(directory)

    def load_vector_store(self, directory: str):
        """Carga el vector store desde disco."""
        self.vector_store.load_vector_store(directory)

    def query(self, question: str) -> str:
        """Realiza una consulta al sistema RAG."""
        try:
            logger.info(f"Procesando pregunta: {question}")
            result = self.chain.invoke(question)
            logger.info("Respuesta generada exitosamente")
            return result
        except Exception as e:
            logger.error(f"Error al procesar la pregunta: {str(e)}")
            raise

    def process_documents(self, data_dir: str, domain_knowledge_dir: str = None):
        """Procesa los documentos y crea el vector store."""
        # Procesar documentos de datos
        documents = self.document_processor.process_directory(data_dir)
        
        # Procesar documentos de conocimiento del dominio si existen
        if domain_knowledge_dir:
            domain_docs = self.document_processor.process_directory(domain_knowledge_dir)
            documents.extend(domain_docs)
        
        # Crear vector store
        self.vector_store.create_vector_store(documents)

    def save_vector_store(self, directory: str):
        """Guarda el vector store en disco."""
        self.vector_store.save_vector_store(directory)

    def load_vector_store(self, directory: str):
        """Carga el vector store desde disco."""
        self.vector_store.load_vector_store(directory)

    def query(self, question: str) -> str:
        """Realiza una consulta al sistema RAG."""
        return self.chain.invoke(question) 