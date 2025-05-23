import os
import logging
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, use_openai: bool = True):
        if use_openai:
            logger.info("Inicializando embeddings de OpenAI")
            self.embeddings = OpenAIEmbeddings()
        else:
            logger.info("Inicializando embeddings de HuggingFace")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]):
        """Crea el vector store a partir de los documentos."""
        if not documents:
            logger.warning("No se proporcionaron documentos para crear el vector store")
            return
            
        logger.info(f"Creando vector store con {len(documents)} documentos")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("Vector store creado exitosamente")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Realiza una búsqueda de similitud en el vector store."""
        if not self.vector_store:
            logger.error("El vector store no está inicializado")
            raise ValueError("El vector store no está inicializado")
            
        logger.info(f"Realizando búsqueda de similitud para: {query}")
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Encontrados {len(results)} documentos relevantes")
        return results

    def save_vector_store(self, directory: str):
        """Guarda el vector store en disco."""
        if not self.vector_store:
            logger.error("No hay vector store para guardar")
            return
            
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Guardando vector store en {directory}")
        self.vector_store.save_local(directory)
        logger.info("Vector store guardado exitosamente")

    def load_vector_store(self, directory: str):
        """Carga el vector store desde disco."""
        if not os.path.exists(directory):
            logger.error(f"El directorio {directory} no existe")
            return
            
        logger.info(f"Cargando vector store desde {directory}")
        self.vector_store = FAISS.load_local(
            directory, 
            self.embeddings,
            allow_dangerous_deserialization=True  # Permitir deserialización segura
        )
        logger.info("Vector store cargado exitosamente") 