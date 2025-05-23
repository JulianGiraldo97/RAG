from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class VectorStore:
    def __init__(self, use_openai: bool = True):
        if use_openai:
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]):
        """Crea un vector store a partir de los documentos."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def save_vector_store(self, directory: str):
        """Guarda el vector store en disco."""
        if self.vector_store:
            self.vector_store.save_local(directory)

    def load_vector_store(self, directory: str):
        """Carga un vector store desde disco."""
        self.vector_store = FAISS.load_local(directory, self.embeddings, allow_dangerous_deserialization=True)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Realiza una búsqueda por similitud."""
        if not self.vector_store:
            raise ValueError("Vector store no inicializado")
        return self.vector_store.similarity_search(query, k=k) 