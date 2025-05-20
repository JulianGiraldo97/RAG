from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .document_processor import DocumentProcessor
from .vector_store import VectorStore

class RAGSystem:
    def __init__(self, use_openai: bool = True):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(use_openai=use_openai)
        
        if use_openai:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0
            )
        else:
            self.llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )

        # Template para el prompt
        template = """Eres un asistente financiero experto. Usa la siguiente información para responder la pregunta.
        Si no sabes la respuesta, di que no tienes suficiente información.

        Información del contexto:
        {context}

        Pregunta: {question}

        Respuesta:"""

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
        return self.chain.invoke(question) 