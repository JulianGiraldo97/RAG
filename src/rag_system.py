from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

class RAGSystem:
    def __init__(self, use_openai: bool = True):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(use_openai=use_openai)
        
        if use_openai:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=500,
                request_timeout=30
            )
        else:
            self.llm = HuggingFaceEndpoint(
                repo_id="google/flan-ul2",
                temperature=0.7,
                max_new_tokens=512
            )

        # Template para el prompt optimizado
        template = """Eres un asistente financiero especializado en finanzas agrícolas.
        Responde la pregunta de manera concisa y práctica usando SOLO la información del contexto.
        
        Reglas:
        1. Usa SOLO la información del contexto
        2. Si el contexto no es suficiente, indícalo brevemente
        3. Sé conciso pero informativo
        4. Enfócate en puntos clave y soluciones prácticas

        Contexto:
        {context}

        Pregunta: {question}

        Respuesta concisa:"""

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