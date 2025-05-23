import os
import json
import csv
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.supported_extensions = {'.pdf', '.json', '.csv'}

    def process_file(self, file_path: str) -> List[Document]:
        """Procesa un archivo individual."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                return loader.load()
            elif file_ext == '.json':
                return self.load_json(file_path)
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
                return loader.load()
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_ext}")

        except Exception as e:
            logger.error(f"Error al procesar {file_path}: {str(e)}")
            raise

    def load_json(self, file_path: str) -> List[Document]:
        """Carga un archivo JSON y lo divide en chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir el JSON a texto
            if isinstance(data, dict):
                text = json.dumps(data, indent=2, ensure_ascii=False)
            elif isinstance(data, list):
                text = "\n".join([json.dumps(item, indent=2, ensure_ascii=False) for item in data])
            else:
                text = str(data)
            
            # Crear un documento y dividirlo en chunks
            doc = Document(page_content=text, metadata={"source": file_path})
            return self.text_splitter.split_documents([doc])
        except Exception as e:
            logger.error(f"Error al procesar JSON {file_path}: {str(e)}")
            raise

    def process_directory(self, directory: str) -> List[Document]:
        """Procesa todos los documentos en un directorio."""
        if not os.path.exists(directory):
            logger.error(f"El directorio {directory} no existe")
            return []

        documents = []
        logger.info(f"Procesando directorio: {directory}")
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in self.supported_extensions:
                    try:
                        logger.info(f"Procesando archivo: {filename}")
                        docs = self.process_file(file_path)
                        documents.extend(docs)
                        logger.info(f"Archivo {filename} procesado exitosamente")
                    except Exception as e:
                        logger.error(f"Error procesando {filename}: {str(e)}")
                        continue

        logger.info(f"Total de documentos procesados: {len(documents)}")
        return documents 