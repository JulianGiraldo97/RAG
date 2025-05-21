from typing import List
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import pandas as pd

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Carga un documento PDF y lo divide en chunks."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def load_json(self, file_path: str) -> List[Document]:
        """Carga un archivo JSON y lo divide en chunks."""
        # Cargar el JSON
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

    def load_csv(self, file_path: str) -> List[Document]:
        """Carga un archivo CSV y lo divide en chunks."""
        # Cargar el CSV
        df = pd.read_csv(file_path)
        
        # Convertir el DataFrame a texto
        text = df.to_string(index=False)
        
        # Crear un documento y dividirlo en chunks
        doc = Document(page_content=text, metadata={"source": file_path})
        return self.text_splitter.split_documents([doc])

    def process_directory(self, directory_path: str) -> List[Document]:
        """Procesa todos los archivos soportados en un directorio."""
        import os
        all_documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if filename.endswith('.pdf'):
                documents = self.load_pdf(file_path)
            elif filename.endswith('.json'):
                documents = self.load_json(file_path)
            elif filename.endswith('.csv'):
                documents = self.load_csv(file_path)
            else:
                continue
                
            all_documents.extend(documents)
        
        return all_documents 