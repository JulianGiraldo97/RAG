import os
import logging
from dotenv import load_dotenv
from src.rag_system import RAGSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY no encontrada en el archivo .env")
        return

    try:
        # Inicializar el sistema RAG
        logger.info("Inicializando sistema RAG")
        rag = RAGSystem(use_openai=True)
        
        # Directorios de datos
        domain_knowledge_dir = "domain_knowledge"
        vector_store_dir = "vector_store"
        
        # Verificar si existe el vector store
        if os.path.exists(vector_store_dir):
            logger.info("Cargando vector store existente...")
            rag.load_vector_store(vector_store_dir)
        else:
            logger.info("Procesando documentos y creando vector store...")
            rag.process_documents(domain_knowledge_dir)
            rag.save_vector_store(vector_store_dir)
        
        print("\nSistema RAG Financiero")
        print("Escribe 'salir' para terminar\n")
        
        while True:
            question = input("Tu pregunta: ").strip()
            
            if question.lower() == 'salir':
                break
                
            if not question:
                print("Por favor, ingresa una pregunta válida.")
                continue
                
            try:
                logger.info(f"Procesando pregunta: {question}")
                response = rag.query(question)
                print("\nRespuesta:", response, "\n")
            except Exception as e:
                logger.error(f"Error al procesar la pregunta: {str(e)}")
                print(f"\nError: {str(e)}\n")
                
    except Exception as e:
        logger.error(f"Error en la aplicación: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 