import os
from dotenv import load_dotenv
from src.rag_system import RAGSystem

def main():
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Se requiere una API key de OpenAI")
    
    # Inicializar sistema RAG con OpenAI
    rag = RAGSystem(use_openai=True)
    
    # Directorios
    data_dir = "data"
    domain_knowledge_dir = "domain_knowledge"
    vector_store_dir = "vector_store"
    
    # Verificar si el vector store ya existe
    if os.path.exists(vector_store_dir):
        print("Cargando vector store existente...")
        rag.load_vector_store(vector_store_dir)
    else:
        print("Procesando documentos...")
        rag.process_documents(data_dir, domain_knowledge_dir)
        print("Guardando vector store...")
        rag.save_vector_store(vector_store_dir)
    
    # Interfaz de usuario simple
    print("\nSistema RAG Financiero")
    print("Escribe 'salir' para terminar")
    
    while True:
        question = input("\nTu pregunta: ")
        if question.lower() == 'salir':
            break
            
        try:
            answer = rag.query(question)
            print("\nRespuesta:", answer)
        except Exception as e:
            print(f"Error al procesar la pregunta: {str(e)}")

if __name__ == "__main__":
    main() 