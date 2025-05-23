from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import RAGSystem

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize RAG system
rag = RAGSystem(use_openai=True)

# Directorios de datos
domain_knowledge_dir = "domain_knowledge"
vector_store_dir = "vector_store"

# Verificar si existe el vector store
if os.path.exists(vector_store_dir):
    print("Cargando vector store existente...")
    rag.load_vector_store(vector_store_dir)
else:
    print("Procesando documentos y creando vector store...")
    rag.process_documents(domain_knowledge_dir)
    rag.save_vector_store(vector_store_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get response from RAG system
        response = rag.query(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 