"# RAG_CHATBOT" 

Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using Ollama models.
Users can upload PDFs, and the system automatically generates:
- 📑 Summaries of the content
- ❓ MCQs (Multiple Choice Questions)
- 📝 Open-ended questions
This makes it ideal for educational use cases, exam preparation, and interactive learning.

✨ Features
- Upload PDF documents for processing
- Extract text and chunk it for retrieval
- Use Ollama models for context-aware question generation
- Generate MCQs with options and correct answers
- Generate descriptive questions for deeper understanding
- REST API + simple UI for interaction

🛠️ Tech Stack
- Backend: FastAPI / Flask (Python)
- LLM: Ollama models (e.g., llama2, mistral)
- Vector Store: FAISS / ChromaDB
- Frontend: Simple HTML/JS or Streamlit (optional)
- File Handling: PyPDF2 / pdfplumber
