# RAG-Based AI Document Assistant

## Task Description
This project implements a Retrieval-Augmented Generation (RAG) based AI system that extends a conversational assistant using LangChain. The system allows users to upload documents (PDF, DOCX, TXT), convert them into searchable embeddings, store them in a vector database, and retrieve relevant content to generate accurate, context-aware answers using an API-based Large Language Model (LLM).

The assistant maintains conversation context and provides responses grounded strictly in the uploaded document content.
---

## Project Objectives
- Accept user-uploaded documents through a Gradio interface
- Extract and preprocess text from multiple document formats
- Convert document text into embeddings
- Store embeddings in a vector database
- Retrieve relevant document chunks during querying
- Generate accurate answers using an API-based LLM
- Maintain conversational memory
- Ensure modular and extensible system design
---

## System Architecture
User → Gradio Interface  
→ Document Upload  
→ Text Extraction  
→ Chunking  
→ Embedding Generation  
→ Vector Database (ChromaDB)  
→ Retriever (Top-K Chunks)  
→ Prompt + Chat History  
→ Gemini API  
→ AI Response

---
## RAG Pipeline Workflow
### 1. Document Ingestion & Preprocessing
- Supports PDF, DOCX, and TXT file uploads
- Uses LangChain document loaders to extract raw text
- Splits text into manageable chunks using RecursiveCharacterTextSplitter

### 2. Vector Database Construction
- Generates embeddings using Sentence Transformers
- Stores embeddings in ChromaDB
- Allows incremental addition of multiple documents

### 3. Retrieval-Augmented Generation
- Retrieves top-k relevant document chunks for each user query
- Injects retrieved context into the LLM prompt
- Uses Google Gemini API for answer generation
- Ensures responses are grounded in document context

### 4. Conversational Memory
- Maintains multi-turn conversation context
- Improves response relevance for follow-up questions

### 5. Gradio Interface Integration
- File upload section for documents
- Button to build the knowledge base
- Real-time chat interface
- Clear display of user queries and AI responses

---
### 6. Technologies and Libraries Used
- Python
- LangChain
- Google Gemini API
- ChromaDB
- Sentence Transformers
- Gradio (v6.2.2)
- dotenv
---
## Project Structure
RAG-based AI assistant project/
│
├── src/
│ ├── app.py # Gradio UI and application entry point
│ ├── rag_engine.py # Core RAG logic (ingestion, retrieval, generation)
│ ├── config.py # Environment configuration
│
├── vector_db/ # Persistent vector database
├── .env # API keys and environment variables
├── requirements.txt
└── README.md
---
## Example Queries
- "Summarize the uploaded document"
- "What is the main objective of the proposal?"
- "Explain Phase 2 of the methodology"
- "Which tools are used for OCR and why?"
- "What is the final output of this project?"
---
## Modularity and Extensibility
The system is designed to be modular, allowing:
- Replacement of LLMs (Gemini → Llama → Mistral)
- Replacement or upgrading of embedding models
- Switching vector databases (ChromaDB → FAISS)
- Extension of the UI or backend without major refactoring
---

## Challenges and Limitations
- Performance depends on document quality and chunking strategy
- Large documents may increase memory and computation requirements
- API-based LLM usage is subject to rate limits and costs
- OCR accuracy may vary for low-quality slides or images
---
## Conclusion
This project demonstrates a complete and modular Retrieval-Augmented Generation pipeline capable of document-based conversational question answering. By combining LangChain, vector databases, and an API-based LLM, the system delivers accurate, context-aware responses grounded in user-provided documents.

