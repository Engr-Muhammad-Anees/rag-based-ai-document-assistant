"""
RAG Engine using Google Gemini API + LangChain (Latest, Stable)
"""

import os
import config

# Gemini
import google.generativeai as genai

# LangChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage


class RAGManager:
    def __init__(self):
        #Gemini Setup
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(" GEMINI_API_KEY is missing")

        genai.configure(api_key=self.api_key)
        self.model_name = config.GEMINI_MODEL  # e.g. gemini-1.5-flash

        #Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        #Vector DB
        self.vector_db = None
        os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)

        # Chat Memory
        self.chat_history = []

        print(" RAGManager initialized successfully")

    # Document Ingestion

    def add_document(self, file_path: str):
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )
            chunks = splitter.split_documents(documents)

            if not chunks:
                raise ValueError("No text chunks generated")

            if self.vector_db is None:
                self.vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=config.VECTOR_DB_DIR,
                )
            else:
                self.vector_db.add_documents(chunks)

            print(f" Added {len(chunks)} chunks from {file_path}")

        except Exception as e:
            print(f" Error adding document: {e}")
            raise
    
    # Gemini Call

    def _call_llm(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=512,
                ),
            )
            return response.text.strip()
        except Exception as e:
            return f" Gemini API Error: {e}"


 # RAG Query

    def query(self, question: str) -> str:
        if self.vector_db is None:
            return " Please upload documents first."

        try:
            #  NEW LangChain-safe retriever call
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)

            if not docs:
                return " No relevant information found in the documents."

            context = "\n\n".join(doc.page_content for doc in docs)

            # Build history
            history_text = ""
            for msg in self.chat_history[-6:]:
                if isinstance(msg, HumanMessage):
                    history_text += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"

            prompt = f"""
You are a helpful AI assistant.
Answer strictly using the provided context.

Context:
{context}

Chat History:
{history_text}

Human: {question}
Assistant:
"""

            answer = self._call_llm(prompt)

            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

            return answer

        except Exception as e:
            error_msg = f" Error processing query: {e}"
            print(error_msg)
            return error_msg
