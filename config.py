import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")