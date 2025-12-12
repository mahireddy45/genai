import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
# The .env file should be in the same directory as rag_app.py (project root)
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback to loading from current directory
    load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or "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" in OPENAI_API_KEY:
    import warnings
    warnings.warn(
        "\n" + "="*70 + "\n"
        "⚠️  WARNING: OPENAI_API_KEY is not configured!\n\n"
        "To use the RAG application, you need to:\n\n"
        "1. Get your API key from: https://platform.openai.com/api-keys\n"
        "2. Set it in .env file:\n"
        "   OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY\n"
        "3. Or set environment variable:\n"
        "   export OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY\n"
        "4. Restart the application\n\n"
        "Without a valid API key, document ingestion and chat will not work.\n"
        "="*70 + "\n",
        RuntimeWarning
    )
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))

# Vector Store Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))