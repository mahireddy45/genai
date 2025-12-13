import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
# The .env file should be in the same directory as rag_app.py (project root)
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)  # override=True to ensure latest .env values are loaded
else:
    # Fallback to loading from current directory
    load_dotenv(override=True)

# OpenAI Configuration - Get from environment (already loaded by load_dotenv)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip('"\'')  # Strip quotes if present
# Explicitly set it back in os.environ so all modules can reliably access it
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not OPENAI_API_KEY:
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
