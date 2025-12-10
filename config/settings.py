"""Configuration settings for the LangChain project"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "RAG Chatbot",
    "page_icon": "🤖",
    "layout": "centered",
}
