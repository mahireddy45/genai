#!/bin/bash

# Multimodal RAG Streamlit Application Launcher

echo "🚀 Starting Multimodal RAG Streamlit Application..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed."
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set."
    echo "   You can enter it in the app sidebar instead."
    echo ""
fi

# Run the Streamlit app
echo "🌐 Launching application..."
echo "   The app will open in your default browser."
echo "   If not, navigate to: http://localhost:8501"
echo ""
streamlit run rag_app.py