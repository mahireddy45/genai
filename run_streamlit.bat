@echo off

REM Multimodal RAG Streamlit Application Launcher

echo "🚀 Starting Multimodal RAG Streamlit Application..."
echo ""

REM Check if streamlit is installed
IF ! "command" "-v" "streamlit" (
  echo "❌ Streamlit is not installed."
  echo "📦 Installing dependencies..."
  pip "install" "-r" "requirements.txt"
)

@REM REM Check if OpenAI API key is set
@REM IF "-z" "%OPENAI_API_KEY%" (
@REM   echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set."
@REM   echo "   You can enter it in the app sidebar instead."
@REM   echo ""
@REM )

REM Run the Streamlit app
echo "🌐 Launching application..."
echo "   The app will open in your default browser."
echo "   If not, navigate to: http://localhost:8501"
echo ""
streamlit "run" "rag_app.py"