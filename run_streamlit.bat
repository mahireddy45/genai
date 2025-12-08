@echo off

REM Multimodal RAG Streamlit Application Launcher

echo Starting Multimodal RAG Streamlit Application...
echo.

REM Check if streamlit is installed
where streamlit >nul 2>nul
IF ERRORLEVEL 1 (
  echo Streamlit is not installed.
  echo Installing dependencies...
  pip install -r requirements.txt
)

REM Check if OpenAI API key is set
IF "%OPENAI_API_KEY%"=="" (
  echo   Warning: OPENAI_API_KEY environment variable is not set.
  echo   You can enter it in the app sidebar instead.
  echo.
)

REM Run the Streamlit app
echo  Launching application...
echo  The app will open in your default browser.
echo  If not, navigate to: http://localhost:8501
echo.
streamlit run rag_app.py