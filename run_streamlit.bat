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

REM Set OpenAI API Key
set OPENAI_API_KEY=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

REM Run the Streamlit app
echo  Launching application...
echo  The app will open in your default browser.
echo  If not, navigate to: http://localhost:8501
echo.
streamlit run main.py