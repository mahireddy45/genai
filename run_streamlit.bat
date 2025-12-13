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

REM Optionally set your OpenAI API Key here (replace the placeholder)
REM set OPENAI_API_KEY=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

REM Run the Streamlit app
echo  Launching application...
echo  The app will open in your default browser.
echo  If not, navigate to: http://localhost:8501
echo.
streamlit run main.py