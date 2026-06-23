@echo off
REM ==========================================
REM OpsMind RAG Demo - Quick Start
REM ==========================================

echo.
echo === OpsMind RAG Demo ===
echo.

REM Check if data exists
python -c "from app.retrieval.vector_store import VectorStore; c = VectorStore().count; exit(0 if c > 0 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] No indexed data found. Running ingestion...
    echo [INFO] This may take 2-3 minutes on first run
    python scripts\ingest.py
    if %errorlevel% neq 0 (
        echo [ERROR] Ingestion failed!
        pause
        exit /b 1
    )
) else (
    echo [OK] Vector store has data, skipping ingestion.
)

echo.
echo [START] Starting backend on http://localhost:8000
start "OpsMind Backend" cmd /k "uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app --reload-dir scripts"

echo [START] Starting frontend on http://localhost:5173
start "OpsMind Frontend" cmd /k "cd frontend && npx vite --host"

echo.
echo ==========================================
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/api/docs
echo   Frontend: http://localhost:5173
echo   Attu GUI: http://localhost:8001
echo ==========================================
echo.
echo Press any key to stop all services...
pause >nul

taskkill /FI "WINDOWTITLE eq OpsMind Backend*" 2>nul
taskkill /FI "WINDOWTITLE eq OpsMind Frontend*" 2>nul
echo All services stopped.
