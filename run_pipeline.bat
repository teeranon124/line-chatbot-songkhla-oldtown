@echo off
chcp 65001 > nul
echo ================================================================================
echo  🏛️ Songkhla Old Town Assistant - AI Pipeline Launcher
echo ================================================================================

set PYTHON_EXE=..\venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" scripts\run_pipeline.py %*
