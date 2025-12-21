@echo off
setlocal enabledelayedexpansion

REM --- go to project root ---
cd /d "C:\Users\kamal\AppData\Local\Programs\Python\Python312\precise predictor_codex" || exit /b 1

REM --- always use python 3.12 ---
set PY=py -3.12

%PY% run_minimal_console.py || exit /b 1

REM --- Auto commit and push changes ---
call tools\git_autopush.bat start

REM === auto-push rewritten artefacts ===
call tools\git_autopush_only.cmd

pause
endlocal
