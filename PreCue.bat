@echo off
rem PreCue for Windows: double-click this file to start. A page opens in your browser.
rem The first time, it sets itself up (a few minutes: it downloads the libraries PreCue needs).
cd /d "%~dp0"

set PYTHON=
for %%v in (3.13 3.12 3.11) do (
  if not defined PYTHON py -%%v -c "import sys" >nul 2>&1 && set PYTHON=py -%%v
)
if not defined PYTHON (
  echo PreCue needs Python 3.11 or newer. Get it from https://www.python.org/downloads/ and double-click again.
  pause
  exit /b 1
)

rem Install (or update) the libraries, only when requirements.txt changed since last time.
set STAMP=.venv\precue-requirements.txt
set SETUP=0
if not exist .venv\Scripts\python.exe set SETUP=1
if not exist %STAMP% set SETUP=1
if exist %STAMP% fc /b requirements.txt %STAMP% >nul || set SETUP=1
if "%SETUP%"=="1" (
  echo Setting up PreCue. The first time takes a few minutes...
  %PYTHON% -m venv .venv || goto failed
  .venv\Scripts\python.exe -m pip install --quiet --upgrade pip || goto failed
  .venv\Scripts\python.exe -m pip install --quiet -r requirements.txt || goto failed
  copy /y requirements.txt %STAMP% >nul
)

if not exist .env copy .env.example .env >nul
.venv\Scripts\python.exe -m scripts.app
exit /b 0

:failed
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1
