@echo off
rem PreCue for Windows: double-click this file to start. A page opens in your browser.
rem You do not need Python: the first start gets uv, a small installer, which downloads Python
rem and the libraries PreCue needs (a few minutes, once). After that it opens in seconds.
cd /d "%~dp0"
set UV_VERSION=0.13.6

rem uv: installs Python and PreCue's libraries. Use the one on this PC, or get it once.
set "UV=uv"
where uv >nul 2>&1 || set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not "%UV%"=="uv" if not exist "%UV%" (
  echo Getting uv %UV_VERSION%, the installer PreCue uses ^(once^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_NO_MODIFY_PATH=1; irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex" || goto failed
)

rem Install (or update) the libraries, only when requirements.txt changed since last time.
set STAMP=.venv\precue-requirements.txt
set SETUP=0
if not exist .venv\Scripts\python.exe set SETUP=1
if not exist %STAMP% set SETUP=1
if exist %STAMP% fc /b requirements.txt %STAMP% >nul || set SETUP=1
if "%SETUP%"=="1" (
  echo Setting up PreCue. The first time takes a few minutes...
  if not exist .venv\Scripts\python.exe "%UV%" venv --python 3.12 .venv || goto failed
  "%UV%" pip install --python .venv\Scripts\python.exe -r requirements.txt || goto failed
  copy /y requirements.txt %STAMP% >nul
)

if not exist .env copy .env.example .env >nul
if defined PRECUE_NO_BROWSER (.venv\Scripts\python.exe -m scripts.app --no-browser) else (.venv\Scripts\python.exe -m scripts.app)
exit /b 0

:failed
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1
