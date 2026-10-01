@echo off
cd /d "%~dp0"
rem serve.py adds this PC's Hugging Face token for gated repos; without Python the page still opens from file://
where py >nul 2>nul
if not errorlevel 1 (
  py -3 serve.py
  goto :eof
)
where python >nul 2>nul
if not errorlevel 1 (
  python serve.py
  goto :eof
)
start "" "%~dp0index.html"
