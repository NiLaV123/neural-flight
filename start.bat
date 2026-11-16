@echo off
rem One command to set everything up (first time only) and open the Neural Flight Hub on Windows.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating .venv and installing requirements...
  py -3 -m venv .venv || python -m venv .venv || goto :nopython
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" setup_env.py || goto :fail
)
".venv\Scripts\python.exe" hub.py %*
goto :eof

:nopython
echo Python 3 was not found. Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH") and run this again.
pause
goto :eof

:fail
echo Installing the requirements failed. See the messages above.
pause
