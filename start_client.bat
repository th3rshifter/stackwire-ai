@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set CONFIG_FILE=%~dp0stealthwire.local.env
set SERVER_PORT=8000
if exist "%CONFIG_FILE%" (
  call :load_config "%CONFIG_FILE%"
)
if not "%~1"=="" set SERVER_IP=%~1
if not "%~2"=="" set SERVER_PORT=%~2

if "%SERVER_IP%"=="" (
  set /p SERVER_IP=Enter StealthWire server IP:
)

if "%SERVER_IP%"=="" (
  echo Server IP is required.
  pause
  exit /b 1
)

call :ensure_venv
if errorlevel 1 exit /b 1

call venv\Scripts\activate.bat

echo.
echo Checking Python dependencies...
python -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Failed to install Python dependencies from requirements.txt.
  pause
  exit /b 1
)

python -c "import requests, PySide6, sounddevice, soundcard, numpy" >nul
if errorlevel 1 (
  echo Dependency import check failed. Reinstalling requirements...
  python -m pip install --disable-pip-version-check --force-reinstall -r requirements.txt
  if errorlevel 1 (
    echo Failed to repair Python dependencies.
    pause
    exit /b 1
  )
)

set STEALTHWIRE_API_URL=http://%SERVER_IP%:%SERVER_PORT%
set STEALTHWIRE_REMOTE_STT=1
set STT_BACKEND=whisper
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
set NO_PROXY=%SERVER_IP%,127.0.0.1,localhost
set no_proxy=%SERVER_IP%,127.0.0.1,localhost

echo.
echo Checking %STEALTHWIRE_API_URL%/status ...
curl.exe --noproxy "*" -f -sS "%STEALTHWIRE_API_URL%/status"
if errorlevel 1 (
  echo.
  echo Cannot reach server.
  echo Check server IP, Windows Firewall, and that start_server.bat is running on the main PC.
  pause
  exit /b 1
)

echo.
echo Starting desktop client connected to %STEALTHWIRE_API_URL%
echo.

python -m app.desktop
exit /b %errorlevel%

:load_config
for /f "usebackq eol=# tokens=1,* delims==" %%A in (%1) do (
  if not "%%A"=="" set "%%A=%%B"
)
exit /b 0

:ensure_venv
set "PYTHON_LAUNCHER=python"
where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo Python was not found. Install Python 3.11+ and run this script again.
    pause
    exit /b 1
  )
  set "PYTHON_LAUNCHER=py -3"
)

if not exist "venv\Scripts\python.exe" (
  echo Creating virtual environment in %CD%\venv ...
  %PYTHON_LAUNCHER% -m venv venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)
exit /b 0
