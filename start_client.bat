@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if "%~1"=="" (
  set /p SERVER_IP=Enter StealthWire server IP:
) else (
  set SERVER_IP=%~1
)

if "%SERVER_IP%"=="" (
  echo Server IP is required.
  pause
  exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
  echo venv not found. Create it first:
  echo python -m venv venv
  echo venv\Scripts\activate.bat
  echo python -m pip install -r requirements.txt
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

set STEALTHWIRE_API_URL=http://%SERVER_IP%:8000
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
