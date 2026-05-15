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

if not exist "venv\Scripts\activate.bat" (
  echo venv not found. Create it first:
  echo python -m venv venv
  echo venv\Scripts\activate.bat
  echo python -m pip install -r requirements.txt
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

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
