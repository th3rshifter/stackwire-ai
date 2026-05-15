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

if not exist "venv\Scripts\activate.bat" (
  echo venv not found. Create it first:
  echo python -m venv venv
  echo venv\Scripts\activate.bat
  echo python -m pip install -r requirements.txt
  pause
  exit /b 1
)

call venv\Scripts\activate.bat

set STEALTHWIRE_HOST=0.0.0.0
set STEALTHWIRE_PORT=%SERVER_PORT%
set OLLAMA_URL=http://127.0.0.1:11434/api/chat
if "%STEALTHWIRE_MODE%"=="" set STEALTHWIRE_MODE=fast
if "%ANSWER_MODEL%"=="" set ANSWER_MODEL=qwen2.5-coder:7b
if "%RECOVERY_MODEL%"=="" set RECOVERY_MODEL=qwen2.5-coder:7b
if "%OLLAMA_NUM_CTX%"=="" set OLLAMA_NUM_CTX=2048
if "%OLLAMA_RECOVERY_NUM_PREDICT%"=="" set OLLAMA_RECOVERY_NUM_PREDICT=180
if "%OLLAMA_ANSWER_NUM_PREDICT%"=="" set OLLAMA_ANSWER_NUM_PREDICT=220
if "%WHISPER_DEVICE%"=="" set WHISPER_DEVICE=cuda
if "%WHISPER_COMPUTE_TYPE%"=="" set WHISPER_COMPUTE_TYPE=float16
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

if not "%SERVER_IP%"=="" (
  set PUBLIC_IP=%SERVER_IP%
) else (
  for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4"') do (
    set PUBLIC_IP=%%A
    goto ip_found
  )
)

:ip_found
set PUBLIC_IP=%PUBLIC_IP: =%
if "%PUBLIC_IP%"=="" set PUBLIC_IP=THIS_PC_IP
set NO_PROXY=%NO_PROXY%,%PUBLIC_IP%
set no_proxy=%no_proxy%,%PUBLIC_IP%

echo.
echo StealthWire API:
echo   http://%PUBLIC_IP%:%STEALTHWIRE_PORT%
echo.

net session >nul 2>&1
if errorlevel 1 (
  echo Firewall rule was not changed because this is not Administrator.
  echo If laptop cannot connect, run this once as Administrator:
  echo   netsh advfirewall firewall add rule name="StealthWire API %STEALTHWIRE_PORT%" dir=in action=allow protocol=TCP localport=%STEALTHWIRE_PORT%
) else (
  netsh interface portproxy delete v4tov4 listenaddress=%PUBLIC_IP% listenport=8001 >nul 2>&1
  netsh advfirewall firewall add rule name="StealthWire API %STEALTHWIRE_PORT%" dir=in action=allow protocol=TCP localport=%STEALTHWIRE_PORT% >nul 2>&1
  echo Firewall rule for TCP %STEALTHWIRE_PORT% is ready.
)

:check_ollama
echo.
echo Checking Ollama on 127.0.0.1:11434...
curl.exe --noproxy "*" -s http://127.0.0.1:11434/api/tags >nul
if errorlevel 1 (
  echo Ollama is not available. Start Ollama, then this script will continue.
  timeout /t 5 /nobreak >nul
  goto check_ollama
)

ollama list | findstr /i "qwen2.5-coder:7b" >nul
if errorlevel 1 (
  echo qwen2.5-coder:7b is not installed. Pulling model...
  ollama pull qwen2.5-coder:7b
  if errorlevel 1 (
    echo Failed to pull qwen2.5-coder:7b.
    pause
    exit /b 1
  )
)

echo.
echo Start laptop with:
echo   set STEALTHWIRE_API_URL=http://%PUBLIC_IP%:%STEALTHWIRE_PORT%
echo   python -m app.desktop
echo.
echo Starting server...
echo.

python -m app.main

echo.
echo Server stopped with exit code %errorlevel%.
pause
exit /b 0

:load_config
for /f "usebackq eol=# tokens=1,* delims==" %%A in (%1) do (
  if not "%%A"=="" set "%%A=%%B"
)
exit /b 0
