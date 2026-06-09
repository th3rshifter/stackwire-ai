@echo off
chcp 65001 >nul
setlocal EnableExtensions

title StackWire Launcher
cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "CONFIG_FILE=%ROOT_DIR%stackwire.local.env"
set "SERVER_PORT=8000"

if exist "%CONFIG_FILE%" call :load_config "%CONFIG_FILE%"

set "MODE=%~1"
if /i "%MODE%"=="help" (
  call :usage
  goto done
)
if /i "%MODE%"=="/?" (
  call :usage
  goto done
)
if /i "%MODE%"=="-h" (
  call :usage
  goto done
)
if /i "%MODE%"=="server" (
  if not "%~2"=="" if not "%~2"=="-" set "SERVER_IP=%~2"
  if not "%~3"=="" set "SERVER_PORT=%~3"
  call :run_server
  goto done
)
if /i "%MODE%"=="client" (
  if not "%~2"=="" set "SERVER_IP=%~2"
  if not "%~3"=="" set "SERVER_PORT=%~3"
  call :run_client
  goto done
)
if /i "%MODE%"=="both" (
  if not "%~2"=="" set "SERVER_PORT=%~2"
  call :run_both
  goto done
)
if /i "%MODE%"=="local" (
  if not "%~2"=="" set "SERVER_PORT=%~2"
  call :run_both
  goto done
)

:menu
cls
call :banner
echo.
echo   [1] Server
echo       Run StackWire API on this PC
echo.
echo   [2] Client
echo       Run desktop client and connect to a server
echo.
echo   [3] Server + local client
echo       One-PC mode: start API, wait until ready, then open client
echo.
echo   [4] Exit
echo.
set "CHOICE="
set /p "CHOICE=Select: "

if "%CHOICE%"=="1" (
  call :run_server
  goto done
)
if "%CHOICE%"=="2" (
  call :run_client
  goto done
)
if "%CHOICE%"=="3" (
  call :run_both
  goto done
)
if "%CHOICE%"=="4" goto done

echo.
echo Unknown option: %CHOICE%
pause
goto menu

:banner
echo  ============================================================
echo    StackWire launcher
echo  ============================================================
exit /b 0

:usage
call :banner
echo.
echo Usage:
echo   start_stackwire.bat
echo   start_stackwire.bat server [server_ip] [port]
echo   start_stackwire.bat client [server_ip] [port]
echo   start_stackwire.bat both [port]
echo.
echo Notes:
echo   server_ip can be "-" in server mode to auto-detect LAN IP.
echo   both starts the server in a separate window and opens a local client.
exit /b 0

:run_server
call :section "SERVER"
call :prepare_python server
if errorlevel 1 exit /b 1

call :configure_server_env
call :resolve_models

call :detect_public_ip
set "PUBLIC_IP=%PUBLIC_IP: =%"
if "%PUBLIC_IP%"=="" set "PUBLIC_IP=THIS_PC_IP"
if not "%PUBLIC_IP%"=="THIS_PC_IP" call :save_server_ip

set "NO_PROXY=%NO_PROXY%,%PUBLIC_IP%"
set "no_proxy=%no_proxy%,%PUBLIC_IP%"

echo.
echo API endpoint:
echo   http://%PUBLIC_IP%:%STACKWIRE_PORT%
echo.

call :ensure_firewall_rule
call :wait_for_ollama
if errorlevel 1 exit /b 1

call :ensure_ollama_model "%ANSWER_MODEL%"
if errorlevel 1 exit /b 1
if /i not "%RECOVERY_MODEL%"=="%ANSWER_MODEL%" (
  call :ensure_ollama_model "%RECOVERY_MODEL%"
  if errorlevel 1 exit /b 1
)
if /i "%VISION_MODEL%"=="%ANSWER_MODEL%" goto vision_model_ready
if /i "%VISION_MODEL%"=="%RECOVERY_MODEL%" goto vision_model_ready
call :ensure_ollama_model "%VISION_MODEL%"
if errorlevel 1 exit /b 1
:vision_model_ready

echo.
echo Client command for another PC:
echo   start_stackwire.bat client %PUBLIC_IP% %STACKWIRE_PORT%
echo.
echo Starting server. Press Ctrl+C to stop.
echo.
"%VENV_PYTHON%" -m app.main

echo.
echo Server stopped with exit code %errorlevel%.
pause
exit /b 0

:run_client
call :section "CLIENT"
call :prepare_python client
if errorlevel 1 exit /b 1

if "%SERVER_IP%"=="" (
  set /p "SERVER_IP=Server IP: "
)
if "%SERVER_IP%"=="" (
  echo Server IP is required.
  pause
  exit /b 1
)
if "%SERVER_PORT%"=="" set "SERVER_PORT=8000"

call :configure_client_env

echo.
echo Checking %STACKWIRE_API_URL%/status ...
curl.exe --noproxy "*" -f -sS "%STACKWIRE_API_URL%/status"
if errorlevel 1 (
  echo.
  echo Cannot reach server at %STACKWIRE_API_URL%.
  echo Check that the server is running and Windows Firewall allows TCP %SERVER_PORT%.
  pause
  exit /b 1
)

echo.
echo Starting desktop client connected to %STACKWIRE_API_URL%
echo.

set "PYTHONW_EXE=%ROOT_DIR%venv\Scripts\pythonw.exe"
set "PYTHON_EXE=%ROOT_DIR%venv\Scripts\python.exe"

if exist "%PYTHONW_EXE%" (
  start "StackWire Client" "%PYTHONW_EXE%" -m app.desktop
) else (
  start "StackWire Client" "%PYTHON_EXE%" -m app.desktop
)

if "%STACKWIRE_KEEP_CMD%"=="1" pause
exit /b 0

:run_both
call :section "LOCAL MODE"
if "%SERVER_PORT%"=="" set "SERVER_PORT=8000"

echo Starting server in a separate window...
start "StackWire Server" cmd /k ""%~f0" server - %SERVER_PORT%"

echo.
echo Waiting for local API:
echo   http://127.0.0.1:%SERVER_PORT%/status
call :wait_for_server "http://127.0.0.1:%SERVER_PORT%/status"
if errorlevel 1 (
  echo.
  echo Server did not become ready in time.
  echo Check the "StackWire Server" window for details.
  pause
  exit /b 1
)

set "SERVER_IP=127.0.0.1"
call :run_client
exit /b %errorlevel%

:section
echo.
echo  ------------------------------------------------------------
echo   %~1
echo  ------------------------------------------------------------
exit /b 0

:prepare_python
set "PROFILE=%~1"
call :ensure_venv
if errorlevel 1 exit /b 1

call "%ROOT_DIR%venv\Scripts\activate.bat"
set "VENV_PYTHON=%ROOT_DIR%venv\Scripts\python.exe"

echo.
echo Checking Python dependencies...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies from requirements.txt.
  pause
  exit /b 1
)

if /i "%PROFILE%"=="server" (
  "%VENV_PYTHON%" -c "import requests, fastapi, uvicorn, PySide6, faster_whisper, sounddevice, pyaudiowpatch, numpy, rapidfuzz, rich" >nul
) else (
  "%VENV_PYTHON%" -c "import requests, PySide6, sounddevice, pyaudiowpatch, numpy, rapidfuzz, rich" >nul
)
if errorlevel 1 (
  echo Dependency import check failed. Reinstalling requirements...
  "%VENV_PYTHON%" -m pip install --disable-pip-version-check --force-reinstall -r requirements.txt
  if errorlevel 1 (
    echo Failed to repair Python dependencies.
    pause
    exit /b 1
  )
)
exit /b 0

:configure_server_env
set "STACKWIRE_HOST=0.0.0.0"
set "STACKWIRE_PORT=%SERVER_PORT%"
set "OLLAMA_URL=http://127.0.0.1:11434/api/chat"
if "%STACKWIRE_MODE%"=="" set "STACKWIRE_MODE=fast"
if "%ANSWER_MODE%"=="" set "ANSWER_MODE=normal"
if "%ANSWER_PROMPT_PROFILE%"=="" set "ANSWER_PROMPT_PROFILE=balanced"
if "%RECOVERY_LOCAL_FAST_PATH%"=="" set "RECOVERY_LOCAL_FAST_PATH=1"
if "%OLLAMA_NUM_CTX%"=="" set "OLLAMA_NUM_CTX=4096"
if "%OLLAMA_RECOVERY_NUM_PREDICT%"=="" set "OLLAMA_RECOVERY_NUM_PREDICT=160"
if "%OLLAMA_ANSWER_NUM_PREDICT%"=="" set "OLLAMA_ANSWER_NUM_PREDICT=950"
if "%OLLAMA_ARTIFACT_NUM_PREDICT%"=="" set "OLLAMA_ARTIFACT_NUM_PREDICT=1200"
if "%OLLAMA_EXPAND_NUM_PREDICT%"=="" set "OLLAMA_EXPAND_NUM_PREDICT=1200"
if "%OLLAMA_KEEP_ALIVE%"=="" set "OLLAMA_KEEP_ALIVE=30m"
if "%WHISPER_DEVICE%"=="" set "WHISPER_DEVICE=auto"
if "%WHISPER_COMPUTE_TYPE%"=="" set "WHISPER_COMPUTE_TYPE=auto"
if "%STT_LANGUAGE_MODE%"=="" set "STT_LANGUAGE_MODE=auto"
if "%WHISPER_LANGUAGE%"=="" set "WHISPER_LANGUAGE=auto"
if "%WHISPER_BEAM_SIZE%"=="" set "WHISPER_BEAM_SIZE=5"
if "%WHISPER_BEST_OF%"=="" set "WHISPER_BEST_OF=5"
if "%WHISPER_CHUNK_SECONDS%"=="" set "WHISPER_CHUNK_SECONDS=3.5"
if "%WHISPER_CHUNK_OVERLAP_SECONDS%"=="" set "WHISPER_CHUNK_OVERLAP_SECONDS=1.0"
if "%WHISPER_VAD_FILTER%"=="" set "WHISPER_VAD_FILTER=1"
if "%WHISPER_RETRY_WITHOUT_VAD%"=="" set "WHISPER_RETRY_WITHOUT_VAD=1"
if "%WHISPER_VAD_THRESHOLD%"=="" set "WHISPER_VAD_THRESHOLD=0.20"
if "%WHISPER_VAD_MIN_SPEECH_MS%"=="" set "WHISPER_VAD_MIN_SPEECH_MS=100"
if "%WHISPER_VAD_MIN_SILENCE_MS%"=="" set "WHISPER_VAD_MIN_SILENCE_MS=650"
if "%WHISPER_VAD_SPEECH_PAD_MS%"=="" set "WHISPER_VAD_SPEECH_PAD_MS=450"
if "%WHISPER_NO_SPEECH_THRESHOLD%"=="" set "WHISPER_NO_SPEECH_THRESHOLD=0.75"
if "%WHISPER_LOG_PROB_THRESHOLD%"=="" set "WHISPER_LOG_PROB_THRESHOLD=-2.0"
if "%WHISPER_COMPRESSION_RATIO_THRESHOLD%"=="" set "WHISPER_COMPRESSION_RATIO_THRESHOLD=3.0"
if "%WHISPER_REPETITION_PENALTY%"=="" set "WHISPER_REPETITION_PENALTY=1.08"
if "%WHISPER_NO_REPEAT_NGRAM_SIZE%"=="" set "WHISPER_NO_REPEAT_NGRAM_SIZE=3"
if "%WHISPER_HALLUCINATION_SILENCE_THRESHOLD%"=="" set "WHISPER_HALLUCINATION_SILENCE_THRESHOLD=1.0"
call :clear_proxy_env
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"
exit /b 0

:configure_client_env
set "STACKWIRE_API_URL=http://%SERVER_IP%:%SERVER_PORT%"
set "STACKWIRE_REMOTE_STT=1"
set "STT_BACKEND=whisper"
call :clear_proxy_env
set "NO_PROXY=%SERVER_IP%,127.0.0.1,localhost"
set "no_proxy=%SERVER_IP%,127.0.0.1,localhost"
exit /b 0

:clear_proxy_env
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
exit /b 0

:resolve_models
for /f "tokens=1,* delims==" %%A in ('"%VENV_PYTHON%" -c "from app.llm import MODEL, VISION_MODEL; from app.question_recovery import DEFAULT_MODEL; print('ANSWER_MODEL_RESOLVED=' + MODEL); print('RECOVERY_MODEL_RESOLVED=' + DEFAULT_MODEL); print('VISION_MODEL_RESOLVED=' + VISION_MODEL)"') do (
  set "%%A=%%B"
)
if "%ANSWER_MODEL%"=="" set "ANSWER_MODEL=%ANSWER_MODEL_RESOLVED%"
if "%RECOVERY_MODEL%"=="" set "RECOVERY_MODEL=%RECOVERY_MODEL_RESOLVED%"
if "%VISION_MODEL%"=="" set "VISION_MODEL=%VISION_MODEL_RESOLVED%"
exit /b 0

:detect_public_ip
set "PUBLIC_IP="
if not "%SERVER_IP%"=="" (
  ipconfig | findstr /l /c:"%SERVER_IP%" >nul
  if errorlevel 1 (
    echo Config SERVER_IP=%SERVER_IP% is not assigned to this PC.
    echo Detecting current LAN IP instead...
  ) else (
    set "PUBLIC_IP=%SERVER_IP%"
    exit /b 0
  )
)
call :detect_lan_ip
exit /b 0

:ensure_firewall_rule
net session >nul 2>&1
if errorlevel 1 (
  echo Firewall rule was not changed because this is not Administrator.
  echo If another PC cannot connect, run this once as Administrator:
  echo   netsh advfirewall firewall add rule name="StackWire API %STACKWIRE_PORT%" dir=in action=allow protocol=TCP localport=%STACKWIRE_PORT%
  exit /b 0
)
netsh advfirewall firewall add rule name="StackWire API %STACKWIRE_PORT%" dir=in action=allow protocol=TCP localport=%STACKWIRE_PORT% >nul 2>&1
echo Firewall rule for TCP %STACKWIRE_PORT% is ready.
exit /b 0

:wait_for_ollama
echo.
echo Checking Ollama on 127.0.0.1:11434...
curl.exe --noproxy "*" -s http://127.0.0.1:11434/api/tags >nul
if errorlevel 1 (
  echo Ollama is not available. Start Ollama, then this launcher will continue.
  timeout /t 5 /nobreak >nul
  goto wait_for_ollama
)
exit /b 0

:ensure_ollama_model
set "REQUIRED_MODEL=%~1"
if "%REQUIRED_MODEL%"=="" exit /b 0
ollama list | findstr /i /l /c:"%REQUIRED_MODEL%" >nul
if errorlevel 1 (
  echo %REQUIRED_MODEL% is not installed. Pulling model...
  ollama pull "%REQUIRED_MODEL%"
  if errorlevel 1 (
    echo Failed to pull %REQUIRED_MODEL%.
    pause
    exit /b 1
  )
)
exit /b 0

:wait_for_server
set "STATUS_URL=%~1"
for /l %%I in (1,1,60) do (
  curl.exe --noproxy "*" -f -sS "%STATUS_URL%" >nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 2 /nobreak >nul
)
exit /b 1

:load_config
for /f "usebackq eol=# tokens=1,* delims==" %%A in (%1) do (
  if not "%%A"=="" set "%%A=%%B"
)
exit /b 0

:save_server_ip
if not exist "%CONFIG_FILE%" exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=$env:CONFIG_FILE; $ip=$env:PUBLIC_IP; $lines=Get-Content -LiteralPath $p; if ($lines -match '^SERVER_IP=') { $lines=$lines -replace '^SERVER_IP=.*', ('SERVER_IP=' + $ip) } else { $lines=@('SERVER_IP=' + $ip)+$lines }; Set-Content -LiteralPath $p -Value $lines -Encoding ASCII"
exit /b 0

:detect_lan_ip
set "PUBLIC_IP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4" ^| findstr /v /c:"169.254." ^| findstr /v /c:"192.168.56."') do (
  set "PUBLIC_IP=%%A"
  goto detect_lan_ip_done
)
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4"') do (
  set "PUBLIC_IP=%%A"
  goto detect_lan_ip_done
)
:detect_lan_ip_done
exit /b 0

:ensure_venv
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo Existing virtual environment is broken: %CD%\venv
    echo Delete the venv folder, install Python 3.11+ from python.org, then run this launcher again.
    pause
    exit /b 1
  )
  exit /b 0
)

call :find_python
if errorlevel 1 exit /b 1

echo Creating virtual environment in %CD%\venv ...
"%PYTHON_EXE%" %PYTHON_ARGS% -m venv venv
if errorlevel 1 (
  echo Failed to create virtual environment.
  pause
  exit /b 1
)
exit /b 0

:find_python
set "PYTHON_EXE="
set "PYTHON_ARGS="
where.exe py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    exit /b 0
  )
)

for %%P in (
  "%LocalAppData%\Programs\Python\Python313\python.exe"
  "%LocalAppData%\Programs\Python\Python312\python.exe"
  "%LocalAppData%\Programs\Python\Python311\python.exe"
  "%ProgramFiles%\Python313\python.exe"
  "%ProgramFiles%\Python312\python.exe"
  "%ProgramFiles%\Python311\python.exe"
  "%ProgramFiles(x86)%\Python313\python.exe"
  "%ProgramFiles(x86)%\Python312\python.exe"
  "%ProgramFiles(x86)%\Python311\python.exe"
) do (
  if exist "%%~P" (
    call :try_python "%%~P"
    if not errorlevel 1 exit /b 0
  )
)

for %%C in (python python3) do (
  for /f "delims=" %%P in ('where.exe %%C 2^>nul') do (
    echo %%P | findstr /i /c:"\Microsoft\WindowsApps\" >nul
    if errorlevel 1 (
      call :try_python "%%P"
      if not errorlevel 1 exit /b 0
    )
  )
)

echo Python 3.11+ was not found outside Microsoft Store aliases.
echo Install Python from https://www.python.org/downloads/windows/ and enable "Add python.exe to PATH".
echo If Windows opens Microsoft Store when running python, disable:
echo   Settings ^> Apps ^> Advanced app settings ^> App execution aliases ^> python.exe / python3.exe
pause
exit /b 1

:try_python
set "PYTHON_TEST=%~1"
"%PYTHON_TEST%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "PYTHON_EXE=%PYTHON_TEST%"
set "PYTHON_ARGS="
exit /b 0

:done
endlocal & exit /b %ERRORLEVEL%
