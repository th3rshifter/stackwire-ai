@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set CONFIG_FILE=%~dp0stackwire.local.env
set SERVER_PORT=8000
if exist "%CONFIG_FILE%" (
  call :load_config "%CONFIG_FILE%"
)
if not "%~1"=="" set SERVER_IP=%~1
if not "%~2"=="" set SERVER_PORT=%~2

if "%SERVER_IP%"=="" (
  set /p SERVER_IP=Enter StackWire server IP:
)

if "%SERVER_IP%"=="" (
  echo Server IP is required.
  pause
  exit /b 1
)

call :ensure_venv
if errorlevel 1 exit /b 1

call venv\Scripts\activate.bat
set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"

echo.
echo Checking Python dependencies...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Failed to install Python dependencies from requirements.txt.
  pause
  exit /b 1
)

"%VENV_PYTHON%" -c "import requests, PySide6, sounddevice, soundcard, numpy, rapidfuzz, rich" >nul
if errorlevel 1 (
  echo Dependency import check failed. Reinstalling requirements...
  "%VENV_PYTHON%" -m pip install --disable-pip-version-check --force-reinstall -r requirements.txt
  if errorlevel 1 (
    echo Failed to repair Python dependencies.
    pause
    exit /b 1
  )
)

set STACKWIRE_API_URL=http://%SERVER_IP%:%SERVER_PORT%
set STACKWIRE_REMOTE_STT=1
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
echo Checking %STACKWIRE_API_URL%/status ...
curl.exe --noproxy "*" -f -sS "%STACKWIRE_API_URL%/status"
if errorlevel 1 (
  echo.
  echo Cannot reach server at %STACKWIRE_API_URL%.
  echo On the main PC, check that start_server.bat is still running.
  echo If the server window says "Uvicorn running on http://0.0.0.0:%SERVER_PORT%", this is usually Windows Firewall or a wrong SERVER_IP.
  echo Run start_server.bat on the main PC as Administrator once, or approve the firewall command shown there.
  echo You can also test on the main PC:
  echo   curl.exe --noproxy "*" http://127.0.0.1:%SERVER_PORT%/status
  echo   curl.exe --noproxy "*" %STACKWIRE_API_URL%/status
  pause
  exit /b 1
)

echo.
echo Starting desktop client connected to %STACKWIRE_API_URL%
echo.

set "PYTHONW_EXE=%~dp0venv\Scripts\pythonw.exe"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if exist "%PYTHONW_EXE%" (
  start "StackWire" "%PYTHONW_EXE%" -m app.desktop
  goto launched
)

start "StackWire" "%PYTHON_EXE%" -m app.desktop

:launched
if "%STACKWIRE_KEEP_CMD%"=="1" exit /b 0
endlocal
exit

:load_config
for /f "usebackq eol=# tokens=1,* delims==" %%A in (%1) do (
  if not "%%A"=="" set "%%A=%%B"
)
exit /b 0

:ensure_venv
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo Existing virtual environment is broken: %CD%\venv
    echo Delete the venv folder, install Python 3.11+ from python.org, then run this script again.
    pause
    exit /b 1
  )
  exit /b 0
)

if not exist "venv\Scripts\python.exe" (
  call :find_python
  if errorlevel 1 exit /b 1
  echo Creating virtual environment in %CD%\venv ...
  "%PYTHON_EXE%" %PYTHON_ARGS% -m venv venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
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
