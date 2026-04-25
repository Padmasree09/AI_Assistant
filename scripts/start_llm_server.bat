@echo off
setlocal

if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%A:~0,1"=="#" set "%%A=%%B"
  )
)

if "%LLAMA_SERVER_EXE%"=="" (
  echo LLAMA_SERVER_EXE is not set.
  echo Example:
  echo set LLAMA_SERVER_EXE=C:\path\to\llama-server.exe
  exit /b 1
)

if "%LLAMA_MODEL_PATH%"=="" (
  echo LLAMA_MODEL_PATH is not set.
  echo Example:
  echo set LLAMA_MODEL_PATH=C:\path\to\phi-3.gguf
  exit /b 1
)

set LLAMA_PORT=%LLAMA_PORT%
if "%LLAMA_PORT%"=="" set LLAMA_PORT=8080

"%LLAMA_SERVER_EXE%" -m "%LLAMA_MODEL_PATH%" --port %LLAMA_PORT%
