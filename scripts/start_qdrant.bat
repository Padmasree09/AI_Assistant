@echo off
setlocal

if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%A:~0,1"=="#" set "%%A=%%B"
  )
)

set QDRANT_PORT=%QDRANT_PORT%
if "%QDRANT_PORT%"=="" set QDRANT_PORT=6333

docker run -p %QDRANT_PORT%:6333 qdrant/qdrant
