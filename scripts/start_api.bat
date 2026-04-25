@echo off
setlocal

if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" if not "%%A:~0,1"=="#" set "%%A=%%B"
  )
)

set APP_HOST=%APP_HOST%
if "%APP_HOST%"=="" set APP_HOST=127.0.0.1

set APP_PORT=%APP_PORT%
if "%APP_PORT%"=="" set APP_PORT=8000

uvicorn main:app --host %APP_HOST% --port %APP_PORT% --reload
