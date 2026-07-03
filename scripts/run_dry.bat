@echo off
:: WHY ANSI/NO-BOM: UTF-8 BOM corrupts "@echo off" - see New-BatchWrapper header.

setlocal EnableDelayedExpansion

:: WHY POWERSHELL DATE: %date% is locale-dependent; Get-Date is not.
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TIMESTAMP=%%i

set LOG_DIR=C:\ai-trading\logs\scheduler

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set LOG_OUT=%LOG_DIR%\run_dry_%TIMESTAMP%.log
set LOG_ERR=%LOG_DIR%\run_dry_%TIMESTAMP%.err.log

cd /d "C:\ai-trading"

:: Unique timestamp per run = each run owns its log file = no Windows lock collision.
:: Separate .log/.err.log = stdout and stderr never interleaved.
"C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe" src\watchdog_runner.py --dry 1>"%LOG_OUT%" 2>"%LOG_ERR%"

endlocal