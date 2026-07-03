@echo off
setlocal EnableDelayedExpansion
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TIMESTAMP=%%i
set LOG_DIR=C:\ai-trading\logs\scheduler
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set LOG_OUT=%LOG_DIR%\weekly_intelligence_%TIMESTAMP%.log
set LOG_ERR=%LOG_DIR%\weekly_intelligence_%TIMESTAMP%.err.log
cd /d "C:\ai-trading"
"C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe" "C:\ai-trading\src\run_weekly_market_intelligence.py" 1>"%LOG_OUT%" 2>"%LOG_ERR%"
endlocal