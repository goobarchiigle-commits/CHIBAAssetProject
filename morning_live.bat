@echo off
cd /d "C:\Users\owner\.gemini\antigravity\scratch\asset_simulation"

:: 日付をYYYYMMDD形式で取得（ロケール依存を避けるためPythonを使用）
for /f %%i in ('python -c "import datetime; print(datetime.datetime.now().strftime('%%Y%%m%%d'))"') do set LOGDATE=%%i

set LOGFILE=data\logs\live_%LOGDATE%.log

echo [%date% %time%] 実発注開始 >> %LOGFILE%
python run_live_signal.py --live --yes >> %LOGFILE% 2>&1
echo [%date% %time%] 実発注完了 >> %LOGFILE%
