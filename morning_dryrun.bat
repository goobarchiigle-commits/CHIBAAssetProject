@echo off
cd /d "C:\Users\owner\.gemini\antigravity\scratch\asset_simulation"

:: 日付をYYYYMMDD形式で取得（ロケール依存を避けるためPythonを使用）
for /f %%i in ('python -c "import datetime; print(datetime.datetime.now().strftime('%%Y%%m%%d'))"') do set LOGDATE=%%i

set LOGFILE=data\logs\dryrun_%LOGDATE%.log

echo [%date% %time%] ドライラン開始 >> %LOGFILE%
python run_live_signal.py >> %LOGFILE% 2>&1
echo [%date% %time%] ドライラン完了 >> %LOGFILE%
