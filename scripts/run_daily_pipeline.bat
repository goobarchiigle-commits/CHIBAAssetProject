@echo off
cd /d C:\ai-trading

echo [%date% %time%] === DRY RUN START === >> logs\scheduler.log
python src\run_morning_signal.py --dry-run > logs\dryrun_latest.log 2>&1
echo [%date% %time%] DRY RUN exit=%errorlevel% >> logs\scheduler.log

echo [%date% %time%] === LIVE START === >> logs\scheduler.log
python src\run_morning_signal.py > logs\live_latest.log 2>&1
echo [%date% %time%] LIVE exit=%errorlevel% >> logs\scheduler.log
