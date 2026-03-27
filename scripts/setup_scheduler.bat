@echo off
REM ================================================================
REM CHIBAAssetProject - Windowsタスクスケジューラー登録スクリプト
REM
REM 登録タスク:
REM   CHIBAAsset_Weekly_PDCA   : 毎週金曜 15:30 に週次PDCAを実行
REM   CHIBAAsset_Daily_Monitor : 毎日 16:00 に日次監視を実行（土日はPython側でスキップ）
REM
REM 使い方（管理者権限不要）:
REM   scripts\setup_scheduler.bat
REM ================================================================

set BASE_DIR=C:\Users\owner\.gemini\antigravity\scratch\asset_simulation
set PYTHON=python
set WEEKLY_SCRIPT=%BASE_DIR%\scripts\run_weekly_pdca.py
set DAILY_SCRIPT=%BASE_DIR%\scripts\daily_monitor.py

echo.
echo === CHIBAAssetProject タスクスケジューラー登録 ===
echo BASE_DIR: %BASE_DIR%
echo.

REM ---- 週次PDCAタスク（毎週金曜 15:30）----
echo [1/2] 週次PDCAタスクを登録中...
schtasks /create ^
  /tn "CHIBAAsset_Weekly_PDCA" ^
  /tr "\"%PYTHON%\" \"%WEEKLY_SCRIPT%\"" ^
  /sc WEEKLY ^
  /d FRI ^
  /st 15:30 ^
  /sd 2026/03/28 ^
  /f
if %ERRORLEVEL% == 0 (
    echo       OK: CHIBAAsset_Weekly_PDCA
) else (
    echo       ERROR: 週次PDCAタスクの登録に失敗しました
)

REM ---- 日次監視タスク（毎日 16:00 / 土日はPython側でスキップ）----
echo [2/2] 日次監視タスクを登録中...
schtasks /create ^
  /tn "CHIBAAsset_Daily_Monitor" ^
  /tr "\"%PYTHON%\" \"%DAILY_SCRIPT%\"" ^
  /sc DAILY ^
  /st 16:00 ^
  /sd 2026/03/28 ^
  /f
if %ERRORLEVEL% == 0 (
    echo       OK: CHIBAAsset_Daily_Monitor
) else (
    echo       ERROR: 日次監視タスクの登録に失敗しました
)

echo.
echo === 登録確認 ===
schtasks /query /tn "CHIBAAsset_Weekly_PDCA"  /fo LIST 2>nul | findstr "タスク名\|次回の実行時刻\|状態"
schtasks /query /tn "CHIBAAsset_Daily_Monitor" /fo LIST 2>nul | findstr "タスク名\|次回の実行時刻\|状態"

echo.
echo 登録完了。
echo.
echo 手動テスト実行:
echo   python scripts\run_weekly_pdca.py --dry-run --no-ai
echo   python scripts\daily_monitor.py --force
echo.
pause
