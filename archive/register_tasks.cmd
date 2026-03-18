@echo off
chcp 65001 > nul

set PROJ=C:\Users\owner\.gemini\antigravity\scratch\asset_simulation

:: 既存タスク削除（エラー無視）
schtasks /delete /tn "CHIBAAsset_DryRun" /f 2>nul
schtasks /delete /tn "CHIBAAsset_Live" /f 2>nul

:: 8:30 ドライラン（月〜金）
schtasks /create /tn "CHIBAAsset_DryRun" /tr "cmd /c \"%PROJ%\morning_dryrun.bat\"" /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 08:30 /rl HIGHEST /f
if %ERRORLEVEL% neq 0 (
    echo [ERROR] DryRunタスク登録失敗
    exit /b 1
)
echo [OK] CHIBAAsset_DryRun 登録完了 (平日 08:30)

:: 9:00 実発注（月〜金）
schtasks /create /tn "CHIBAAsset_Live" /tr "cmd /c \"%PROJ%\morning_live.bat\"" /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 09:00 /rl HIGHEST /f
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Liveタスク登録失敗
    exit /b 1
)
echo [OK] CHIBAAsset_Live 登録完了 (平日 09:00)

:: 確認
echo.
echo === 登録済みタスク ===
schtasks /query /tn "CHIBAAsset_DryRun" /fo LIST
schtasks /query /tn "CHIBAAsset_Live" /fo LIST
