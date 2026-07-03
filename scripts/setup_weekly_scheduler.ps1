# scripts\setup_weekly_scheduler.ps1
# CHIBAAssetProject - WeeklyMarketIntelligence Task Scheduler registration
#
# Usage:
#   cd C:\ai-trading
#   powershell -ExecutionPolicy Bypass -File scripts\setup_weekly_scheduler.ps1
#
# Registers: WeeklyMarketIntelligence (every Friday 15:30)
#
# NOTE: Run as standard user (no admin required).
# NOTE: ASCII-only source to avoid UTF-8 BOM issues in Windows PowerShell 5.1.

$ErrorActionPreference = "Stop"
$ROOT      = "C:\ai-trading"
$LOGDIR    = "$ROOT\logs\scheduler"
$BAT_PATH  = "$ROOT\scripts\run_weekly_intelligence.bat"
$TASK_NAME = "WeeklyMarketIntelligence"

# Create log directory
New-Item -ItemType Directory -Path $LOGDIR -Force | Out-Null

# Locate Python
try {
    $PYTHON = (Get-Command python -ErrorAction Stop).Source
    Write-Host "Python: $PYTHON" -ForegroundColor Cyan
} catch {
    Write-Error "Python not found in PATH."
    exit 1
}

# WHY ASCII ENCODING: UTF-8 BOM corrupts '@echo off' in cmd.exe.
# WHY UNIQUE TIMESTAMP: Windows locks log files exclusively; unique names avoid contention.
# WHY POWERSHELL DATE IN BATCH: %date% is locale-dependent; Get-Date is not.
$batLines = @(
    '@echo off',
    ':: WeeklyMarketIntelligence - ASCII/no-BOM batch launcher',
    '',
    'setlocal EnableDelayedExpansion',
    '',
    'for /f %%i in (''powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"'') do set TIMESTAMP=%%i',
    '',
    "set LOG_DIR=$LOGDIR",
    'if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"',
    '',
    'set LOG_OUT=%LOG_DIR%\weekly_intelligence_%TIMESTAMP%.log',
    'set LOG_ERR=%LOG_DIR%\weekly_intelligence_%TIMESTAMP%.err.log',
    '',
    "cd /d `"$ROOT`"",
    '',
    "`"$PYTHON`" `"$ROOT\src\run_weekly_market_intelligence.py`" 1>`"%LOG_OUT%`" 2>`"%LOG_ERR%`"",
    '',
    'endlocal'
)
$batContent = $batLines -join "`r`n"
[System.IO.File]::WriteAllText($BAT_PATH, $batContent, [System.Text.Encoding]::ASCII)
Write-Host "Batch wrapper: $BAT_PATH" -ForegroundColor Green

# Remove existing task if present
schtasks /Delete /TN $TASK_NAME /F 2>$null | Out-Null

# Register (no /RL HIGHEST - standard user task)
$result = schtasks /Create `
    /TN $TASK_NAME `
    /TR "`"$BAT_PATH`"" `
    /SC WEEKLY `
    /D  FRI `
    /ST 15:30 `
    /F 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: $TASK_NAME registered (every Friday 15:30)" -ForegroundColor Green
} else {
    Write-Host "FAIL: $result" -ForegroundColor Red
    throw "Task registration failed: $TASK_NAME"
}

# Verify
Write-Host ""
Write-Host "=== Registered Task ===" -ForegroundColor Cyan
schtasks /Query /TN $TASK_NAME /FO LIST 2>&1 | Select-String "Task Name|Next Run Time|Status"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Schedule  : every Friday 15:30"
Write-Host "  Log dir   : $LOGDIR"
Write-Host "  Test run  : python src\run_weekly_market_intelligence.py --dry-run"
