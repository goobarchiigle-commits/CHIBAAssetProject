# scripts\setup_task_scheduler.ps1
# CHIBA Asset Trading - Windows Task Scheduler Registration
#
# Run (no administrator required for current user tasks):
#   cd C:\ai-trading
#   powershell -ExecutionPolicy Bypass -File scripts\setup_task_scheduler.ps1
#
# Tasks created:
#   CHIBATrading_DryRun  - 08:43 weekdays (dry run, no orders)
#   CHIBATrading_Live    - 08:44 weekdays (live orders via watchdog)

$ErrorActionPreference = "Stop"
$ROOT   = "C:\ai-trading"
$LOGDIR = "$ROOT\logs\scheduler"

# Find Python executable
try {
    $PYTHON = (Get-Command python -ErrorAction Stop).Source
    Write-Host "Python: $PYTHON" -ForegroundColor Cyan
} catch {
    Write-Error "Python not found in PATH."
    exit 1
}

# Create log directory upfront (batch wrappers also create it, but ensure it here)
New-Item -ItemType Directory -Path $LOGDIR -Force | Out-Null
Write-Host "Log dir: $LOGDIR" -ForegroundColor Green

# New-BatchWrapper: generates a .bat file for the given watchdog args.
#
# WHY ASCII ENCODING:
#   [System.Text.Encoding]::UTF8 in Windows .NET writes a UTF-8 BOM (EF BB BF)
#   as the first three bytes. cmd.exe interprets those bytes as part of the first
#   command token, turning "@echo off" into garbage and aborting the script.
#   [System.Text.Encoding]::ASCII emits no BOM and is safe for all batch content
#   that is pure ASCII (paths, flags). Use this for every .bat you generate.
#
# WHY UNIQUE TIMESTAMP IN LOG NAMES:
#   Windows locks files open for writing exclusively. If a previous run's log
#   is still open (or two tasks fire at near the same second with the same date),
#   the >> operator on the same filename throws "process cannot access the file".
#   Embedding yyyyMMdd_HHmmss makes each run write to its own file - zero contention.
#
# WHY POWERSHELL DATE IN BATCH:
#   %date% expands differently by locale (yyyy/MM/dd, MM/dd/yyyy, ddd MM/dd/yyyy).
#   Calling PowerShell Get-Date -Format inside the batch guarantees a fixed format
#   on any Windows locale or regional setting.
function New-BatchWrapper {
    param(
        [string]$Name,        # e.g. "run_dry"
        [string]$ScriptArgs   # e.g. "--dry" or "--live --yes"
    )

    $bat = "$ROOT\scripts\$Name.bat"

    # Build batch content as a single string with CRLF line endings.
    # %%i is correct batch syntax for a for-loop variable (double %% in .bat).
    $lines = @(
        '@echo off',
        ':: WHY ANSI/NO-BOM: UTF-8 BOM corrupts "@echo off" - see New-BatchWrapper header.',
        '',
        'setlocal EnableDelayedExpansion',
        '',
        ':: WHY POWERSHELL DATE: %date% is locale-dependent; Get-Date is not.',
        'for /f %%i in (''powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"'') do set TIMESTAMP=%%i',
        '',
        "set LOG_DIR=$LOGDIR",
        '',
        'if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"',
        '',
        "set LOG_OUT=%LOG_DIR%\${Name}_%TIMESTAMP%.log",
        "set LOG_ERR=%LOG_DIR%\${Name}_%TIMESTAMP%.err.log",
        '',
        "cd /d `"$ROOT`"",
        '',
        ':: Unique timestamp per run = each run owns its log file = no Windows lock collision.',
        ':: Separate .log/.err.log = stdout and stderr never interleaved.',
        "`"$PYTHON`" src\watchdog_runner.py $ScriptArgs 1>`"%LOG_OUT%`" 2>`"%LOG_ERR%`"",
        '',
        'endlocal'
    )

    $content = $lines -join "`r`n"

    # ASCII encoding: no BOM, no multi-byte surprises, safe for cmd.exe.
    [System.IO.File]::WriteAllText($bat, $content, [System.Text.Encoding]::ASCII)
    Write-Host "  Written (ANSI/no-BOM): $bat" -ForegroundColor Gray
    return $bat
}

$batDry  = New-BatchWrapper -Name "run_dry"  -ScriptArgs "--dry"
$batLive = New-BatchWrapper -Name "run_live" -ScriptArgs "--live --yes"
Write-Host "Batch wrappers created" -ForegroundColor Green

# Register tasks using schtasks.exe (works without admin for current user)
function Register-TradingTask {
    param([string]$Name, [string]$BatFile, [string]$Time)

    schtasks /Delete /TN $Name /F 2>$null | Out-Null

    $result = schtasks /Create `
        /TN  $Name `
        /TR  "`"$BatFile`"" `
        /SC  WEEKLY `
        /D   MON,TUE,WED,THU,FRI `
        /ST  $Time `
        /F 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: $Name registered at $Time" -ForegroundColor Green
    } else {
        Write-Host "  FAIL: $result" -ForegroundColor Red
        throw "Task registration failed: $Name"
    }
}

Write-Host ""
Write-Host "[1/2] CHIBATrading_DryRun (08:43)..." -ForegroundColor Yellow
Register-TradingTask -Name "CHIBATrading_DryRun" -BatFile $batDry  -Time "08:43"

Write-Host "[2/2] CHIBATrading_Live (08:44)..." -ForegroundColor Yellow
Register-TradingTask -Name "CHIBATrading_Live"   -BatFile $batLive -Time "08:44"

# Verify
Write-Host ""
Write-Host "=== Registered Tasks ===" -ForegroundColor Cyan
schtasks /Query /TN "CHIBATrading_DryRun" /FO LIST 2>&1 | Select-String "Task Name|Next Run Time|Status"
schtasks /Query /TN "CHIBATrading_Live"   /FO LIST 2>&1 | Select-String "Task Name|Next Run Time|Status"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Weekday schedule: 08:43 dry / 08:44 live"
Write-Host "Log files: $LOGDIR\"
Write-Host "  stdout: run_dry_YYYYMMDD_HHMMSS.log"
Write-Host "  stderr: run_dry_YYYYMMDD_HHMMSS.err.log"
Write-Host ""
Write-Host "Email notifications (add to src\.env):"
Write-Host "  NOTIFY_SMTP_USER     = your@gmail.com"
Write-Host "  NOTIFY_SMTP_PASSWORD = xxxx xxxx xxxx xxxx"
Write-Host "  NOTIFY_TO            = destination@example.com"
