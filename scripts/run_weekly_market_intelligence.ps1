# scripts\run_weekly_market_intelligence.ps1
# CHIBAAssetProject — 週次市場インテリジェンス 自動実行ラッパー
#
# 実行 (手動テスト):
#   cd C:\ai-trading
#   powershell -ExecutionPolicy Bypass -File scripts\run_weekly_market_intelligence.ps1
#
# 引数:
#   -DryRun   : ファイル書き出し・メール送信なし（動作確認用）
#   -NoEmail  : メール送信のみスキップ
#   -Date     : 基準日 YYYY-MM-DD（省略時は直前の金曜）
#
# Task Scheduler 登録は setup_weekly_scheduler.ps1 で行う。
# タスク名: WeeklyMarketIntelligence / 毎週金曜 15:30

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$NoEmail,
    [string]$Date = ""
)

$ErrorActionPreference = "Stop"
$ROOT    = "C:\ai-trading"
$LOGDIR  = "$ROOT\logs\scheduler"

# ── タイムスタンプ (locale-safe) ─────────────────────────────────────────────
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOGFILE   = "$LOGDIR\weekly_intelligence_$TIMESTAMP.log"
$ERRFILE   = "$LOGDIR\weekly_intelligence_$TIMESTAMP.err.log"

if (-not (Test-Path $LOGDIR)) {
    New-Item -ItemType Directory -Path $LOGDIR -Force | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts  = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    $line = "$ts [$Level] $Message"
    Write-Host $line
    Add-Content -Path $LOGFILE -Value $line -Encoding UTF8
}

Write-Log "=== WeeklyMarketIntelligence 開始 timestamp=$TIMESTAMP ==="

# ── Python 検索 ──────────────────────────────────────────────────────────────
try {
    $PYTHON = (Get-Command python -ErrorAction Stop).Source
    Write-Log "Python: $PYTHON"
} catch {
    Write-Log "Python が PATH に見つかりません" "ERROR"
    Add-Content -Path $ERRFILE -Value "Python not found in PATH" -Encoding UTF8
    exit 1
}

# ── venv 有効化（存在する場合） ───────────────────────────────────────────────
$VENV_ACTIVATE = "$ROOT\.venv\Scripts\Activate.ps1"
if (Test-Path $VENV_ACTIVATE) {
    Write-Log "venv 有効化: $VENV_ACTIVATE"
    try {
        & $VENV_ACTIVATE
        $PYTHON = (Get-Command python -ErrorAction Stop).Source
        Write-Log "venv Python: $PYTHON"
    } catch {
        Write-Log "venv 有効化失敗（システム Python で継続）" "WARN"
    }
}

# ── 引数構築 ─────────────────────────────────────────────────────────────────
$ARGS_LIST = @()
if ($DryRun)  { $ARGS_LIST += "--dry-run" }
if ($NoEmail) { $ARGS_LIST += "--no-email" }
if ($Date)    { $ARGS_LIST += "--date"; $ARGS_LIST += $Date }

$SCRIPT = "$ROOT\src\run_weekly_market_intelligence.py"
Write-Log "実行: $PYTHON $SCRIPT $ARGS_LIST"

# ── Python 実行 ───────────────────────────────────────────────────────────────
try {
    $proc = Start-Process -FilePath $PYTHON `
        -ArgumentList (@($SCRIPT) + $ARGS_LIST) `
        -WorkingDirectory $ROOT `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $LOGFILE `
        -RedirectStandardError  $ERRFILE `
        -Wait

    $EXIT = $proc.ExitCode
} catch {
    Write-Log "Python 実行エラー: $_" "ERROR"
    exit 1
}

# ── 出力確認 ─────────────────────────────────────────────────────────────────
Write-Log "Python 終了コード: $EXIT"

if ($EXIT -ne 0) {
    Write-Log "実行失敗。エラーログ: $ERRFILE" "ERROR"
    if (Test-Path $ERRFILE) {
        Get-Content $ERRFILE | Select-Object -Last 20 | ForEach-Object {
            Write-Log "  STDERR: $_" "ERROR"
        }
    }
    exit $EXIT
}

# ── 出力ファイル検証 ──────────────────────────────────────────────────────────
if (-not $DryRun) {
    $REPORT_DIR = "$ROOT\runtime\reports"
    $DATE_TAG   = if ($Date) {
        [datetime]::ParseExact($Date, "yyyy-MM-dd", $null).ToString("yyyyMMdd")
    } else {
        # 直前の金曜日
        $today = Get-Date
        $daysSinceFriday = ($today.DayOfWeek.value__ - 5 + 7) % 7
        $friday = $today.AddDays(-$daysSinceFriday)
        $friday.ToString("yyyyMMdd")
    }

    $EXPECTED_FILES = @(
        "$REPORT_DIR\weekly_market_intelligence_$DATE_TAG.md",
        "$REPORT_DIR\weekly_market_intelligence_$DATE_TAG.html",
        "$REPORT_DIR\weekly_market_intelligence_$DATE_TAG.json"
    )

    $allOk = $true
    foreach ($f in $EXPECTED_FILES) {
        if (Test-Path $f) {
            $size = (Get-Item $f).Length
            Write-Log "OK: $f ($size bytes)"
        } else {
            Write-Log "MISSING: $f" "ERROR"
            $allOk = $false
        }
    }

    if (-not $allOk) {
        Write-Log "出力ファイルが一部欠落しています。ログを確認してください: $LOGFILE" "ERROR"
        exit 1
    }
}

Write-Log "=== WeeklyMarketIntelligence 正常完了 ==="
exit 0
