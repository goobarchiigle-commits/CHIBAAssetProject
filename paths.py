"""
paths.py — プロジェクトパス・ライブ安全設定 一元管理
最終更新: 2026-03-31（C:/ai-trading 移行対応）

設計原則:
  1. PROJECT_ROOT  — このファイル（paths.py）の場所 = コードベースの起点
  2. BASE_DIR      — C:/ai-trading（外部データの単一正本）
                     AI_TRADING_HOME 環境変数で上書き可能（.env で設定推奨）
  3. ライブ安全ガード — LIVE_MODE / MAX_ORDERS_PER_DAY を環境変数で制御

環境変数一覧:
    AI_TRADING_HOME      外部データ基点（デフォルト: C:/ai-trading）
    LIVE_UNIVERSE_FILE   ユニバースファイルパス（PROJECT_ROOT 相対 or 絶対パス）
    LIVE_MODE            true にすると実発注モード（デフォルト: false）
    MAX_ORDERS_PER_DAY   1日の最大発注件数（デフォルト: 20）

使い方:
    from src.paths import PROJECT_ROOT, BASE_DIR, RESULTS_DIR, SIGNALS_DIR
    from src.paths import LIVE_MODE, assert_live_ready
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────────
# PROJECT_ROOT — コードベースの起点（このファイルの場所）
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ────────────────────────────────────────────────────────────────────────────
# .env 読み込み（python-dotenv があれば使用、なければ手動パース）
# paths を import するだけで .env が環境変数に反映される
# ────────────────────────────────────────────────────────────────────────────
_env_file = PROJECT_ROOT / ".env"
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_env_file)
except ImportError:
    if _env_file.exists():
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


# ────────────────────────────────────────────────────────────────────────────
# BASE_DIR — 外部データの単一正本（C:/ai-trading）
#
# AI_TRADING_HOME 環境変数（.env に記載）で上書き可能。
# 未設定の場合は PROJECT_ROOT（コードと同じ場所）にフォールバックするが、
# 通常は .env の AI_TRADING_HOME=C:/ai-trading が有効になる。
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(os.environ.get("AI_TRADING_HOME", str(PROJECT_ROOT)))


# kabuステーション API 接続設定
KABUS_HOST: str = os.environ.get("KABUS_HOST", "localhost")
KABUS_PORT: int = int(os.environ.get("KABUS_PORT", "18080"))
KABUS_PASSWORD: str = os.environ.get("KABUS_PASSWORD", "")

# ────────────────────────────────────────────────────────────────────────────
# 外部データディレクトリ（BASE_DIR 配下 = C:/ai-trading 配下）
# ────────────────────────────────────────────────────────────────────────────
RESULTS_DIR:  Path = BASE_DIR / "backtests"
LOGS_DIR:     Path = BASE_DIR / "logs"
CACHE_DIR:    Path = BASE_DIR / "cache"
DATASET_DIR:  Path = BASE_DIR / "data"
RUNTIME_DIR:  Path = BASE_DIR / "runtime"
REPORTS_DIR:  Path = BASE_DIR / "reports"
DOCS_DIR:     Path = BASE_DIR / "docs" / "research"


# ────────────────────────────────────────────────────────────────────────────
# コードと一体のディレクトリ（PROJECT_ROOT 相対・上書き不可）
# ────────────────────────────────────────────────────────────────────────────
CONFIGS_DIR:  Path = PROJECT_ROOT / "configs"
UNIVERSE_DIR: Path = CONFIGS_DIR  / "universe"


# ────────────────────────────────────────────────────────────────────────────
# データパイプライン
# ────────────────────────────────────────────────────────────────────────────
BACKTEST_DATASET_DIR: Path = DATASET_DIR / "backtest_dataset"
SIGNALS_DIR:          Path = DATASET_DIR / "signals"


# ────────────────────────────────────────────────────────────────────────────
# 設定ファイル（環境変数で上書き可能）
# ────────────────────────────────────────────────────────────────────────────
RSR_UNIVERSE_FILE: Path = CONFIGS_DIR / "rsr_universe_42.csv"

def _resolve_path(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var, "").strip()
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else PROJECT_ROOT / p

LIVE_UNIVERSE_FILE:   Path = _resolve_path(
    "LIVE_UNIVERSE_FILE", UNIVERSE_DIR / "rsr42_trading.json"
)
SHADOW_UNIVERSE_FILE: Path = UNIVERSE_DIR / "shadow_universe.json"


# ────────────────────────────────────────────────────────────────────────────
# ライブ運用ファイル
# ────────────────────────────────────────────────────────────────────────────
ORDER_LOCK_FILE:     Path = RUNTIME_DIR / "order_lock.json"
LIVE_LOG_DIR:        Path = LOGS_DIR    / "live"
PHASE2_METRICS_FILE: Path = LOGS_DIR    / "phase2_live_metrics.jsonl"


# ────────────────────────────────────────────────────────────────────────────
# ライブ運用安全設定
# ────────────────────────────────────────────────────────────────────────────
LIVE_MODE: bool = os.environ.get("LIVE_MODE", "false").lower() == "true"
MAX_ORDERS_PER_DAY: int = int(os.environ.get("MAX_ORDERS_PER_DAY", "20"))
MAX_SINGLE_POSITION_YEN: int = int(os.environ.get("MAX_SINGLE_POSITION_YEN", "600000"))
ORDER_RATE_LIMIT_SEC: float = float(os.environ.get("ORDER_RATE_LIMIT_SEC", "5"))

# ────────────────────────────────────────────────────────────────────────────
# kabuステーション API 接続設定
# ────────────────────────────────────────────────────────────────────────────
KABUS_HOST:     str = os.environ.get("KABUS_HOST", "localhost")
KABUS_PORT:     int = int(os.environ.get("KABUS_PORT", "18080"))
KABUS_PASSWORD: str = os.environ.get("KABU_API_PASSWORD", "")

# 発注タイムスタンプ記録ファイル（レートリミット用）
_LAST_ORDER_FILE: Path = RUNTIME_DIR / "_last_order_ts.json"


def assert_live_ready() -> None:
    """
    ライブ発注の直前に呼ぶ安全チェック（4項目一括検証）。
    条件を満たさない場合は RuntimeError を送出してプロセスを止める。
    """
    errors: list[str] = []

    if not LIVE_MODE:
        errors.append(
            "LIVE_MODE が有効ではありません。\n"
            "  .env に LIVE_MODE=true を追加してください。"
        )

    if not LIVE_UNIVERSE_FILE.exists():
        errors.append(
            f"ユニバースファイルが見つかりません: {LIVE_UNIVERSE_FILE}"
        )

    if MAX_ORDERS_PER_DAY <= 0:
        errors.append(
            f"MAX_ORDERS_PER_DAY={MAX_ORDERS_PER_DAY} が不正です（1以上を設定してください）。"
        )

    if errors:
        raise RuntimeError(
            "ライブ発注の安全チェックに失敗しました:\n" +
            "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# 実行コンテキスト検証（LIVE_MODE 時のスクリプト許可リスト）
# ────────────────────────────────────────────────────────────────────────────
_LIVE_ALLOWED_SCRIPTS: frozenset[str] = frozenset({
    "run_live_signal.py",
    "run_morning_signal.py",
})


def assert_execution_context() -> None:
    """
    呼び出し元スクリプトが実発注を許可されているか検証する。
    LIVE_MODE=true のとき、許可リスト外から呼ばれた場合は RuntimeError を送出。
    """
    if not LIVE_MODE:
        return

    import sys
    script = Path(sys.argv[0]).resolve().name

    if script not in _LIVE_ALLOWED_SCRIPTS:
        raise RuntimeError(
            f"実発注ブロック: '{script}' は LIVE_MODE での実行が許可されていません。\n"
            f"  許可スクリプト: {sorted(_LIVE_ALLOWED_SCRIPTS)}\n"
            "  ドライランで実行するか、許可リスト(_LIVE_ALLOWED_SCRIPTS)を確認してください。"
        )


# ────────────────────────────────────────────────────────────────────────────
# データ整合性チェック
# ────────────────────────────────────────────────────────────────────────────
def assert_kabus_connection() -> None:
    """
    kabuステーション API 接続設定の最低限チェック。
    接続前に必ず呼ぶ。
    """
    errors: list[str] = []

    if not KABUS_HOST:
        errors.append("KABUS_HOST が未設定です")

    if not isinstance(KABUS_PORT, int) or KABUS_PORT <= 0:
        errors.append(f"KABUS_PORT={KABUS_PORT} が不正です")

    if not KABUS_PASSWORD:
        errors.append(
            "KABU_API_PASSWORD が未設定です。\n"
            "  .env に KABU_API_PASSWORD=<パスワード> を設定してください。"
        )

    if errors:
        raise RuntimeError(
            "kabuステーション接続設定エラー:\n" +
            "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )


def acquire_runtime_lock() -> None:
    """
    二重起動防止ロックを取得する。
    プロセス終了時に自動解放（atexit 登録）。
    """
    import atexit
    if ORDER_LOCK_FILE.exists():
        try:
            data = json.loads(ORDER_LOCK_FILE.read_text(encoding="utf-8"))
            pid = data.get("pid", "?")
        except Exception:
            pid = "?"
        raise RuntimeError(
            f"別のインスタンスが実行中です (PID={pid})。\n"
            f"  {ORDER_LOCK_FILE} を確認してください。\n"
            "  正常終了していれば手動で削除してください。"
        )
    ORDER_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    ORDER_LOCK_FILE.write_text(
        json.dumps({"pid": os.getpid(), "started": time.time()}),
        encoding="utf-8",
    )
    atexit.register(release_runtime_lock)


def release_runtime_lock() -> None:
    """ロックファイルを削除してロックを解放する。"""
    try:
        ORDER_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def enforce_order_rate_limit() -> None:
    """
    直前の発注から ORDER_RATE_LIMIT_SEC 秒未満の場合は RuntimeError を送出する。
    kabuステーション API の過剰発注 BAN 防止。
    """
    if not _LAST_ORDER_FILE.exists():
        return
    try:
        data = json.loads(_LAST_ORDER_FILE.read_text(encoding="utf-8"))
        elapsed = time.time() - data.get("ts", 0.0)
        if elapsed < ORDER_RATE_LIMIT_SEC:
            wait = ORDER_RATE_LIMIT_SEC - elapsed
            raise RuntimeError(
                f"レートリミット: あと {wait:.1f}秒 待機が必要です "
                f"（ORDER_RATE_LIMIT_SEC={ORDER_RATE_LIMIT_SEC}秒）。"
            )
    except (json.JSONDecodeError, KeyError):
        pass


def record_order_sent() -> None:
    """発注タイムスタンプを記録する（enforce_order_rate_limit 用）。"""
    _LAST_ORDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAST_ORDER_FILE.write_text(
        json.dumps({"ts": time.time()}),
        encoding="utf-8",
    )


def verify_dataset_integrity(data_version: str = "") -> None:
    """
    バックテスト用データセットの存在を確認する。

    Args:
        data_version: 確認する DATA_VERSION（例 "2026-03-28"）。
                      空文字の場合は BACKTEST_DATASET_DIR の存在のみ確認。

    Raises:
        RuntimeError: 必要なパスが存在しない場合。
    """
    errors: list[str] = []

    if not BACKTEST_DATASET_DIR.exists():
        errors.append(
            f"backtest_dataset ディレクトリが存在しません: {BACKTEST_DATASET_DIR}\n"
            "  先に build_dataset_snapshot.py を実行してください。"
        )
    elif data_version:
        snapshot_dir = BACKTEST_DATASET_DIR / data_version
        if not snapshot_dir.exists():
            errors.append(
                f"スナップショット '{data_version}' が存在しません: {snapshot_dir}\n"
                f"  DATA_VERSION={data_version} python build_dataset_snapshot.py"
            )
        else:
            meta = snapshot_dir / "_meta.json"
            parquets = list(snapshot_dir.glob("*.parquet"))
            if not meta.exists():
                errors.append(f"_meta.json が見つかりません: {meta}")
            if not parquets:
                errors.append(f"parquet ファイルが見つかりません: {snapshot_dir}")

    if not RSR_UNIVERSE_FILE.exists():
        errors.append(f"RSRユニバース CSV が見つかりません: {RSR_UNIVERSE_FILE}")

    if errors:
        raise RuntimeError(
            "データ整合性チェックに失敗しました:\n" +
            "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )
