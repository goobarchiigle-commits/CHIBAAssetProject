"""
scripts/run_weekly_pdca.py
週次PDCAオーケストレーター

実行順序:
  1. research/weekly_report.py を実行 → research_log/weekly_YYYY-MM-DD.md に保存
  2. Claude CLI でAIレビュー（PDCA分析）→ 同ファイルに追記
  3. git add + git commit

使い方:
  python scripts/run_weekly_pdca.py              # 通常実行（金曜15:30 タスクスケジューラーから）
  python scripts/run_weekly_pdca.py --dry-run    # ファイル保存なし・git commit なし（テスト用）
  python scripts/run_weekly_pdca.py --no-ai      # AIレビューなし（データ確認のみ）
  python scripts/run_weekly_pdca.py --no-git     # git commit をスキップ
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

JST        = timezone(timedelta(hours=9))
BASE_DIR   = Path(__file__).parent.parent
REPORT_DIR = BASE_DIR / "research_log"
LOG_DIR    = BASE_DIR / "logs" / "pdca"

CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")
PYTHON_CMD = sys.executable

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("weekly_pdca")

# ------------------------------------------------------------------ #
# AIレビュープロンプト
# ------------------------------------------------------------------ #
AI_REVIEW_PROMPT = """\
あなたは定量トレーディング戦略の分析者です。
以下は今週のフジコ法（モメンタム戦略）の運用レポートです。PDCAサイクルで分析してください。

## 分析の焦点
1. **レジーム変化**: trend_market / trend_strength の変化。bull→neutral 転換の兆候があるか
2. **モメンタム減衰**: RSRリーダー半減期が短くなっていないか（<8日 → 回転相場で戦略効果低下）
3. **リスク集中**: 同一セクターへの集中度。blocked_leaders_count が高い場合の対策
4. **発注リスク監査**: 過剰発注頻度・流動性インパクト・シグナルクラスタリングの異常値
5. **供給構造**: rsr_pass / near_breakout の推移からブレイクアウト機会の枯渇・増加を判定

## 出力形式（必ずこのMarkdown構造で出力すること）

---

## AI レビュー（PDCA） {DATE}

### P（今週の評価）
- 戦略パフォーマンス:
- 市場環境:
- 主なリスク:

### D（今週実施したこと）
- 取引サマリー:
- パラメータ変更:

### C（結果の検証）
- 期待通りだった点:
- 期待外れだった点:
- 4/8 判断ロジックの結果:

### A（来週のアクション）
1. （最優先・具体的コマンドまたはパラメータ名を示す）
2.
3.

### 発注リスク監査
- 過剰発注:
- 流動性インパクト:
- シグナルクラスタリング:

---
"""


# ------------------------------------------------------------------ #
# Step 1: レポート生成
# ------------------------------------------------------------------ #
def generate_report(report_path: Path, dry_run: bool = False) -> bool:
    """weekly_report.py を実行してファイルに保存する"""
    weekly_report_py = BASE_DIR / "research" / "weekly_report.py"
    if not weekly_report_py.exists():
        logger.error("weekly_report.py が見つかりません: %s", weekly_report_py)
        return False

    logger.info("週次レポート生成中...")
    header = (
        f"# 週次レポート {datetime.now(JST).strftime('%Y-%m-%d')}\n\n"
        f"生成日時: {datetime.now(JST).strftime('%Y-%m-%dT%H:%M:%S JST')}\n\n"
        "---\n\n"
    )

    try:
        result = subprocess.run(
            [PYTHON_CMD, str(weekly_report_py), "--weeks", "8"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
            timeout=120,
        )
        if result.returncode != 0 and result.stderr.strip():
            logger.warning("weekly_report.py stderr: %s", result.stderr[:500])

        report_body = result.stdout or "（データなし。logs/trades.jsonl と logs/diagnostics/metrics.jsonl を確認してください）"
        content = header + "```\n" + report_body + "\n```\n\n"

        if not dry_run:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            report_path.write_text(content, encoding="utf-8")
            logger.info("レポート保存: %s (%d bytes)", report_path, len(content))
        else:
            logger.info("[DRY RUN] レポート生成OK（先頭400文字）:\n%s", report_body[:400])

        return True

    except subprocess.TimeoutExpired:
        logger.error("weekly_report.py がタイムアウト（120秒）")
        return False
    except Exception as e:
        logger.error("weekly_report.py 実行エラー: %s", e)
        return False


# ------------------------------------------------------------------ #
# Step 2: AIレビュー（Claude CLI）
# ------------------------------------------------------------------ #
def run_ai_review(report_path: Path, dry_run: bool = False) -> bool:
    """Claude CLI でAIレビューを実行して report_path に追記する"""
    if not report_path.exists():
        logger.error("レポートファイルが存在しません: %s", report_path)
        return False

    report_content = report_path.read_text(encoding="utf-8")
    date_str  = datetime.now(JST).strftime("%Y-%m-%d")
    prompt    = (
        AI_REVIEW_PROMPT.replace("{DATE}", date_str)
        + "\n\n## 今週のレポート\n\n"
        + report_content
    )

    logger.info("AI レビュー実行中（Claude CLI --print）...")

    try:
        result = subprocess.run(
            [CLAUDE_CMD, "--print"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
            timeout=300,
        )

        if result.returncode != 0 and result.stderr.strip():
            logger.warning("claude stderr: %s", result.stderr[:300])

        ai_output = result.stdout.strip()
        if not ai_output:
            logger.warning("AIレビュー出力が空でした（claude CLIの応答なし）")
            return False

        if not dry_run:
            with report_path.open("a", encoding="utf-8") as f:
                f.write("\n\n" + ai_output + "\n")
            logger.info("AIレビュー追記完了: %s", report_path)
        else:
            logger.info("[DRY RUN] AIレビュー出力 (先頭400文字):\n%s", ai_output[:400])

        return True

    except subprocess.TimeoutExpired:
        logger.error("claude CLI がタイムアウト（300秒）")
        return False
    except FileNotFoundError:
        logger.error(
            "claude コマンドが見つかりません。"
            "環境変数 CLAUDE_CMD で正しいパスを指定してください: %s",
            CLAUDE_CMD,
        )
        return False
    except Exception as e:
        logger.error("claude CLI 実行エラー: %s", e)
        return False


# ------------------------------------------------------------------ #
# Step 3: git commit
# ------------------------------------------------------------------ #
def git_commit(report_path: Path) -> bool:
    """research_log/ の変更を git commit する"""
    logger.info("git commit 実行中...")
    try:
        subprocess.run(
            ["git", "add", str(REPORT_DIR)],
            cwd=str(BASE_DIR), check=True, capture_output=True,
        )
        commit_msg = f"weekly PDCA: {report_path.name}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode == 0:
            logger.info("git commit 完了: %s", commit_msg)
        elif "nothing to commit" in (result.stdout + result.stderr):
            logger.info("git: 変更なし（既にコミット済み）")
        else:
            logger.warning("git commit 出力: %s", (result.stdout + result.stderr).strip()[:200])
        return True
    except Exception as e:
        logger.error("git commit エラー: %s", e)
        return False


# ------------------------------------------------------------------ #
# エントリーポイント
# ------------------------------------------------------------------ #
def main() -> int:
    p = argparse.ArgumentParser(description="週次PDCAオーケストレーター")
    p.add_argument("--dry-run", action="store_true", help="ファイル保存・git commit なし（動作確認用）")
    p.add_argument("--no-ai",  action="store_true", help="AIレビューをスキップ")
    p.add_argument("--no-git", action="store_true", help="git commit をスキップ")
    args = p.parse_args()

    now_str     = datetime.now(JST).strftime("%Y-%m-%d")
    report_path = REPORT_DIR / f"weekly_{now_str}.md"

    logger.info("=== 週次PDCA開始: %s ===", now_str)

    # Step 1
    if not generate_report(report_path, dry_run=args.dry_run):
        logger.error("レポート生成失敗。中断します。")
        return 1

    # Step 2
    if not args.no_ai:
        ok = run_ai_review(report_path, dry_run=args.dry_run)
        if not ok:
            logger.warning("AIレビュー失敗（レポート自体は保存済み）")
    else:
        logger.info("AIレビュースキップ（--no-ai）")

    # Step 3
    if not args.dry_run and not args.no_git:
        git_commit(report_path)

    logger.info("=== 週次PDCA完了: %s ===", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
