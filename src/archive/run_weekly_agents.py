"""
run_weekly_agents.py
週次エージェントチーム実行スクリプト

【実行方法】
  python run_weekly_agents.py

【自動実行】
  タスクスケジューラで日曜 20:00 に登録済み

【エージェント構成】
  01_監督  → 約定照合・エラー管理
  02_分析  → 市場レジーム判定
  03_批判  → 論理的欠陥の指摘
  04_設計  → コード改善提案
  05_総括  → 来週の運用方針決定

【必要な環境変数（.env）】
  ANTHROPIC_API_KEY=sk-ant-api03-...
"""

import sys
import os
import json
import glob
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))

# .env 読み込み
from dotenv import load_dotenv
load_dotenv()

import anthropic

AGENTS_DIR  = Path("agents")
OUTPUTS_DIR = Path("agents/outputs")
SIGNALS_DIR = Path("data/signals")
LOGS_DIR    = Path("data/logs")

MODEL = "claude-opus-4-6"


# ------------------------------------------------------------------ #
# データ収集
# ------------------------------------------------------------------ #
def collect_weekly_data() -> dict:
    """今週のシグナル・ログデータを収集する。"""
    data = {}

    # 今週のシグナルJSONを収集
    signal_files = sorted(SIGNALS_DIR.glob("signal_*.json"))[-5:]  # 直近5日分
    signals_summary = []
    for f in signal_files:
        try:
            content = json.loads(f.read_text(encoding="utf-8"))
            signals_summary.append({
                "date":       content.get("data_as_of"),
                "buy":        len([o for o in content.get("orders", []) if o.get("side") == "BUY"]),
                "sell":       len([o for o in content.get("orders", []) if o.get("side") == "SELL"]),
                "n_universe": content.get("n_universe"),
                "top3_rsr":   [s["symbol"] for s in content.get("signals", [])[:3]],
            })
        except Exception:
            pass
    data["signals_this_week"] = signals_summary

    # ログファイル（エラー行のみ抽出）
    log_file = LOGS_DIR / "morning_signal.log"
    errors = []
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        errors = [l for l in lines if "[ERROR]" in l or "[WARNING]" in l][-20:]
    data["errors_this_week"] = errors

    return data


# ------------------------------------------------------------------ #
# エージェント実行
# ------------------------------------------------------------------ #
def run_agent(client: anthropic.Anthropic, agent_name: str, user_message: str) -> str:
    """1エージェントを実行して出力を返す。"""
    system_prompt_file = AGENTS_DIR / f"{agent_name}.md"
    system_prompt = system_prompt_file.read_text(encoding="utf-8")

    print(f"  [{agent_name}] 実行中...", end="", flush=True)
    response = client.messages.create(
        model   = MODEL,
        max_tokens = 1500,
        system  = system_prompt,
        messages = [{"role": "user", "content": user_message}],
    )
    result = response.content[0].text
    print(f" 完了（{len(result)}文字）")
    return result


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY が .env に設定されていません。")
        print("   .env に以下を追加してください:")
        print("   ANTHROPIC_API_KEY=sk-ant-api03-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    now    = datetime.now(JST)
    week   = now.strftime("%Y/%m/%d")

    print("=" * 60)
    print(f"  フジコ法 週次エージェントチーム")
    print(f"  実行日時: {now.strftime('%Y-%m-%d %H:%M:%S JST')}")
    print("=" * 60)

    # ---- データ収集 ----
    print("\n📊 週次データ収集中...")
    weekly_data = collect_weekly_data()
    data_summary = json.dumps(weekly_data, ensure_ascii=False, indent=2)

    # ---- 01_監督 ----
    print("\n👔 01_監督")
    msg_kantoку = f"""
今週（{week}）のシグナルとログデータです。約定照合レポートを作成してください。

【今週のデータ】
{data_summary}

※現在はドライランフェーズのため実際の約定はありません。
シグナルの生成状況とエラーの確認のみ行ってください。
"""
    out_kantoku = run_agent(client, "01_監督", msg_kantoку)

    # ---- 02_分析 ----
    print("\n📈 02_分析")
    msg_bunseki = f"""
今週（{week}）の市場レジーム分析を行ってください。

【今週のシグナルデータ】
{data_summary}

RSRランキングの変化とTOPIX全体の方向性から、
フジコ法の継続・調整・様子見のいずれかを推奨してください。
"""
    out_bunseki = run_agent(client, "02_分析", msg_bunseki)

    # ---- 03_批判 ----
    print("\n🔥 03_批判")
    msg_hihan = f"""
以下の今週の報告に対して、辛口の批判的分析を行ってください。

【監督レポート】
{out_kantoku}

【分析レポート】
{out_bunseki}

【今週のデータ】
{data_summary}

論理的欠陥・過信・運による成功の可能性を鋭く指摘してください。
"""
    out_hihan = run_agent(client, "03_批判", msg_hihan)

    # ---- 04_設計 ----
    print("\n⚙️  04_設計")
    msg_sekkei = f"""
以下の3つのレポートを踏まえ、Pythonコードの改善提案を行ってください。

【監督レポート】
{out_kantoku}

【分析レポート】
{out_bunseki}

【批判レポート】
{out_hihan}

具体的なファイル名・行番号・変更前後のコードを示してください。
優先度をつけて「今週やること」と「来週持ち越し」を仕分けてください。
"""
    out_sekkei = run_agent(client, "04_設計", msg_sekkei)

    # ---- 05_総括 ----
    print("\n🎯 05_総括")
    msg_sokatsu = f"""
以下の4エージェントの報告を統合し、来週の運用方針を決定してください。

【監督レポート】
{out_kantoku}

【分析レポート】
{out_bunseki}

【批判レポート】
{out_hihan}

【設計レポート】
{out_sekkei}

50歳リタイアへの進捗と来週の方針を明確に示してください。
"""
    out_sokatsu = run_agent(client, "05_総括", msg_sokatsu)

    # ---- 出力保存 ----
    date_str = now.strftime("%Y%m%d")
    outputs = {
        "01_監督": out_kantoku,
        "02_分析": out_bunseki,
        "03_批判": out_hihan,
        "04_設計": out_sekkei,
        "05_総括": out_sokatsu,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUTS_DIR / f"weekly_report_{date_str}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 週次エージェントレポート {week}\n\n")
        for name, content in outputs.items():
            f.write(f"---\n\n## {name}\n\n{content}\n\n")

    print(f"\n✅ 完了")
    print(f"📄 レポート保存: {output_file}")
    print("\n" + "=" * 60)
    print("  05_総括 — 来週の運用方針")
    print("=" * 60)
    print(out_sokatsu)


if __name__ == "__main__":
    main()
