"""
tests/test_run_morning_signal_entry_rsr.py

2026-07-08 RCA 回帰テスト: run_morning_signal.py の
bridge.update_state_after_execution(send_results, today_str) 呼び出しが
signal_rsr_map を渡していなかったため、このスクリプト経由のBUYは
position_entry_rsrs が常に欠落していた（entry_rsr missing proxy行きが確定的）。

run_morning_signal.py は巨大な単一 main() 関数のため呼び出しを直接実行する
統合テストは実用的でない。呼び出し箇所が signal_rsr_map を渡している
ことをソースレベルで固定する回帰テストとする。
"""
from __future__ import annotations
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "run_morning_signal.py"


def _find_update_state_calls() -> list[ast.Call]:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update_state_after_execution"
        ):
            calls.append(node)
    return calls


def test_update_state_after_execution_passes_signal_rsr_map():
    calls = _find_update_state_calls()
    assert calls, "update_state_after_execution() 呼び出しが見つからない"
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "signal_rsr_map" in kw_names, (
            "bridge.update_state_after_execution() が signal_rsr_map を渡していない "
            "→ position_entry_rsrs が記録されず QR proxy 行きが確定する (2026-07-08 RCA)"
        )
