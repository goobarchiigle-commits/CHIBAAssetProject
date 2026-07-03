# logs/sim/ — シミュレーション専用ログ置き場

## 目的
バックテスト・感度テスト・仮想実行など「本番ではない」実行のログを
本番ログ（logs/ 直下）と分離するためのディレクトリ。

## ルール
| ログ種別 | 保存先 |
|---|---|
| 本番シグナル (`--live`) | `logs/runtime/` |
| CB イベント（本番） | `logs/cb_events/YYYYMMDD.jsonl` |
| CB イベント（シミュレーション・未来日付）| `logs/cb_events/archive/` |
| バックテスト実行ログ | `logs/sim/` ← ここ |
| 感度テスト・WF ログ | `logs/research/` |

## 注意
- `cb_20260502.jsonl` / `cb_20260510.jsonl` は 2026-04-04 時点で未来日付のため
  `logs/cb_events/archive/` へ移動済み（2026-04-04）。
- シミュレーションスクリプトは `LOGS_DIR / "sim"` 配下に書き出すこと。
  本番の `cb_events/` に書き込まないよう `_save_cb_event()` の呼び出し元を確認すること。
