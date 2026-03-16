# research_state.md — CHIBAAssetProject 研究状態
# Single Source of Truth / 最終更新: 2026-03-15
# ⚠ 会話メモリは信用しない。必ずこのファイルから状態を復元すること。

---

## プロジェクト概要

**ゴール**: 50歳でリタイア（月30万円超 × 12ヶ月）
**口座**: auカブコム証券（特定口座・開設済み）
**API**: kabuステーション REST API（localhost:18080）
**資本**: 200万円
**現在フェーズ**: Phase 2（少額実運用）開始

| フェーズ | 基準 | 状態 |
|---|---|---|
| Phase 1 | Sharpe>0.5、MaxDD<20% | ✅ 達成（Sharpe=1.7、MaxDD=-7.8%） |
| Phase 2 | 月1〜5万円 × 3ヶ月連続 | 🔄 実運用テスト中 |
| Phase 3 | 月20万円 × 6ヶ月、DD<15% | ⬜ 未着手 |
| Phase 4 | 月30万円超 × 12ヶ月 | ⬜ 未着手 |

---

## 現在の最良戦略（V2設定）

**スクリプト**: `run_live_signal.py`（実運用）/ `backtest/portfolio_v2.py`（バックテスト）

| 指標 | 値 |
|---|---|
| CAGR | +16.26% |
| Sharpe | 1.693 |
| MaxDD | -6.12% |
| Calmar | 2.656 |
| 平均保有銘柄数 | 1.6 |
| 資本稼働率 | **約17%**（= 83%キャッシュ ← 最大の課題） |

**パラメータ**: `configs/strategy.yaml` 参照
**ユニバース**: `configs/universe.yaml` 参照（G29_V2: 29銘柄）

---

## 検証済み項目（再実行不要）

### ✅ 3重クロス検証（2026-03-14）
- ルックアヘッドバイアス: なし（独立Vectorized実装と全指標一致）
- 過学習: なし（ウォークフォワード OOS/IS=0.98）
- 詳細: `results/backtest_summary.json` → `triple_validation`

### ✅ 頑健性総合検証（2026-03-15）
- Monte Carlo N=2000: Sharpe>1.0 確率=100%
- パラメータ感度: 滑らかな感度曲線（過学習なし）
- 銘柄サブセット20試行: Phase1達成率100%
- 詳細: `results/backtest_summary.json` → `robustness_analysis`

---

## 完了した研究（2026-03-16）

### ✅ 外部レビュー対応 4タスク統合検証（2026-03-16）
**スクリプト**: `backtest/advanced_analysis.py`（新規）
**背景**: 外部レビュー「スクリーニングによる過学習」への対応

**重要な結論**:
- Ex-ante 109銘柄（スクリーニングなし）: CAGR -0.7%〜-2.2%（全設定マイナス）
- 年間シグナルは516件あり十分。問題は「銘柄の適合性」
- **仮説「稼働率がボトルネック」→ 棄却**
- **G27スクリーニング維持を推奨**（廃止は逆効果と確認）

**セクター適合性の確定（フジコ法が機能するセクター）**:
電機精密(Sharpe2.83) / 輸送機器(2.92) / 鉄鋼(2.57) / 機械(2.16) / 電機(1.76) / 海運(1.27) / 商社(1.38)

**除外候補**: 小売（3382.T）勝率18.2%・Sharpe-1.24

---

## 次の研究タスク（優先順）

| 優先度 | タスク | 根拠 |
|---|---|---|
| **1** | 3382.T（小売）除外効果バックテスト | 勝率18%・Sharpe-1.24は明確な損失源 |
| **2** | 2025年実運用OOS検証 | バイアス排除の唯一の方法 |
| **3** | 1.5xレバレッジ実務検討 | MaxDD-10%なら-15%以内、CAGR+24.7% |
| **4** | サバイバルバイアス定量評価 | 2018-2024上場廃止銘柄を追加して差分測定 |

---

## ファイル構成

```
asset_simulation/
├── research_state.md          ← このファイル（Single Source of Truth）
├── configs/
│   ├── strategy.yaml          ← 戦略・ポートフォリオパラメータ
│   └── universe.yaml          ← 銘柄ユニバース設定
├── results/
│   ├── backtest_summary.json  ← 全バックテスト結果サマリー
│   └── universe_expansion.json ← ⏳ 作成待ち
├── research_log/
│   └── 2026-03-15.md          ← 本日の研究ログ
├── backtest/
│   ├── portfolio_v2.py        ← 最良バックテスト（Calmar=2.656）
│   ├── portfolio_cross_validate.py ← 3重クロス検証
│   ├── robustness_analysis.py ← 頑健性総合検証（2026-03-15新規）
│   ├── universe_expansion.py  ← ユニバース拡張検証（2026-03-15新規）
│   └── ...
├── kabusapi/
│   ├── client.py              ← kabuステーション APIクライアント
│   └── signal_bridge.py       ← シグナル→発注ブリッジ
├── run_live_signal.py         ← ★実運用エントリーポイント（V2設定）
└── data/
    └── signals/               ← 発注シグナルJSON（.gitignore対象）
```

---

## 重要な既知問題・注意事項

| 項目 | 内容 |
|---|---|
| サバイバルバイアス | 現在構成銘柄のみ使用。廃止銘柄除外。CAGR 1〜3%過大評価の可能性 |
| 銘柄選択バイアス | 現行27銘柄はin-sample screeningで選定（改善中） |
| SELL/BUY非対称 | SELL=当日終値/BUY=翌日始値。CAGR差≈0.2%（許容範囲） |
| キャッシュ比率 | 平均83%キャッシュ → 資本効率が低い（改善中） |
| 株価上限 | 資本200万×25%=50万 → 株価5,000円以下の銘柄のみ購入可能 |

---

## セッション復元手順

新しい会話でプロジェクトを再開する場合:
```
1. このファイル（research_state.md）を読む
2. configs/strategy.yaml と configs/universe.yaml を読む
3. results/backtest_summary.json で最新結果を確認
4. research_log/ の最新日付のログを読む
5. 作業開始
```
