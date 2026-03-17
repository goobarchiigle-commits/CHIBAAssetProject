# research_state.md — CHIBAAssetProject 研究状態
# Single Source of Truth / 最終更新: 2026-03-18
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

### ✅ CB問題・セクタートップ銘柄・CB改善 統合検証（2026-03-16）
**スクリプト**: `backtest/cb_sector_analysis.py`（新規）

**Task 1: セクター内Top-N仮説**
| シナリオ | 銘柄数 | CAGR | Sharpe | MaxDD | Calmar | CB日数 |
|---|---|---|---|---|---|---|
| G27現行（ベースライン） | 19 | +14.0% | 1.541 | -8.5% | 1.639 | 0日 |
| Top1/セクター | 7 | +4.7% | 0.658 | -11.4% | 0.413 | 0日 |
| **Top2/セクター** | **14** | **+9.2%** | **1.012** | **-10.2%** | **0.897** | **0日** |
| Top3/セクター | 21 | +5.4% | 0.569 | -15.1% | 0.359 | 554日 |
| 全銘柄(48) | 48 | +4.3% | 0.461 | -15.3% | 0.280 | 758日 |

**Task 2: CB構造的デッドロック確認**
- 発動後 全キャッシュ → peak_equity固定 → DD=-15.4%永久 → 解除条件(-7.5%)到達不可
- 2021年ワースト: 5706.T鉄鋼(-15.1%), 7011.T機械(-13.2%★G27), 6762.T電機精密(-11.7%★G27)
- 2022-2024 G27∩Sector平均: +119.1% vs Sectorのみ: +61.9% → **G27はin-sample選定で上位株を保有**

**Task 3: CB設計改善比較（Sector Filter 48銘柄）**
| モード | CAGR | Sharpe | MaxDD | CB日数 |
|---|---|---|---|---|
| A: 現行standard | +4.3% | 0.461 | -15.3% | 758日 |
| B: time_limit 120日 | +4.0% | 0.438 | -16.8% | 752日（悪化） |
| C: time_limit 60日 | +3.3% | 0.369 | -20.7% | 746日（悪化） |
| **D: entry_stop_only** | **+13.1%** | **1.029** | **-16.1%** | **175日（✅大幅改善）** |
| E: partial_size 50% | +13.1% | 1.029 | -16.1% | 175日（D同等） |
| F: no_cb | +11.4% | 0.871 | -22.5% | 0日 |

**4問への回答**:
1. **G27高Sharpeの真因**: セクター選択（+0.4 Sharpe）＋in-sample銘柄選択バイアス（+0.5 Sharpe）の両方。Top2/セクターで1.012止まり → 残りは過学習由来
2. **Sector Filterで再現可能か**: Top2/Sec=1.012 vs G27=1.541 → **まだ差がある ❌**
3. **CB改善で2022-2024解決**: D(entry_stop_only)でSharpe 0.461→1.029、2022-2024年次+7.2%/+41.5%/+9.4% → **✅解決**
4. **最再現性の高いUniverse**: Top2/セクター（流動性ランキングで決定） + entry_stop_only CB → OOS検証推奨

---

## 完了した研究（2026-03-17〜18）

### ✅ 朝のルーティン自動化（2026-03-17）
- `morning_dryrun.bat`（8:30）/ `morning_live.bat`（9:00）作成
- Windowsタスクスケジューラ登録（平日自動実行・StartWhenAvailable=True）
- `signal_bridge.py` バグ修正: `dropna(how="all")` → `dropna(subset=["Close"])`
  - 原因: yfinanceが直近営業日のCloseをNaNで返しRSR全体がnanになる問題

### ✅ 段階型DDリスク制御バックテスト（2026-03-17）
**スクリプト**: `backtest/portfolio_dd_control.py`（新規）
**結果**: `results/dd_control_2026-03-17.json`

G27+V2のMaxDD=-7.53%は全スキーム閾値未満 → 758日ロック問題はV2移行で実質解消済み

**ストレステスト**: `backtest/portfolio_stress_test.py`（新規）
**結果**: `results/stress_test_2026-03-17.json`
- 2.0x時: 案AC(-12%entry_stop/-6%DD-only解除)が1508日ロック
- **重要**: entry_stop解除にDD条件のみは禁止。必ず「30〜60営業日 OR TOPIX>MA200」を追加すること

### ✅ 資本効率最大化バックテスト（2026-03-17）
**スクリプト**: `backtest/portfolio_capital_efficiency.py`（新規）
**結果**: `results/capital_efficiency_2026-03-17.json`

| 機能 | CAGR変化 | 評価 |
|---|---|---|
| ① ランキング加重（RSR順位加重, max_pos_weight=0.40） | +16.66%→+18.92%（+2.26pp） | ✅ 採用推奨 |
| ② MIN_POSITIONS=3（補完エントリー） | -6.85pp（稼働率も低下） | ❌ 廃止 |
| ③ 強制ローテーション（diff=5） | ±0（0回/年） | △ Phase 3以降で再評価 |

---

## 次の研究タスク（優先順）

| 優先度 | タスク | 根拠 |
|---|---|---|
| **1** | ①ランキング加重を`run_live_signal.py`に統合 | CAGR+2.26ppの改善をPhase 2実運用に反映 |
| **2** | 案AC修正版（entry_stop解除に時間条件追加）再検証 | DD-only解除の自己ロック問題を修正 |
| **3** | Top2/セクター + entry_stop_only + G27比較バックテスト | CB改善×再現性ユニバースの組み合わせ効果未測定 |
| **4** | 2025年実運用OOS検証 | バイアス排除の唯一の方法 |

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
│   ├── 2026-03-15.md          ← 頑健性検証ログ
│   └── 2026-03-16.md          ← CB問題・セクター分析ログ
├── backtest/
│   ├── portfolio_v2.py        ← 最良バックテスト（Calmar=2.656）
│   ├── portfolio_cross_validate.py ← 3重クロス検証
│   ├── robustness_analysis.py ← 頑健性総合検証（2026-03-15新規）
│   ├── universe_expansion.py  ← ユニバース拡張検証（2026-03-15新規）
│   ├── advanced_analysis.py   ← Ex-ante Universe検証（2026-03-16新規）
│   ├── walk_forward_universe.py ← ウォークフォワード過学習検証（2026-03-16新規）
│   ├── sector_filter_universe.py ← セクターフィルターユニバース（2026-03-16新規）
│   ├── cb_sector_analysis.py  ← CB改善・Top-N仮説（2026-03-16新規）
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
