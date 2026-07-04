# Study80A — Observation Infrastructure & CAP_MISS Root Cause Foundation

**日付**: 2026-07-04
**性格**: 改善研究ではない。恒久的な観測基盤の構築（新規BTは合計1回のみ許可・使用）。
**Parity**: **PASS（完全一致）** — 詳細→`reports/parity_report.md`。CAGR=11.22%/n_trades=309/Sharpe=0.564/MaxDD=-18.22%/Calmar=0.616、全て変更前基準値と完全一致。

---

## 背景

Study74B-RCAは「見送り候補RSR中央値=採用候補RSR中央値=81.0（品質差なし）なのに、max_positions緩和はCAGR悪化する」という矛盾を**解決できなかった**。原因は解析能力不足ではなく、**見送り候補(date,symbol)が個別レコードとして永続化されていなかったこと**。本Studyの目的は、この問題を二度と起こさないための恒久的観測基盤を構築すること。

---

## Part1: Observation Infrastructure（実装完了）

`composite_alpha_bt.py`に観測専用の拡張を実施（詳細スキーマ→`reports/observation_schema.md`）:

- 日次コンテキストスナップショット（cash_before_entry/used_slots/max_slots/selected_symbols/selected_scores/position_weights/candidate_count_today/market_regime）を候補ループ直前に1回計算。
- 候補ごとにmomentum_63d_pct（63日リターン）・sectorを追加算出。
- 既存の候補ログ4種（`_missed_cands`/`_skip_detail`/`_rejected_by_lot_detail`/`_admitted_by_ratio_detail`）に上記コンテキスト + `skip_reason`ラベルを追加。
- **新規**`_selected_cands`リストを追加し、SELECTED候補も同一スキーマで記録（採用/見送りの直接比較が可能に）。

**結果**: 今回のfresh run（FULL 2018-2025）で**607件の見送り候補**（CAP_MISS 448 / SECTOR_CAP 75 / CLUSTER_CAP 65 / GROSS_EXPOSURE 2 / LOT_REJECT 15 + rescued(ADMITTED_BY_RATIO) 0件）を個別レコードとして永続化した（`backtests/missed_candidates_full.json`）。

## Part2: Forward Return Framework（実装完了）

見送り候補607件全件にForward Return（forward_5/10/20/40/60・MFE・MAE・最大DD・holding_days_equivalent）を付与（`backtests/forward_return_dataset.json`）。**BTではなく価格データの直接参照のみ**で計算（`run_scenario`不使用）。

## Part3: Trade Dataset v2（実装完了）

採用トレード309件に candidate_rank / candidate_count / cash_before_entry / slots_used / portfolio_exposure / entry_cluster_id / entry_sector / entry_regime を付与（`backtests/trade_dataset_v2.json`）。

## Part4: Correlation Observation（実装完了・新規ロジックなし）

同日に複数候補が発生した192日について、セクター集中度・モメンタム同方向率を記録（`backtests/correlation_dataset.json`）。**平均最大セクター集中度=63.8%、平均モメンタム同方向率=91.0%**。

## Part5: Opportunity Cost Framework（実装完了）

採用トレードと見送り候補のforward_20をSector別/Regime別/Rank別/skip_reason別に比較可能な構造で保存（`backtests/opportunity_cost_dataset.json`）。

## Part6: 統計設計（実装完了）

`src/backtest/study81_analysis_template.py` — Mann-Whitney U・KS検定・Permutation Test・Bootstrap CIを実装し、Study81が追加BTなしに即実行できる状態にした。

## Part7: Parity（確認完了）

`reports/parity_report.md`参照。CAGR/Trades/Sharpe/MaxDD/Calmar全て完全一致。

---

## 副産物: Study74B-RCA未解決事項への手がかり（本Studyの主目的ではないが重要な進展）

観測基盤を構築し実データで検証した結果、以下の**統計的に有意な**手がかりが得られた（インフラ構築の副産物・Study81での本格検証を推奨）:

### 1. RSR差は「見送り理由を区別すると」有意（Mann-Whitney U）

CAP_MISS単独では品質差なし（Study74B確認済み）だが、SECTOR_CAP/CLUSTER_CAP/LOT_REJECTを含む607件全体で見ると、**採用(中央値81.0) vs 見送り全体(中央値83.3)でp=0.0355（有意）**。見送り全体としてはむしろRSRが高い側に偏る傾向がある。

### 2. セクター集中度は偶然を有意に上回る（Permutation Test）

同日複数候補の平均最大セクター集中度（実測63.8%）は、母集団のセクター頻度分布に基づくランダム配分（permutation平均57.26%、p95=59.64%）を**有意に上回る（p=0.0）**。同日に発生する候補は偶然以上に同一セクターへ偏る。

### 3. 同日群 vs 日をまたぐ群の分散縮小率に大差（Bootstrap比較・最重要）

- 日をまたいだ無作為3件抽出の分散縮小率: **67.3%**（理論値≈66.7%と整合＝真に独立なら期待される水準）
- **同日に実際に競合していた3候補群**の分散縮小率: **24.8%**（大幅に低い）

**これはStudy74B-RCAの仮説「見かけの分散が実質的な相関の高い集中になっている」を初めて定量的に裏付ける結果**。同日に複数の良質候補が現れる場合、それらは互いにほぼ独立ではなく、共通のセクター・モメンタム方向を持つ「隠れた集中リスク」であり、max_positions緩和によって同時に保有しても期待される分散効果（67%）の3分の1程度（25%）しか得られない。これがmax_positions緩和がCAGRを改善しない一因である可能性が高い。

### 4. rank0見送り率は依然として過半数（Bootstrap CI）

607件ベースでもrank0(最上位候補)見送り率=63.6%（95%CI [60.5%, 66.9%]）— 引き続き「質の低い候補の除外」ではなく「最良機会の喪失」であることを裏付ける。

**結論**: Study74B-RCAで「仮説止まり」だった因果メカニズムに、本Studyの観測基盤によって初めて**統計的に有意な裏付け**が得られた。ただし本Studyの主目的はあくまで基盤構築であり、上記は今後Study81で正式に検証・報告すべき先行知見として申し送る。

---

## 成果物一覧

| ファイル | 内容 |
|---|---|
| `reports/observation_schema.md` | 全JSON・全フィールドのスキーマ定義 |
| `reports/parity_report.md` | エンジン変更の無害性検証（PASS） |
| `backtests/trade_dataset_v2.json` | 採用トレード309件（v2拡張フィールド付き） |
| `backtests/missed_candidates_full.json` | 見送り候補607件（個別レコード・Forward Return付き） |
| `backtests/forward_return_dataset.json` | Forward Return専用ビュー |
| `backtests/opportunity_cost_dataset.json` | Opportunity Cost（Sector/Regime/Rank/skip_reason別） |
| `backtests/correlation_dataset.json` | 同日候補の集中度観測 |
| `src/backtest/study81_analysis_template.py` | 統計解析テンプレート（7関数・即実行可能） |
| `src/backtest/study80a_observation_infrastructure_2026-07-04.py` | 本Study実行スクリプト（唯一の新規BT） |
