# 再ベースラインに必要なバックテスト実行コマンド（v202606）

作成日: 2026-06-24
前提: `production_research_diff_v202606.md` で確定した通り、Study9〜29系列は本番未実装の仮想戦略（RSR[92,95) d90≤5 slope5≤5 entry / RSR<90 exit）を評価していた。本番の実体（`FujikoStrategy` + `signal_bridge.py`の優先順位付きExit）を実際に再現するエンジンは **`src/backtest/composite_alpha_bt.py`** である（`load_strategy_config()`経由で`strategy.yaml`を直接読み込み、`FujikoStrategy`/`MeanReversionStrategy`を使用、`shock_exit_mode`/`dyn_rsr42_bear_rs0`/sector・cluster capを全て反映 — `composite_alpha_bt.py:1092-1356`で確認済み）。

実行はすべて `cd C:/ai-trading` を起点とする（CLAUDE.md `run_from=C:/ai-trading`）。CLAUDE.md PERMISSIONにより、**実行（バックテスト読み取り実行）はAUTO_OK**、新規スクリプト作成・既存スクリプトの大規模改修はASK_FIRST。本書はコマンド提示のみ、実行は別途判断。

---

## 1. 本番ロジック単体の現在地確認（BASELINE再実行）

```bash
cd C:/ai-trading
python src/backtest/composite_alpha_bt.py --full-history --full-scenarios
```
- `--full-history`: 2018-2024フル期間（`composite_alpha_bt.py:1097`）
- `--full-scenarios`: BASELINE〜STEP6Bの全シナリオ実行（`:1098`）。**本番と完全一致する評価対象はBASELINEのみ**（STEP1以降は研究用の追加オーバーレイ比較）。
- 出力: `backtests/composite_alpha_bt_rsr42_full_<date>.json`
- 既存参照値（`generate_strategy_spec.py`出力§13より）: IS CAGR=22.4%/Sharpe=1.582/MaxDD=-12.32%、OOS(2025) CAGR=0.1%/Sharpe=0.067。**データが2026-04時点のものである可能性が高く、2026-06現在のデータで再実行し最新値を確認すべき**。

---

## 2. 動的ユニバース込みWalk-Forward再検証

```bash
cd C:/ai-trading
python src/backtest/wf_dyn_rsr42.py
```
- `dyn_rsr42_bear_rs0`設定（本番採用設定, `strategy.yaml:101-117`）を含む4設定をWF比較（`wf_dyn_rsr42.py:50-79`）。
- 判定基準: Seg3_2022 OOS Sharpe>0 / WF 5/5 / 2025 OOS Sharpe≥0.80（`wf_dyn_rsr42.py:82,91-94`）。
- 出力: `backtests/wf_dyn_rsr42_<date>.json`
- 既存参照値（2026-04-05時点）: WF 5/5、Full IS Sharpe=0.812、True OOS(2025) Sharpe=0.805。**最新データで再実行し、2025年後半〜2026年データを含めた再評価が必要**（既存結果は2025年早期までのデータと推定される）。

---

## 3. Study9〜29系列を本番ロジックで再実行（最重要・新規実装が必要）

現状、`src/backtest/study9_standalone_validation.py`およびStudy20〜29の各スクリプトは独自のRSR[92,95)ロジックを内蔵しており、**`FujikoStrategy`をimportしていない**（`production_research_diff_v202606.md` §中心的発見 参照）。これらの研究結論（EXHAUSTED/RESEARCH_EXIT等）を本番に対して意味のあるものにするには、以下のいずれかが必要:

### 選択肢A（推奨・低コスト）: 既存研究を「本番非適用」として明示的にクローズし、composite_alpha_bt.py上で同種の研究設問を再設定
Study25(Portfolio Geometry)/Study27(Risk Activation)/Study28(Allocation)/Study29(Exit Monitoring)が問うた設問（「資金配分だけでCalmar改善余地はあるか」等）自体は本番にも有効な問いである。`composite_alpha_bt.py`の`run_scenario()`が返す実際のトレードログ（本番FujikoStrategyベース）を入力に、同じ監査ロジック（`attribute_trade()`等、Study21由来）を再適用する新規スクリプトが必要。
**この新規スクリプト作成はCLAUDE.md PERMISSIONの「新規スクリプト作成」に該当 → 実装前にASK_FIRST。**

設計概要（実装許可後の参考骨子）:
```bash
# (実装予定・現時点では存在しない)
python src/backtest/study30_production_faithful_baseline.py \
    --use-fujiko-strategy \
    --capital 3000000 --max-positions 3 \
    --start 2018-01-01 --end 2025-12-31
```
- `composite_alpha_bt.py: run_scenario(scenario="BASELINE", ...)`が生成するトレードログをそのまま使用（独自再実装をしない）。
- Study21の`attribute_trade()`（tail_capture/profit_left計算）をそのトレードログに適用。
- これにより「本番が実際に生成するトレードのExit品質」を初めて正しく評価できる。

### 選択肢B（即時実施可・コード変更不要）: 既存のCase Bパラメータを`FujikoStrategy`が表現できるか確認する感度分析
`FujikoStrategy`はRSR floor方式（`rsr>=min_rsr`）のみで上限・d90・slopeを持たないため、Case Bの狙い（「RSRが90を超えた直後・急騰直後を避ける」）を**既存パラメータの範囲で近似**できるか、`min_rsr`の感度分析で確認する:
```bash
cd C:/ai-trading
python src/backtest/min_rsr_sensitivity.py
```
（既存スクリプト、`src/backtest/min_rsr_sensitivity.py`、追加実装不要。出力で`min_rsr=75`が現在最適点か、`min_rsr`を92近傍まで上げた場合にCAGR/Calmarがどう変化するかを確認できる。Study9のRSR[92,95)という「狙い」に最も近い既存ツール。）

---

## 4. `generate_strategy_spec.py`のバグ修正（ドキュメント生成の正確性）

`production_research_diff_v202606.md` §8で指摘した通り、`src/generate_strategy_spec.py:549`が exit閾値ラベルに`min_rsr`(75.0)を誤用している（実際の本番exit閾値は`rsr_exit`=70.0）。

```bash
# 修正後に再生成して確認（軽微な既存スクリプト修正のためASK_FIRST対象 — 大規模改修ではないが念のため確認推奨）
cd C:/ai-trading
python src/generate_strategy_spec.py --out docs/research/strategy_spec_2026-06-24.md
```
修正内容: `generate_strategy_spec.py`内で exit RSR閾値表示に`min_rsr`変数ではなく、`strategy.yaml: fujiko.rsr_exit`（`_c("fujiko", "rsr_exit", 70.0)`のような独立した取得）を使用するよう修正。

---

## 5. 実行優先順位

| 優先 | コマンド | 目的 | 新規実装要否 |
|---|---|---|---|
| 1 | `python src/backtest/composite_alpha_bt.py --full-history --full-scenarios` | 本番ロジックの現在の実績を最新データで確認 | 不要 |
| 2 | `python src/backtest/wf_dyn_rsr42.py` | 動的ユニバース込みWF再検証（最新データ） | 不要 |
| 3 | `python src/backtest/min_rsr_sensitivity.py` | Study9の狙い（RSR上限/直近急騰除外）を既存パラメータ空間で近似評価 | 不要 |
| 4 | Study30（本番ロジックのトレードログにStudy21のattribution手法を再適用） | Study20-29の研究的問いを本番に正しく適用し直す | **要・ASK_FIRST** |
| 5 | `generate_strategy_spec.py`のexit閾値バグ修正 | ドキュメント生成の正確性確保 | 軽微・確認推奨 |

優先1-3は既存スクリプトの再実行のみでCLAUDE.md AUTO_OK範囲。優先4はEntry/Exitロジックを変更するものではなく観測専用の新規分析スクリプトだが、CLAUDE.mdの定義上「新規スクリプト作成」に該当するため着手前に確認を取ること。
