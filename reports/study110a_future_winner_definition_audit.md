# Study110A — Future Winner Definition Audit（実行結果・Phase1=7ラベル）

**日付**: 2026-07-22
**性格**: 実行結果（記述統計のみ・新規データ取得ゼロ・alpha提案なし・backtestなし）。
**入力**: Universe C（`backtests/study75_rule_universe.json`）+ 価格キャッシュ（Study95と同一ソース）
**スクリプト**: `src/backtest/study110a_future_winner_definition_audit.py`
**出力**: `backtests/study110a_future_winner_definition_audit_2026-07-22.json` / `backtests/study110a_panel_enriched_2026-07-22.csv`
**QA**: panel行数=108,895——Study95/95Eと完全一致。同一Universe/価格ソース再利用を確認。

---

## 0. 目的

```
未来勝者の定義を確定すること。
3M winner = 12M winner? or 全く別物?
```

Phase1ラベル（7種）: 3M/6M/12M × Top10%・Top5%(raw) + 12M Calmar調整Top10%。

---

## 1. Winner Overlap Matrix

### 主判定（3M/6M/12M Top10・Jaccard平均）

| Pair | Jaccard |
|---|---|
| 3M_top10 vs 6M_top10 | 0.334 |
| 3M_top10 vs 12M_top10 | **0.197** |
| 6M_top10 vs 12M_top10 | 0.315 |
| **平均** | **0.282** |

**判定: Case2（partial multi-sleeve, 20-50%）**——ただし3M vs 12Mは0.197でCase3境界(<20%)のほぼ真上。短期winnerと長期winnerは6Mを介してわずかに橋渡しされるが、3Mと12Mの重なりはほぼ無い。

### 副次的発見（7x7 full matrix・`backtests/study110a_future_winner_definition_audit_2026-07-22.json`参照）

- **同一horizon内のtop10 vs top5**: 3M=0.503・6M=0.503・12M=0.503——ネストされたランク集合の理論値通り（健全性確認）。
- **12M raw Top10 vs 12M Calmar Top10 = 0.611**——同一horizon内でのrisk-adjust変更による影響は、horizon変更の影響（0.20-0.33）より小さい。**「どのhorizonで見るか」の方が「どうrisk-adjustするか」より支配的な変数**。

---

## 2. Persistence / Transition

| Label | Strict persistence | Loose(Top20%) persistence | Random baseline(strict) | Lift |
|---|---|---|---|---|
| 3M_top10 | 14.9% | 24.8% | 10% | 1.49x |
| 3M_top5 | 10.7% | 26.0% | 5% | 2.14x |
| 6M_top10 | 14.0% | 22.7% | 10% | 1.40x |
| 6M_top5 | 9.5% | 23.4% | 5% | 1.89x |
| 12M_top10 | 11.8% | 20.4% | 10% | 1.18x |
| 12M_top5 | 8.3% | 20.7% | 5% | 1.66x |
| **12M_calmar_top10** | 9.4% | **17.9%** | 10% / 20%(loose) | 0.94x / **0.89x（chance未満）** |

**全ラベルでstrict persistenceは10-15%程度に収束——randomからのlift幅は1.2〜2.1倍で、決して「強い持続性」ではない**。特に**12M_calmar_top10はloose persistenceが17.9%とrandom baseline(20%)を下回る**——risk-adjusted winnerは翌年、広いbandですら平均並み以下にしか残らない。これは「良好なCalmar比は一時的な低ボラ/低DD環境の反映であり、その環境自体が翌期には平均回帰する」ことを示唆する（過剰解釈は禁物・観察のみ）。

---

## 3. Concentration Statistics

| Horizon | Top1% | Top2% | Top5% | Top10% |
|---|---|---|---|---|
| 3M | 14.6% | 22.4% | 38.3% | 55.3% |
| 6M | 14.4% | 22.4% | 38.5% | 55.9% |
| 12M | 15.0% | 23.0% | 39.2% | 56.5% |

全期間で総プラスリターンの約55-56%が上位10%銘柄に集中——リターンは有意に偏っているが、「上位1%だけが全てを作る」という極端な集中（懸念されていたシナリオ）ではない。株式市場で一般的に報告される集中度と整合的なレンジ。

---

## 4. Decision Tree適用

```
Case1: Overlap高い + Persistence高い → Study112へ
Case2: Overlap低い → 複数Universe化
Case3: Persistence低い → Universe研究停止
```

**Overlap軸**: Case2判定（平均28%・3M-12M間は実質Case3境界）。
**Persistence軸**: 全7ラベルでstrict persistenceがrandom baselineの1.2〜2.1倍にとどまり、risk-adjusted版はrandom未満——**「低い」と読める水準**（ユーザー指示に定量閾値の明示はないため、random baseline比較で判定）。

**両条件が同時に成立し得る構成——ツリーに優先順位の明示なし**。CLD見解としては、Overlap問題（Case2＝複数Universe化で対処可能）よりPersistence問題（Case3＝そもそも未来勝者予測が困難）の方が根本的かつ深刻と判断する。理由: 複数Universe化してもそれぞれのUniverse内でwinner persistenceがrandom水準に近ければ、Universe Generatorが「拾うべき対象」自体が翌期にはほぼ入れ替わってしまい、Generatorの構造をいくら工夫しても改善余地が乏しい。**Persistenceが一次制約**という整理を提案する。

---

## 5. 禁止事項遵守確認

新規alpha提案なし。最適化なし。新規データ取得なし（Study95と同一4ソースのみ）。backtestなし（Q10-Q1スプレッド等のトレーディング統計は一切算出していない・ラベルは記述分類のみ）。

*作成: CLD (Fable 5)・2026-07-22。実行済み・既存キャッシュのみ再利用・新規データ取得なし。*
