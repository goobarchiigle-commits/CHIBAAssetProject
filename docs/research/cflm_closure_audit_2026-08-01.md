# CFLM Closure Audit（2026-08-01）

**性質**: Study111（Sector Layer Gate=CLOSED）後の最終監査。新規feature mining・CFLM延命が目的ではない。既存研究（Study95/98/99/111）で閉じられる論点は閉じ、唯一未検証だったStock↔Sector lead-lag構造のみを一度確認した上で、CFLMの最終Dispositionを機械的に確定する。

**新規production code変更なし・PARAMS_LOCKED変更なし・J-Quants追加利用なし・新規データ取得なし・33業種新規BTなし・新規Study番号なし。**

新規実施は1点のみ: `src/backtest/cflm_closure_stock_sector_leadlag.py`（Study111既存関数を再利用する一回限りの診断スクリプト・ユーザーASK_FIRST承認済み・出力=`backtests/cflm_closure_stock_sector_leadlag.json`）。

---

## ① H3/H4 Formal Evidence（PIT-clean・公式TOPIX-17指数ベース）

Study111既存結果の再確認（再計算なし）。

| Hypothesis | Verdict | Discovery spread | Validation spread | Validation \|NW-t\| |
|---|---|---|---|---|
| H3_sector_rank_change | **REJECT** | +0.21% (NW-t=0.56) | +0.59% (NW-t=0.90) | 0.90 < 2.0（有意性不足） |
| H4_sector_rank_acceleration | **REJECT** | +0.50% (NW-t=1.34) | -0.22% (NW-t=-0.28) | 符号不一致 |

年度別（20d horizon mean IC）:
- H4: 2018-2020は強い正（+0.09〜+0.15）、2021-2022で弱化、validation期(2023-2024)で明確に負転換（-0.09/-0.11）、2025-2026で弱く回復——特定レジーム依存の一時的パターンで構造的edgeではない。
- H3: 年度別振れが大きく系統的パターンなし。discovery/validationとも|NW-t|<1、効果量がゼロ近傍。

**結論**: 正式判定可能な2項目（PIT健全・分類非依存）は共にREJECT。OOS edge確認せず。

---

## ② H1/H2/H5 Limitation（classification non-PIT）

`database/market/master/classifications.parquet`は単一スナップショット（2026-07-31取得）。過去時点のセクター所属を復元不可能なため、H1/H2/H5は正式なKEEP/REJECT判定に使用できない（MEASUREMENT_LIMITATION固定）。

Study111観察事項の再掲のみ（新規解釈を追加しない）:
- H1（trading value accel）・H5（breadth expansion）: discovery→validationで符号反転、不安定。
- H2（volume accel）: 60d horizonのみ discovery/validation双方で同符号（負）・validationで強まる（t=-0.88→-2.38）。ただし符号は「早期参加→将来outperformance」という当初仮説と**逆**（volume急増後にrelative underperformance）。CFLMが想定した資金流入初期捕捉効果の証拠ではない。

**確認事項**: 「CFLMを救済する明確な兆候があるか」→ **無し**。H2 negative resultをEDMR等の新仮説へ接続しない（タスク指示通り・本Auditでも接続していない）。

---

## ③ RGP-3横断評価（Study95 → Study98 → Study99 → Study111）

| Study | Operationalization | 主要結果 | OOS/Validation一貫性 |
|---|---|---|---|
| Study95 | Level（12-1 momentum, Clenow slope×R²） | FAIL_ZERO_SPREAD（12M NW-t=-0.368 / -0.669）・Kill機械発動 | 全horizon spread≈0 |
| Study98 | Persistence（P(sector持続\|超過)） | 全量1M z=1.547（非有意）・上位3限定3M z=3.228のみ有意 | 単一条件のみ境界的有意、他は非有意 |
| Study99 | Interaction（Sector×RS×25MA乖離、静的） | 2016-2020 vs 2021-2025で符号反転（t(FM)=-2.07→+2.89等） | **符号反転**（sub-period不安定） |
| Study111 | Change/Acceleration（PIT-clean分のみ） | H3/H4 REJECT | 符号不一致 or 効果ゼロ近傍 |

**判断**: level・persistence・interaction・change・accelerationの5つのoperationalizationいずれも、OOSで再現性のある一貫したedgeを示していない（Study98のみ部分的有意だが単一条件・他条件で非有意という選択的結果であり、Study99の符号反転パターンと合わせ、多重比較下のノイズと解釈するのが妥当）。**RGP-3系統の追加operationalizationを続ける合理性は残っていない。**

---

## 新角度: Stock↔Sector Lead-Lag構造診断（探索的・一回限り）

### 手法（事前固定）

`cflm_closure_stock_sector_leadlag.py`。Study111既存関数（`load_ohlcv_panel`/`build_sector_daily_series`/`accel_signal_from_level`/`breadth_expansion_signal`/`build_official_rank_signals`）を再利用。新規処理は個別銘柄側accel signal（Study111と同一式をper-Code適用）とsector-day別「個別accel>0の銘柄比率」（stock-side participation breadth）の構築のみ。forward-return targetは一切未使用。

6ペアについて、lag=-10〜+10営業日でPearson相関（sector別z-score後pooled 17業種）を計算。分類規則（事前固定）: argmax\|corr\|のlag≤-2→Sector→Stock、≥+2→Stock→Sector、-1〜+1→Same-day。

### 結果

| ペア | Pooled argmax lag | Pooled \|corr\| | 方向 | 年度別一貫性 |
|---|---|---|---|---|
| turnover_participation ↔ sector_H1（turnover accel） | 0 | 0.739 | **Same-day** | 完全一致（discovery/validation/全11年度すべてSame-day、corr 0.60-0.82） |
| volume_participation ↔ sector_H2（volume accel） | 0 | 0.679 | **Same-day** | 完全一致（discovery/validation/全11年度すべてSame-day、corr 0.59-0.75） |
| turnover_participation ↔ sector_H5（breadth expansion） | -3 | 0.203 | Sector→Stock（弱） | **不一致**（2017/2021=Same-day、2022=符号反転(-0.14)、2024/2026=Stock→Sector反転） |
| volume_participation ↔ sector_H5（breadth expansion） | -4 | 0.139 | Sector→Stock（弱） | **不一致**（2020=符号反転(-0.23)、2021/2022/2024/2026=Stock→Sector反転） |
| turnover_participation ↔ sector_H3（rank change・PIT-clean） | +2 | 0.064 | Stock→Sector（極弱） | **不一致**（2018/2019=Sector→Stock符号反転、\|corr\|<0.12全期間） |
| turnover_participation ↔ sector_H4（rank acceleration・PIT-clean） | -1 | 0.063 | Same-day（極弱） | ほぼnoise水準（\|corr\|<0.12全期間、方向は年度で散在） |

（全lag×全期間×全年度の詳細数値は`backtests/cflm_closure_stock_sector_leadlag.json`に保存済み。）

### 5項目の確認

1. **Stock-side先行**: H3のみ弱いStock→Sector傾向（+2営業日、\|corr\|=0.06程度）——効果量がノイズ水準で構造的先行とは言えない。
2. **Sector-side先行**: H5のみ弱いSector→Stock傾向（-3〜-4営業日、\|corr\|=0.14-0.20）——ただし2021/2022/2024/2026で方向反転し、regime依存で不安定。
3. **同日発生**: H1/H2ペアで極めて強く一貫（corr 0.6-0.8・discovery/validation/全11年度で例外なくlag=0）。
4. **一貫性**: H1/H2は完全に一貫（構造的）。H5/H3/H4は年度間で符号・方向が反転し、**特定regime/年度依存の非構造的パターン**。
5. **breadth/dispersion/leader emergenceとの関連**: 個別資金流入参加率とprice-based breadth（H5）・rank dynamics（H3/H4）との相関は一貫して弱い（\|corr\|<0.25）。CFLMの核心仮説（participation expansion→sector-level breadth/rank transition）を支持する再現性のある証拠は見つからなかった。

### 解釈上の重要な注記

H1/H2ペアの強い同日相関は**構造的（definitional）関係**である——sector-level turnover/volume accelerationはそもそも個別銘柄TurnoverValue/Volumeの集計から計算されており、個別銘柄側の同一proxy（参加率）が同時に強く相関するのは統計的にほぼ自明（sector aggregate = Σ individual）。これは新しい経済的メカニズムの発見ではなく、データの内部整合性確認に過ぎない。同日相関である以上、予測的に利用不可能（sector側シグナルが観測できる時点で既にstock側も同時に観測されている）。

CFLMが本来検証したかった「参加拡大が段階的に伝播するか」という問い（H5/H3/H4との関係）については、弱く・年度間で不安定な相関以外の証拠はない。

---

## ④ Sector Granularity（未検証論点として記録のみ）

TOPIX-17（17業種）が粗すぎる可能性は**未検証のまま**残る。「17業種でedgeなし ≠ あらゆるsector/theme groupingでedgeなし」——ただしSector33・テーマ分類による追加BTは実施しない（タスク指示通り）。この論点単独を理由にCFLMを継続しない。

## ⑤ Target Specification（確認のみ）

Study111のtargetは一貫してsector-relative forward return（20d primary/60d secondary）だった。今回のStock→Sector構造・breadth・participationは診断目的のみで使用し、結果を見て新しいforward-return targetを後付けしていない（`forward_return_target_used: false`をJSON出力に明記・確認済み）。

---

## 最終Disposition

**適用ケース: CASE A**

判定根拠:
- PIT-clean H3/H4 = REJECT（①）。
- Stock↔Sector lead-lag診断（新角度）: 唯一の強い相関（H1/H2同日）は構造的・非経済的関係であり予測的意味を持たない。経済的に意味のある方向（H5/H3/H4との関係）は全て弱く（\|corr\|<0.25）、年度間で符号反転し**再現性のあるedgeが存在しない**。

→ **CASE Aの条件（PIT-clean H3/H4 REJECT + Stock↔Sector lead-lagにも構造的・再現性のあるedgeなし）を満たす。**

### 明記事項

**「CFLM Sector Layerが否定された」のか、「今回のoperationalizationではedgeが確認できなかった」のか**:
後者。RGP-3 Cross-sectional Momentumという経済メカニズム自体を反証したわけではない。Study95/98/99/111・および今回のlead-lag診断という**複数の独立したoperationalization**（level/persistence/interaction/change・acceleration/lead-lag structure）で一貫してOOS edgeが確認できなかった、という累積的な経験的証拠。

**未検証mechanismとして何が残ったか**:
1. Stock↔Sector lead-lag構造そのものは今回診断したが、classification non-PIT制約のため正式なKEEP/REJECT判定はできない（H1/H2/H5と同一のMEASUREMENT_LIMITATION相当）。
2. Sector33・テーマ分類等、より細かい粒度でのcapital flow transition（④）。
3. H1/H2/H5のnon-PIT問題自体（過去時点セクター所属の復元）。

**それでもIndividual Layerへ進まない理由**:
Sector Layer Gate（Study111）はCLOSEDのまま。新角度診断（Stock↔Sector lead-lag）はSector Layer Gateを再開する根拠を提供しなかった（弱く不安定な相関のみ）。「Sector Layerで確認できない場合、個別銘柄Layerへfeatureを追加して救済する」行為はStudy111冒頭から一貫して禁止されており、本Auditでもこの原則を維持する。

**今後再開するための明確な条件**:
以下のいずれかが将来満たされない限り、CFLM（RGP-3系統のchange/acceleration/participation operationalization）は再開しない。
1. `classifications.parquet`のnon-PIT制約を解消する過去時点セクター所属データが、J-Quants以外の合法的ソースから調達できる（新規データ取得は別途ASK_FIRST）。
2. 独立した新しい経済的仮説（本Auditのスコープ外）が別途pre-registrationを経て提起される。
CASE Dにより、H1/H2/H5のnon-PIT問題「が解決できそう」という理由だけではCFLMを再開しない。

**RGP-3 feature miningをこれ以上継続しない理由**:
5つの独立したoperationalization（Study95/98/99/111+本Audit）全てがOOSで一貫したedgeを示さなかった。これ以上の新しいfeature/threshold/horizonの探索は、事前登録されたhypothesisの検証ではなく事後的なfeature miningになるリスクが高く（multiple comparison下でのfalse discovery）、研究資源の効率的配分に反する。

---

## 次アクション

CFLM（RGP-3系統）は研究棚上げ。研究資源はRGP-2（Event-driven MR・Study82F後継）へ振り替える。EDMRについてはStudy82Fとのoverlap audit→Research Question固定→RGP分類確認→Hypothesis Registry pre-registration→ASK_FIRSTの順序を厳守する（本Auditのスコープ外・別途起案）。

`index_prices.py`の`to_parquet()`単純上書きバグ（Study111で発見・§3参照）は研究とは別件としてASK_FIRSTで修正タスクを起案する。

---

*生成: CFLM Closure Audit, 2026-08-01。新規BT・戦略シグナル・Composite Score・新Study番号一切なし。新規スクリプト1個（診断専用・ユーザーASK_FIRST承認済み）。production code/live signal/Scheduler変更なし。*
