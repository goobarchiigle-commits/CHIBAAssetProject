# PEAD手法仕様書 3案 — 完全定量化ベンチマーク定義（2026-07-24）

**性格**: 起案書（設計のみ・実装/BT未実行・Freeze Rule非抵触）。
**⚠正典移行（2026-07-24同日）**: 本書の3案は versioned Strategy Specification として `research/strategies/` へFrozen発行済み（1案→`pead/pead_classic_v1.0.md` / 2案→`pead/pead_practical_v1.0.md` / 3案→`edmr/edmr_v1.0.md`）。**以後の参照は仕様書側が正**。本書は起案原本として永続（統治規則=`research/governance/strategy_specification_governance.md`・台帳=`research/strategies/REGISTRY.md`）。
**目的**: 「PEAD」の抽象語を廃し、変更禁止ベンチマーク v1.0 を文献+自プロジェクト実測に基づき固定する。
**文献ソース**: `book.pdf/02_PEAD_Event/`（Bernard&Thomas 1989 / Foster-Olsen-Shevlin 1984経由 / Chan-Jegadeesh-Lakonishok 1996 / Brandt et al. 2008 / Kats&McCubbins 2018）
**自プロジェクト実測ソース**: Study82 PhaseD（30,952イベント・スプレッド-0.538%逆転）/ Study82E（逆転=Small/Mid集中・Large≈0・binary YoY分類の解像度不足）/ Study82F（Case C確定: 0-20d reversal -0.687% NW-t=-4.44・古典PEAD 40-120d不在・61-80d再出現-0.688%）

---

## 0. 文献から抽出した定量事実（3案の根拠）

| 論文 | サプライズ定義 | ランク法 | 保有 | 効果量 |
|---|---|---|---|---|
| FOS1984 / Bernard&Thomas1989 | SUE=季節ランダムウォーク(AR1補正)予測誤差/σ | **前四半期SUE分布**基準の十分位（look-ahead回避） | 発表後60営業日 | D10-D1年率≈25%（コスト前） |
| CJL1996 | ①SUE=(EPS_q−EPS_{q-4})/σ(直近8四半期) ②ABR=発表日-2〜+1の対等加重市場超過リターン ③REV6=アナリスト予想改定 | NYSE基準十分位 | 6ヶ月 | 6M spread 8.3%・SUE/ABR相関低（独立情報） |
| Brandt et al.2008 | EAR=発表日±1の3日窓・サイズ×B/Mマッチトポートフォリオ超過リターン | 五分位（前四半期分布） | 発表2日後から1四半期 | EAR年率7.55%・SUE比+1.3%・**3四半期後の反転なし**・EAR+SUE併用12.5% |
| Kats&McCubbins2018 | SUE=(実績−期待)/株価・IBES実績 | 前四半期分布の十分位 | **発表2日後**から13週 | 集計レベルではドリフト存在・分解すると非単調（anomalous anomaly） |

日本市場への含意:
- コンセンサス不要のサプライズ定義は文献内に2系統存在: **SUE季節ランダムウォーク型**（CJL式・四半期EPSのみで計算可）と**EAR/ABR型**（価格のみで計算可）。両者は相関が低く独立情報（CJL: SUE-ABR相関0.115-0.44帯）。
- ランク閾値は絶対値でなく**前四半期分布の分位点**を使うのが標準（look-ahead回避の正典手続き）。
- 保有期間の文献標準は60営業日（FOS/B&T）または1四半期（Brandt/Kats）。40営業日は文献標準ではないがStudy82 PhaseD台帳と直接比較可能。

---

## 1. 共通仕様（3案すべてに適用・変更禁止）

### 1.1 データソース（J-Quants Standard・疎通確認済み）
| 項目 | エンドポイント | フィールド（実名・Study82 Phase0実測） |
|---|---|---|
| 決算開示 | `/v2/fins/summary` | `Code, DiscDate, DiscTime, DiscNo, DocType, CurPerType, EPS, OP, Sales, NP, FOP, FEPS, FSales, FNP, Eq, ShOutFY, AvgSh` |
| 株価 | `/v2/equities/daily_quotes` | 調整後OHLCV |
| 市場区分 | `/v2/listed/info` | `MarketCode`（0111=プライム, 0112=スタンダード, 0113=グロース） |
| 指数 | TOPIX日次（daily_quotes or indices） | ベンチマーク調整用 |

### 1.2 イベント台帳規則
- イベント日 T0 = `DiscDate`。`DiscTime`によらず**エントリーは常にT0翌営業日以降**（寄前開示でも当日エントリー禁止=lookahead安全側）。
- 訂正開示（DocTypeに訂正を示す種別）は**新規イベントとして不採用**。`DiscNo`で重複排除。
- 同一銘柄・保有期間内の後続イベント: **初回イベントのみ採用**（ピラミッディング禁止・研究台帳では独立イベントとして別途記録可）。
- PIT: ユニバース所属判定は**Study75 PITユニバースのイベント月時点所属**（hindsight排除・PhaseD同一手続き）。

### 1.3 執行・コスト（PARAMS_LOCKED準拠）
```
entry  = T_entry営業日の寄付成行（バックテスト=調整後始値）
exit   = 保有満了営業日の寄付成行
slippage   = 0.001  (0.1% 片道・往復適用)
commission = 0.00055 (0.055% 片道・往復適用)
sizing = イベント等ウェイト（研究）/ live時 max_positions=3・max_single_weight=0.25（CIRCUIT）
損切り = v1.0仕様ではなし（タイムイグジットのみ。SL付与は別Studyとしてのみ許可——ベンチマーク改変禁止）
```

### 1.4 検証プロトコル（共通ゲート）
```
fresh_run必須（キャッシュ判定禁止・Study52規則）
効果量表記 = Spread / 95%CI / Newey-West t / n（Study82F様式）
IS = 2016-04〜2022-12 / OOS = 2023-01〜2026-06、oos_is_ratio ≥ 0.7
Right-censoring: 有効標本比率 ≥ 70%（Study82F 70%ルール）
n_min = 1,000イベント（全体）・サブ群 ≥ 200
sanity: sharpe ≤ 3.0 / パラメータスイープ禁止（本書の全パラメータは事前固定）
Multiple Testing: Primary判定は各案1個のみ。Secondary以下は診断扱い（Study82F三層分離）
```

---

## 2. 【1案】PEAD-J Classic v1.0 — 古典ベンチマーク（Bernard-Thomas/FOS直系）

**仮説**: 「日本株に古典的SUE型PEADが存在する」（H0: D10-D1スプレッド ≤ 0）
**事前予測（登録）**: Study82F Case C確定に基づき**FAIL（spread ≤ 0）を予測**。本案の価値は勝つことではなく、改善研究の**較正原点**となること。

| 項目 | 仕様（固定値） |
|---|---|
| Universe | Study75 PITユニバース全銘柄。追加フィルタ: 株価 ≥ 100円・ADV20 ≥ 5,000万円（執行可能性最低限のみ。文献同様、市場区分・時価総額フィルタなし） |
| Event | 四半期決算短信（`DocType`が`FinancialStatements`を含む・訂正除外）。`CurPerType ∈ {1Q,2Q,3Q,FY}` |
| Surprise | **SUE = (EPS_q − EPS_{q−4}) / σ(EPS_q − EPS_{q−4})**、σは直近8四半期の季節差分標準偏差（CJL1996式・コンセンサス不要）。**8四半期未満の銘柄は除外**。EPSは`/fins/summary`の`EPS`（連結優先・単体のみはNCEPS） |
| Ranking | **前四半期のSUE分布**で十分位カットオフを確定（当四半期分布使用=look-ahead・禁止）。D10=最上位 |
| Signal | 研究: D10ロング−D1ショートのスプレッド。live検証: **D10ロングオンリー** |
| Entry | T0+1営業日 寄付 |
| Exit | **エントリーから60営業日後 寄付**（FOS/B&T標準） |
| Benchmark調整 | CAR = 個別リターン − TOPIX同期間リターン（B&Tのサイズ十分位マッチングは日本で再現困難のためTOPIX単純化・v1.0固定） |
| Primary判定 | D10−D1スプレッド（60d・コスト後）> 0 かつ NW-t ≥ 2.0 → PASS |

**再現手順（API）**:
1. `/fins/summary`を全銘柄収集（PhaseD方式: 1.2秒/リクエストスロットリング・失敗と空の区別必須）
2. 銘柄×四半期でEPS系列構築 → SUE計算（≥8四半期）
3. 各暦四半期末に前四半期SUE分布から十分位境界を保存 → 当四半期イベントに適用
4. T0+1寄付〜T0+61寄付のコスト後リターン → D10/D1群平均・スプレッド・NW-t

---

## 3. 【2案】PEAD-J Practical v1.0 — 日本市場実戦型（ガイダンス修正イベント）

**仮説**: 「会社予想の大幅上方修正+ファンダ確認+市場初動確認の複合イベントに正のドリフトが存在する」
**根拠**: 日本固有の情報源=会社予想（`EarnForecastRevision`開示はStudy82 Phase0で実在確認済み・54件中13件）。EAR型確認フィルタはBrandt2008（EARはSUEと独立・反転しない）+CJL1996（ABR）に依拠。ユーザー提示のLevel1定義を正典化。

| 項目 | 仕様（固定値） |
|---|---|
| Universe | 東証プライム+スタンダード（`MarketCode ∈ {0111, 0112}`）・時価総額 ≥ 50億円（`ShOutFY`×前日終値）・ADV20 ≥ 1億円・株価 ≥ 100円 |
| Event | 次のいずれか（OR）:<br>**(a)** `DocType = EarnForecastRevision` かつ ΔFOP ≥ +10%<br>**(b)** 決算短信同時修正: 短信の`FOP`が直前開示`FOP`比 +10%以上<br>ΔFOP = (FOP_new − FOP_prev) / \|FOP_prev\|、**FOP_prev > 0 必須**（黒字予想からの上方修正のみ） |
| Filter（全て必須） | ① Sales YoY > 0（直近実績四半期累計）<br>② OP YoY > 0<br>③ NP > 0（赤字企業除外）<br>④ FEPS > 0<br>⑤ Eq > 0（債務超過除外） |
| 初動確認（EAR型） | **T0+1寄付ギャップ**: `Open(T0+1)/Close(T0) − 1` − TOPIX同ギャップ **> 0** の場合のみエントリー。<br>live執行: 寄前気配（kabuステーションAPI板情報・08:55時点）でギャップ正を確認→寄付成行。BT proxy=実際の始値ギャップ |
| Entry | T0+1営業日 寄付（ギャップ確認と同時執行） |
| Exit | **エントリーから40営業日後 寄付**（ユーザー固定値・PhaseD台帳と比較可能） |
| Primary判定 | イベント群コスト後CAR(40d) vs TOPIX > 0 かつ NW-t ≥ 2.0 かつ n ≥ 300 → PASS |

**留保（事前登録）**: (a)+(b)の年間イベント数は未実測（Phase0では54件中13件が予想修正——全銘柄×10年では数千件規模の見込みだが要実測）。n < 300ならINCONCLUSIVE（FAILではない）。

**再現手順（API）**:
1. `/fins/summary`全収集 → `DocType=EarnForecastRevision`抽出 + 短信のFOP時系列差分で(b)検出
2. Filter①-⑤をイベント時点の最新実績短信から適用（PIT: 開示済みデータのみ）
3. T0+1ギャップ計算 → 正のみ採用 → 40営業日CAR測定

---

## 4. 【3案】EDMR v1.0 — Event-Driven Mean Reversion（自プロジェクト実測最適化型）

**仮説**: 「日本Small/Mid-cap株の決算後ネガティブサプライズは過剰反応であり、0-20営業日で訂正（リバウンド）する」
**根拠（全て自プロジェクト実測・事前登録済み効果量）**:
- Study82 PhaseD: Negative群40dコスト後 +1.157%（t=7.9）> Positive群 +0.619%（t=5.3）
- Study82E: 逆転はSmall/Mid集中（Size寄与1.06pp=最大）・Large-capスプレッド≈0（t=0.34）・regime不変
- Study82F: 0-20d spread −0.687%（95%CI[−0.991,−0.384]・NW-t=−4.44）→ **保有20営業日が効果最大窓**
- 方向は古典PEADの**逆**（ネガティブ側をロング）。Study82F Case C「Event-driven Mean Reversion独立研究」の起案実体=本案。

| 項目 | 仕様（固定値） |
|---|---|
| Universe | Study75 PITユニバース ∩ **時価総額 < 1,000億円**（Large-cap除外=Study82E根拠）∩ 時価総額 ≥ 50億円 ∩ ADV20 ≥ 1億円 ∩ 株価 ≥ 100円 |
| Event | 四半期決算短信（1案と同一規則）で **YoY EPS変化が負**（EPS_q − EPS_{q−4} < 0）。<br>※binary符号分類はPhaseD/82E/82F台帳30,952件と直接比較可能にするため意図的に維持（82Eの解像度批判への対応=連続量化はv1.1のStudy候補であり本仕様では禁止） |
| Filter | ① Eq > 0（債務超過除外）② NP > −0.5×Eq相当の大赤字除外は**しない**（過剰反応仮説はバッドニュース側が対象。ただし監理・整理銘柄除外）③ FEPS欠損可 |
| Entry | T0+1営業日 寄付 |
| Exit | **エントリーから20営業日後 寄付**（Study82F 0-20d窓・効果最大） |
| 方向 | **ロングオンリー**（ネガティブサプライズ群を買う） |
| Primary判定 | Negative群コスト後CAR(20d) vs TOPIX > 0 かつ NW-t ≥ 2.0 → PASS |
| Secondary（診断のみ） | ① Positive群との0-20dスプレッド再現（−0.687%±CI内）② Small/Mid限定でスプレッド拡大するか（82E予測: する）③ 61-80d再出現窓の構造（82F未解明点） |

**リスク（事前明示）**:
- 逆張り固有のテール: 悪材料継続銘柄の下方continuation。v1.0はSLなし（タイムイグジットのみ）のためイベント単位MaxDD分布を必須記録。
- 82Fの効果量は**スプレッド**であり、ロング片脚のTOPIX超過が正である保証はない（PhaseD 40dではNeg群絶対+1.157%だが、単一スロットCAGR≈4%<TOPIX 12.76%の前例あり）。Primary判定が「vs TOPIX」なのはこのため。
- 本案は research_state.md 記載の「Event-driven Mean Reversion独立研究（ASK_FIRST・未着手）」の起案書に相当。**実装・BT実行はユーザー承認後**。

---

## 5. 3案の関係と検証順序

```
1案 Classic v1.0   = 較正原点（予測: FAIL）。改善研究のΔ測定基準。文献直系・変更禁止。
2案 Practical v1.0 = 日本固有情報源(会社予想)の独立検証。1案とサプライズ定義が直交
                     （SUE型 vs ガイダンス修正+EAR型・CJL/Brandtの低相関知見）。
3案 EDMR v1.0      = 実測が支持する唯一の方向（逆張り20d）。Study82F後継の本命。

検証順序（推奨）: 3案 → 2案 → 1案
  理由: 3案は既存台帳30,952件の再利用で新規データ取得ほぼゼロ（Freeze Rule観点で最軽量）。
        1案・2案はSUE 8四半期系列/予想修正台帳の新規構築が必要。
改変規則: v1.0パラメータの変更は一切禁止。変更したものは v1.1+ の別Studyとして起案
        （Entry変更のみ/Exit変更のみ/Filter追加のみ、の単一変数比較を正典手続きとする）。
```

## 6. 未確定事項（実装前にユーザー決裁が必要な点）
1. 3案の実装承認（ASK_FIRST・新規スクリプト作成に該当）
2. 2案イベント(b)「短信同時修正」の検出には全銘柄FOP時系列の新規構築が必要——工数増を許容するか、(a)予想修正開示のみで開始するか
3. 1案の60営業日保有はPhaseD台帳(40d)と直接比較不能——60d正典を維持するか40dに寄せるか（本書は文献忠実=60dで固定を推奨）
