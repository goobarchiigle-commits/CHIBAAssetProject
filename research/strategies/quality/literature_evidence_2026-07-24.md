# Quality Multi-Factor — 文献根拠原本（2026-07-24）

**性格**: 起案原本（3仕様書の共通根拠・定量事実抽出）。ソース=`book.pdf/03_Quality_MultiFactor/`（書籍10冊・pypdf抽出）+ `book.pdf/02_PEAD_Event/`所収の学術論文（Novy-Marx2013/Piotroski2000/Sloan1996/HXZ2015・前回抽出済み）。

## 文献から抽出した定量事実

| 文献 | ファクター定義（原文準拠） | 実装規約 | 効果量 |
|---|---|---|---|
| Novy-Marx 2013 | **GP/A = (REVT − COGS) / AT**（売上総利益/総資産） | NYSE breakpoints十分位・VW・金融(SIC 6)除外・会計データはFYE翌暦年6月末から使用・年次 | B/Mと同等の説明力・高収益企業が有意に高リターン |
| Piotroski 2000 | **F-Score 9項目**: ①ROA>0 ②CFO>0 ③ΔROA>0 ④CFO>ROA(accrual) ⑤長期負債比率低下 ⑥流動比率上昇 ⑦新株発行なし ⑧売上総利益率上昇 ⑨総資産回転率上昇 | **高B/M分位内**で適用・FYE後5ヶ月目からリターン測定・F≥8をHigh | 高B/M内で年率+7.5%のシフト |
| Sloan 1996 | **Accruals = (ΔCA−ΔCash) − (ΔCL−ΔSTD−ΔTP) − Dep / 平均TA**・低accrualsロング | 十分位・リターン累積はFYE後4ヶ月目開始・サイズ調整リターン | accrual成分の持続性をミスプライス（市場効率性棄却 LR=180.9） |
| Berkin&Swedroe（QMJ引用） | 高収益・安定・成長・高還元 = Quality minus Junk | — | QMJプレミアム年率3.8%（1927-2015）・valueより持続的 |
| Berkin&Swedroe（Ball et al.引用） | **Cash-based operating profitability**（accrual調整を除いた営業CF基準収益性）が全収益性指標中最強 | — | 年率4.8%（t=6.3）> operating 3.5%（t=4.0） |
| Greenblatt（Magic Formula） | **ROC = EBIT/(正味運転資本+正味固定資産)** + **EY = EBIT/EV** の2ランク合算 | 時価総額$50M以上・20-30銘柄・年次リバランス | 169/169の3年期間全てで市場超過（大型1000銘柄でも成立） |
| O'Shaughnessy | 複数ファクターの**パーセンタイルランク平均=コンポジット**方式 | 単一ファクターより複合が頑健 | （コンポジット手法の方法論的根拠として採用） |
| HXZ 2015 | ROE四半期ファクター（q-factor） | 四半期更新 | 収益性ファクターの独立性 |

## J-Quants `/v2/fins/summary` フィールド可用性（実測・Study82 Phase0）

| 文献要求 | フィールド | 可否 |
|---|---|---|
| 売上総利益（GP） | なし（COGS非提供） | **✗ → Classic v1.0は補助データ要（設計上の留保）・Practical系はOP代替** |
| 営業利益/総資産 | `OP`/`TA` | ✓（RMW=operating profitability・Fama-French2015系の文献支持あり） |
| CFO | `CFO` | ✓ |
| NP・ROA | `NP`/`TA` | ✓ |
| 自己資本比率 | `EqAR`（または`Eq`/`TA`） | ✓（長期負債比率の代替） |
| 流動比率 | なし（流動資産/負債の内訳非提供） | ✗ |
| 発行済株式数 | `ShOutFY`・`AvgSh` | ✓ |
| 売上総利益率 | ✗ → 営業利益率`OP`/`Sales`で代替 | △ |
| 回転率 | `Sales`/`TA` | ✓ |
| BPS（B/M用） | `BPS`または`Eq`/`ShOutFY` | ✓ |

**正典移行**: 本原本の3案は versioned Strategy Specification としてFrozen発行済み（`quality_mf_classic_v1.0.md` / `quality_mf_practical_v1.0.md` / `quality_value_smallmid_v1.0.md`）。以後の参照は仕様書側が正。

**改名注記（2026-07-24同日）**: 3案の仕様書名は初版発行時"Quality MF SmallMid"だったが、内容がPiotroski原典のB/M条件（Value）を核心とするためユーザー指摘により"Quality Value SmallMid"へ改名。RGP・内容の変更なし（命名のみ）。
