---
document: RGP Taxonomy — Literature-driven Return Generation Process Classification
version: "1.0"
status: Frozen（2026-07-24ユーザー決裁。例外: RGP-7とSmall Growth分類のみPROVISIONAL=暫定・v1.1で再評価）
created: 2026-07-24
frozen: 2026-07-24
sources: book.pdf/01-08全フォルダ+ロジック原版（計60ファイル・重複除き約45タイトル）・pypdfテキスト抽出（scratchpad保存）
position: 研究体系の最上位文書（strategy_specification_governance.md・roadmapの上位。全RGP/Family/Specification/Studyはここから派生）
---

# RGP Taxonomy v1.0 — 超過収益の経済的メカニズム分類

## §0 目的と遵守事項

**目的**: 戦略（アプリケーション）ではなく、**超過収益が発生する経済的メカニズム（Return Generation Process）**を唯一の分類軸として、保有文献全体を再整理する。本書は今後発行される全Strategy Specificationの親文書（研究のOS）。

**本書が守った禁止事項（ユーザー指定）**: 新規売買ルール考案なし／AI独自仮説なし／文献にない改善案なし／Strategy Specification記述なし／バックテスト条件記述なし／実装コードなし／複数RGP混合なし／特徴量クラスタリング分類なし。各RGPの記載は文献の主張と自プロジェクト実測の要約のみ。

**分類原則**:
1. 著者分類ではなくメカニズム分類（同一書籍が複数RGPの根拠になりうる。例: O'Shaughnessy→Value/Momentum/Quality）
2. 「典型Entry/Exit」等は文献に現れる典型値の記録であり仕様ではない（仕様化はLayer3=Strategy Specificationの仕事）
3. メカニズムが文献内で争いのある場合（例: Value=行動 vs リスク）は争いをそのまま記録する
4. **内部実測は定型2行（Current Internal Evidence / Internal Verdict）でのみ記録**（Taxonomyの研究日誌化禁止。詳細・数値はresearch_state.md/各Study正典が正）
5. **各RGPにEvidence Source欄**（Literature / Internal / Literature+Internal）——何が文献由来で何が自プロジェクト実測かを一目で判別可能にする
6. 分類確度が他RGPと同水準に達しないものは**PROVISIONAL（暫定）**と明示し、Frozenの対象外とする（将来の再分類の権利を留保）

---

## §1 Taxonomy Tree v1.0

```
Return Generation Process
│
├── RGP-0  Market Risk Premium（市場リスクプレミアム・β）
│
├── RGP-1  Underreaction（情報への過少反応・ドリフト）
│      ├── PEAD（決算サプライズドリフト）
│      ├── Guidance Revision Drift（会社予想修正ドリフト・日本固有情報源）
│      └── Analyst Revision Drift（アナリスト予想改定ドリフト）
│
├── RGP-2  Overreaction Correction（過剰反応の訂正・平均回帰）
│      ├── Event-driven Mean Reversion（イベント起点・短期）
│      └── Long-term Reversal（長期反転）
│
├── RGP-3  Cross-sectional Momentum（相対強度・銘柄間比較）
│      ├── 12-2型（Jegadeesh-Titman系）
│      └── ボラ調整トレンド強度型（Clenow回帰勾配系）
│
├── RGP-4  Time-series Momentum / Trend Following（絶対モメンタム・自己比較）
│      ├── Absolute Momentum（Antonacci系・ルックバック超過リターン符号）
│      └── Channel Breakout / MA系（Turtle・Kaufman系）
│
├── RGP-5  Quality Premium / Junk Avoidance（品質プレミアム）
│      ├── Profitability（収益性: GP/A・RMW・ROC）
│      ├── Earnings Quality / Accruals（会計品質）
│      └── Financial Strength（財務健全性: F-Score）
│
├── RGP-6  Value Premium（割安性プレミアム）
│      ├── 簿価型（B/M・Fama-French系）
│      ├── 収益型（EV/EBIT: Greenblatt EY・Carlisle Acquirer's Multiple）
│      └── コンポジット型（O'Shaughnessy Value Composite）
│
├── RGP-7  Long-horizon Growth Compounding（長期成長複利の過小評価）★PROVISIONAL
│      └── 100 Baggers / Lynch Fast Grower / Fisher型（同一機構か未確定・§2参照）
│
├── RGP-8  Defensive / Low-Risk Premium（低リスク・防御プレミアム）
│
└── RGP-9  Microstructure / Order Flow / Liquidity Provision（マイクロ構造・日中）
       ├── Order Flow / Auction Theory（CVD・Footprint・Market Profile）
       └── VWAP / Execution Anomaly
```

**独立RGPと認めなかった候補**（§4境界整理で詳述）:
- **Turnaround** → RGP-6×RGP-5の交差族（Piotroski=割安×財務改善の検出。独立メカニズムの文献根拠なし）
- **Small Growth** → **Current Classification: Non-independent（★PROVISIONAL）**。O'Shaughnessy実測では小型グロース（高PSR等）は歴代最悪級のパフォーマンス群であり、現時点の文献精査では「小型×Quality/Value」（RGP-5/6の適用領域）またはRGP-7（成長持続の過小評価）に還元される。ただし全文献の精査は未了——将来の独立RGP化の可能性は閉ざさない（v1.1以降で再評価）
- **Reflexivity（Soros）** → RGP-3/4の正のフィードバック機構の理論的説明として吸収（独立の定量実装文献なし）

---

## §2 各RGP定義

### RGP-0 Market Risk Premium

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 株式保有リスクの対価。非効率ではなく正当な報酬 |
| 非効率性仮説 | なし（効率市場でも存在） |
| 保有期間 | 無期限（B&H） |
| 典型Universe/Entry/Exit | 市場全体（指数）/ 常時保有 / なし |
| 必要データ | 指数のみ |
| 代表文献 | Siegel『Stocks for the Long Run』・Ellis『Winning the Loser's Game』 |
| 日本再現性 / J-Quants単独 | ◎ / ◎（TOPIX） |
| Evidence Source | Literature + Internal |
| Current Internal Evidence | Study101・Study103 |
| Internal Verdict | ADOPTED（判定基準として採用——TOPIX B&Hが全RGPの超過対象。詳細はresearch_state.md） |

### RGP-1 Underreaction

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 企業固有情報（決算・予想修正）への価格反応が不完全で、後日残余が織り込まれる |
| 非効率性仮説 | 限定注意・アンカリング（Bernard&Thomas: 期待更新の不完全性）・情報の緩慢な拡散（Antonacci引用のHong-Stein系）・BSV1998「momentumは良いニュースへの過少反応」（Gray&Vogel経由） |
| 保有期間 | 文献典型: 60営業日（FOS/B&T）〜1四半期〜6ヶ月（CJL） |
| 典型Universe | 全上場（効果は小型・低カバレッジで最大=Bernard&Thomas/Brandt） |
| 典型Entry/Exit | イベント翌〜2営業日後 / 60営業日 or 次回決算まで |
| 必要データ | 決算開示（日時・実績・予想）・株価 |
| 代表文献 | Ball&Brown1968・Bernard&Thomas1989・FOS1984・CJL1996・Brandt2008(EAR)・Kats&McCubbins2018（懐疑側: 分解すると非単調） |
| 日本再現性 / J-Quants単独 | ○（コンセンサス不要のSUE/EAR型が文献内に存在）/ ◎（/fins/summary） |
| Evidence Source | Literature + Internal |
| Current Internal Evidence | Study82 Phase0 / Phase D / 82F |
| Internal Verdict | 古典形（Earnings Event族）=REJECT（PhD FAIL・82F古典形不在）／Guidance Revision族=UNTESTED |

### RGP-2 Overreaction Correction

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 悪材料への過剰反応が生む価格の行き過ぎが、後日訂正される |
| 非効率性仮説 | BSV1998「valueは悪いニュースへの過剰反応」（Gray&Vogel経由）・Carlisle: mean reversionは市場の基本動力・損失回避による投げ売り |
| 保有期間 | 文献典型: 短期(数日-1ヶ月)＝short-term reversal / 長期(3-5年)＝DeBondt-Thaler型（Berkin&Swedroe/Ilmanen章参照） |
| 典型Universe | 全上場（効果は小型・低流動性で最大とされる） |
| 典型Entry/Exit | 急落・ネガティブイベント後 / 数日〜数週間 |
| 必要データ | 株価・イベント台帳 |
| 代表文献 | Berkin&Swedroe(reversal章)・Ilmanen・Carlisle（メカニズム論） |
| 日本再現性 / J-Quants単独 | ◎ / ◎ |
| Evidence Source | Internal（文献は補助——直接根拠は自プロジェクト実測） |
| Current Internal Evidence | Study82E / Study82F（Case C） |
| Internal Verdict | SUPPORTED（現象の実在を支持。戦略としてはEDMR v1.0=UNTESTED） |

### RGP-3 Cross-sectional Momentum

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 相対的に強い銘柄が中期的に強さを維持する（銘柄間の横比較） |
| 非効率性仮説 | 初期過少反応→追随による遅延過剰反応（Antonacci引用のHong-Stein/DHS）・CJL1996「momentumの一部はearnings momentumで説明」・limits to arbitrage（Gray&Vogel） |
| 保有期間 | 文献典型: 形成6-12ヶ月（2-12型=直近1ヶ月除外）・保有1-3ヶ月・月次リバランス（Jegadeesh-Titman、Antonacci/Gray&Vogel記載） |
| 典型Universe | 大型含む全上場・上位10-30%を買い |
| 典型Entry/Exit | 月次ランク上位入り / ランク脱落 |
| 必要データ | 株価のみ |
| 代表文献 | Gray&Vogel『Quantitative Momentum』・Clenow『Stocks on the Move』（年率化指数回帰勾配90日×100日MA×ギャップ15%除外）・Faber RS・O'Shaughnessy・CJL1996 |
| 日本再現性 / J-Quants単独 | △（Internal Verdict参照）/ ◎ |
| Evidence Source | Literature + Internal |
| Current Internal Evidence | Study95 / Study99 / Study95E |
| Internal Verdict | REJECT（Study95=FAIL_ZERO_SPREAD Kill判定・Study99追認・95E=逆転構造もPEADと不一致）。Study76(Clenowベンチマーク)は正典計画のみ存置 |

### RGP-4 Time-series Momentum / Trend Following

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 資産自身の過去リターン符号が将来リターンを予測（自己比較・絶対モメンタム） |
| 非効率性仮説 | Antonacci: absolute momentum=対T-bill相対モメンタムの単一資産版・トレンドの持続（行動的追随+機関の緩慢な資金移動） |
| 保有期間 | 文献典型: ルックバック12ヶ月・月次判定（Antonacci）/ チャネルブレイク20-55日（Turtle=Faith）/ 各種MA（Kaufman・Carver） |
| 典型Universe | 指数・先物・ETF（分散が本質=Clenow『Following the Trend』）・個別株は補助 |
| 典型Entry/Exit | 正シグナルで保有・負転換で現金/債券退避 |
| 必要データ | 価格のみ |
| 代表文献 | Antonacci『Dual Momentum』・Clenow『Following the Trend』・Faith『Way of the Turtle』・Kaufman・Carver『Systematic Trading』・Fitchen |
| 日本再現性 / J-Quants単独 | △（クロスアセット分散は現行口座制約で不可）/ ○（指数・ETF） |
| Evidence Source | Literature + Internal |
| Current Internal Evidence | Study83 |
| Internal Verdict | REJECT（TOPIX単一指数版・0/3アームゲート不通過）。文献が本質とする多資産分散版=UNTESTED（口座制約により当面検証不能） |

### RGP-5 Quality Premium / Junk Avoidance

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 高収益・高会計品質・財務健全企業の将来収益持続が過小評価される／ジャンク回避 |
| 非効率性仮説 | Novy-Marx: 高GP/Aは「valueの裏面」・Sloan: accrual成分の低持続性を市場が見誤る・Piotroski: 財務シグナルの遅延織り込み・QMJ（Berkin&Swedroe: 年率3.8%・valueより持続的） |
| 保有期間 | 文献典型: 12ヶ月・年次リバランス（低回転が本質） |
| 典型Universe | 全上場−金融 |
| 典型Entry/Exit | 年次формация（FYE+4〜6ヶ月ラグ: Sloan4/Piotroski5/Novy-Marx6）/ 次回リバランス |
| 必要データ | 財務諸表（PL/BS/CF）・株価 |
| 代表文献 | Novy-Marx2013・Piotroski2000・Sloan1996・Greenblatt（ROC）・Cunningham・Fisher・Lynch（定性側）・Berkin&Swedroe・Ilmanen |
| 日本再現性 / J-Quants単独 | ○ / △（GP・長期負債・流動比率が/fins/summary非提供→literature_evidence_2026-07-24.md可用性表） |
| Evidence Source | Literature |
| Current Internal Evidence | なし |
| Internal Verdict | UNTESTED（仕様書3本発行済み・実装未承認） |

### RGP-6 Value Premium

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 割安株の超過リターン。**メカニズムは文献内で争いあり**: ①行動説=悪材料への過剰反応の訂正（BSV・Carlisle mean reversion）②リスク説=distress riskの対価（Fama-French） |
| 非効率性仮説 | ①なら過剰反応（RGP-2と機構共有）・②なら非効率ではない |
| 保有期間 | 文献典型: 12ヶ月〜数年・年次リバランス |
| 典型Universe | 全上場（Carlisle: EV算出可能な非金融） |
| 典型Entry/Exit | 割安ランク上位（B/M高位・EV/EBIT低位=Acquirer's Multiple）/ 年次入替 |
| 必要データ | 財務（簿価・営業利益・純有利子負債）・時価総額 |
| 代表文献 | Carlisle『Acquirer's Multiple』（EV/営業利益・deep value）・Greenblatt（EYレッグ）・O'Shaughnessy（Value Composite）・Piotroski（高B/M前提）・Berkin&Swedroe・Ilmanen |
| 日本再現性 / J-Quants単独 | ○ / △（純有利子負債の内訳が/fins/summary限定的→EV算出に留保。B/M型は◎） |
| Evidence Source | Literature |
| Current Internal Evidence | なし（注: 旧RSR42/フジコ法はvalueではない——Study101文脈） |
| Internal Verdict | UNTESTED |

### RGP-7 Long-horizon Growth Compounding ★PROVISIONAL（暫定・Frozen対象外）

**分類確度に関する注記（v1.0時点）**: Mayer/Lynch/Fisherが**同一の経済メカニズムを記述しているかは未確定**。Study110の内部知見「Winner=複数種類（3M/6M/12M勝者のJaccard重複28.2%・3Mvs12Mは19.66%）」とも整合する曖昧さが残る——「成長複利」は単一RGPではなく複数機構の混合である可能性がある。**本RGPの定義・存在自体をv1.1以降で再評価する**（それまで本RGPを親とするFamily/Specificationの起案は保留推奨）。

| 項目 | 内容 |
|---|---|
| 経済メカニズム | （暫定）高ROE×再投資×長期成長持続の複利効果を市場が過小評価（Mayer: 100倍株=成長×再投資の長期保有・Lynch: Fast Grower・Fisher: 卓越企業の長期保有） |
| 非効率性仮説 | （暫定）成長持続期間の系統的過小評価・早すぎる利確 |
| 保有期間 | 数年〜十数年（本質的に超長期） |
| 典型Universe | 小型〜中型の高成長企業 |
| 典型Entry/Exit | 定性+定量スクリーン / 原則売らない（thesis破綻時のみ） |
| 必要データ | 長期財務履歴・定性情報（定量化困難成分が大きい） |
| 代表文献 | Mayer『100 Baggers』・Lynch『One Up on Wall Street』・Fisher『Common Stocks and Uncommon Profits』・Tillinghast |
| 日本再現性 / J-Quants単独 | △（定性成分・超長期検証の標本不足）/ △ |
| Evidence Source | Literature（定性中心）+ Internal（負の間接証拠のみ） |
| Current Internal Evidence | Study110A / Study110B |
| Internal Verdict | NEGATIVE-INDIRECT（110B: 勝者予測可能性TERMINAL・全horizon lift<1.5。本RGPの直接検証は未実施だが、隣接命題が機械的停止済み） |

### RGP-8 Defensive / Low-Risk Premium

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 低ボラ・低β銘柄のリスク調整後（時に絶対）超過リターン |
| 非効率性仮説 | Berkin&Swedroe: CAPM予測に反しフラット〜負のリスク-リターン関係・宝くじ需要・レバレッジ制約（Ilmanen） |
| 保有期間 | 12ヶ月前後・低回転 |
| 典型Universe | 全上場・低ボラ/低β分位 |
| 典型Entry/Exit | ボラ/βランク下位分位 / 年次〜四半期入替 |
| 必要データ | 価格のみ（β・ボラ計算） |
| 代表文献 | Berkin&Swedroe（defensive章）・Ilmanen両著 |
| 日本再現性 / J-Quants単独 | ◎ / ◎ |
| Evidence Source | Literature |
| Current Internal Evidence | なし |
| Internal Verdict | UNTESTED |

### RGP-9 Microstructure / Order Flow / Liquidity Provision

| 項目 | 内容 |
|---|---|
| 経済メカニズム | 日中の需給不均衡・オークション構造・流動性供給の対価（CVD/Footprint/Market Profile/VWAP乖離） |
| 非効率性仮説 | 注文フローの情報性・執行需要の一時的価格圧力 |
| 保有期間 | 分〜日（デイトレード帯） |
| 典型Universe | 高流動性銘柄・先物 |
| 必要データ | ティック・板・出来高内訳（EODでは不可） |
| 代表文献 | ロジック原版10本（CVD/Footprint/Market Profile/Order Flow/VWAP）・Johnson『Algorithmic Trading and DMA』・『Inside the Black Box』・Al Brooks（price action） |
| 日本再現性 / J-Quants単独 | △ / ✗（EOD不可。ただしBulk APIインフラ（minute/tick・commit a79a0b4）が整備済み=データ面の前提は部分的に存在） |
| Evidence Source | Literature（実務文書中心） |
| Current Internal Evidence | なし |
| Internal Verdict | UNTESTED（現行研究トラック未接続。執行改善目的ならRGPでなく執行層の知見として利用可） |

---

## §3 文献マッピング（全ライブラリ→RGP）

| フォルダ/文献 | 主RGP | 副RGP |
|---|---|---|
| Ball&Brown1968 / Bernard&Thomas / FOS(引用) / Brandt2008 / Kats&McCubbins / Brown&Pope | RGP-1 | — |
| CJL1996 | RGP-1 | RGP-3（両者の関係を直接研究） |
| Sloan1996 | RGP-5(Accruals) | RGP-1（決算情報の遅延織込み） |
| Novy-Marx2013 | RGP-5(Profitability) | RGP-6（valueの補完と明言） |
| Piotroski2000 | RGP-5(Fin.Strength) | RGP-6（高B/M前提）→Turnaround族の根拠 |
| HXZ2015 | RGP-5(ROE) | 多アノマリー統合（方法論） |
| Antonacci『Dual Momentum』 | RGP-4 | RGP-3（relative半分） |
| Gray&Vogel『Quantitative Momentum』 | RGP-3 | RGP-1（BSV機構論） |
| Clenow『Stocks on the Move』 | RGP-3 | RGP-4（トレンドフィルタ） |
| Clenow『Following the Trend』/ Faith『Turtle』/ Kaufman / Fitchen / Tang | RGP-4 | — |
| Carver『Systematic Trading』 | RGP-4 | システム設計層 |
| Faber RS | RGP-3 | RGP-4 |
| Greenblatt『Magic Formula』 | RGP-5×RGP-6（2ファクター合成と明示） | — |
| Carlisle『Acquirer's Multiple』 | RGP-6 | RGP-2（mean reversion機構論） |
| O'Shaughnessy『What Works』 | RGP-6/RGP-3/RGP-5 | 方法論（コンポジット・小型グロース否定の実測） |
| Piotroski+Carlisle（組合せ） | Turnaround族（RGP-6×RGP-5交差） | — |
| Mayer『100 Baggers』/ Lynch / Fisher / Tillinghast | RGP-7★PROVISIONAL | RGP-5（定性quality） |
| Cunningham『Quality Investing』 | RGP-5（定性） | RGP-7★PROVISIONAL |
| Berkin&Swedroe / Ilmanen両著 | 横断（RGP-2/3/4/5/6/8の実証サーベイ） | 方法論 |
| Siegel / Ellis | RGP-0 | — |
| Soros『Alchemy of Finance』 | RGP-3/4の理論背景（reflexivity） | 独立RGPとしない |
| Al Brooks / Link高勝率2冊 / たーちゃん | RGP-9・裁量（非系統） | — |
| ロジック原版10本 | RGP-9 | 執行層 |
| Johnson DMA / Inside the Black Box / Chan2冊 / Narang | RGP-9+システム設計層 | — |
| Thorp / López de Prado / Pardo / Yoder | 検証方法論層（RGPではない） | — |
| Penfold / クリアー / Siegel以外の08 | メタ・規律層（RGPではない） | — |

**非RGP層の位置づけ**: 06_SystemDesign=執行・リスク管理層（全RGPに適用）／07_ResearchMethodology=検証層（common_conventionsの源流）／05_Discretionary=機構の定性記述源（系統化はRGP経由でのみ）。

---

## §4 RGP境界整理（文献ベース）

| 問い | 整理 | 根拠 |
|---|---|---|
| PEAD vs Momentum | **PEAD ⊂ Underreaction（RGP-1）・CS Momentum（RGP-3）とは別RGP**。ただし一部重複 | CJL1996: SUE/ABRとR6の相関は低い（0.115-0.44）が、momentumの一部はearnings momentumで説明される。両者は独立情報を含む別現象 |
| CS Momentum vs Trend Following | **別RGP（比較軸が異なる: 銘柄間 vs 自己）**。ただし機構は近縁 | Antonacci: absolute momentum≒「T-billとペアにしたrelative momentum」——形式上は変換可能だが、対象（個別株横断 vs 資産クラス縦断）と文献体系が分離 |
| Absolute Momentum vs Trend Following | **ほぼ包含（同一RGP内の下位分類）** | Antonacci自身が「trend-following absolute momentum」と呼ぶ |
| Quality vs Value | **独立（負相関・相互補完）** | Novy-Marx: GP/Aは「the other side of value」・valueと組み合わせて改善と明言 |
| Value vs Overreaction | **一部重複（機構帰属が文献内で未決着）** | BSV: value=悪材料への過剰反応（→RGP-2と同機構）vs Fama-French: リスク対価。本Taxonomyは現象面で分離し争いを記録 |
| Turnaround vs Quality | **Turnaroundは独立RGPでなくRGP-6×RGP-5の交差族** | Piotroski=高B/M（割安=悪材料織込み）×財務改善シグナル。独立メカニズムの提示文献なし |
| Small Growth vs 各RGP | **Current Classification: Non-independent（★PROVISIONAL・全文献精査未了）** | O'Shaughnessy実測: 小型グロース単独は歴代最悪級。文献が支持するのは小型×Value/Quality（適用領域）か、成長持続の過小評価（RGP-7、これ自体もPROVISIONAL） |
| PEAD逆転（日本実測） vs RGP-2 | 日本Small/Midの決算後現象はRGP-1でなく**RGP-2の実例** | Study82E/82F: 過少反応でなく過剰反応訂正が観測された（EDMRの根拠） |

---

## §5 RGP独立性評価マトリクス（スリーブ分散設計用）

凡例: **独立**=機構・実証とも別 / **一部重複**=機構共有or相関あり / **包含**=一方が他方の特殊形。記載なきペアは独立扱い。

| RGP A | RGP B | 関係 | 根拠 |
|---|---|---|---|
| RGP-1 Underreaction | RGP-3 CS Momentum | **一部重複** | CJL1996（momentumの一部=earnings momentum・ただし相関低） |
| RGP-1 Underreaction | RGP-2 Overreaction | **対抗関係（同一イベントの逆符号）** | 同一イベント台帳上でどちらが優勢かは実証問題（Study82F=日本Small/MidはRGP-2優勢） |
| RGP-3 CS Momentum | RGP-4 TSMOM | **一部重複（近縁・変換可能性）** | Antonacci（absolute≒relative vs T-bill）。実測相関は高くなりがち→スリーブ分散効果は限定的と想定すべき |
| RGP-3 CS Momentum | RGP-6 Value | **独立（負相関）** | Gray&Vogel/Berkin: momentumとvalueは負相関・併用が定石 |
| RGP-5 Quality | RGP-6 Value | **独立（負相関・補完）** | Novy-Marx明言 |
| RGP-5 Quality | RGP-7 Growth Compounding★PROV | **一部重複（片方PROVISIONALのため関係自体が暫定）** | Mayer/Fisher/Lynchの銘柄像は高quality成長企業（定量side=RGP-5と共通因子） |
| RGP-5 Quality | RGP-1 Underreaction | **一部重複（情報遅延の共有）** | Sloan/Piotroski=財務情報の遅延織込み（PEADと同型の遅延機構・対象情報が異なる） |
| RGP-6 Value | RGP-2 Overreaction | **一部重複（機構帰属論争）** | BSV/Carlisle（§4参照） |
| RGP-6 Value | RGP-7 Growth Compounding★PROV | **ほぼ対極（相互排他的傾向・片方PROVISIONALのため関係自体が暫定）** | 割安性 vs 成長持続——同時充足は稀（Greenblattは中間解） |
| RGP-2 Overreaction | RGP-8 Defensive | **独立** | 機構無関係（訂正 vs リスク選好の歪み） |
| RGP-8 Defensive | RGP-5 Quality | **一部重複** | QMJのsafety成分は低リスクと重なる（Berkin/Ilmanen） |
| RGP-9 Microstructure | 全RGP | **独立（時間軸が非重複）** | 分〜日 vs 週〜年。分散源としては有効だが現行データ/執行体制で未接続 |
| RGP-0 Market | 全RGP | **包含（全RGPのベース）** | 全戦略はRGP-0超過分のみが付加価値（Study101基準） |

**スリーブ設計への含意（記録のみ・提案ではない)**: 文献上、分散効果が最も期待できる組は「RGP-5×RGP-6」「RGP-3×RGP-6」「RGP-2×RGP-8」。最も期待できない組は「RGP-3×RGP-4」「RGP-5×RGP-7★PROVISIONAL」（後者はRGP-7自体が暫定のため参考値扱い）。RGP-3/4はInternal Verdict=REJECT（§2参照）のため、現実の候補空間は RGP-2（Internal Verdict=SUPPORTED）・RGP-5/6/8（UNTESTED）・RGP-1変種（Guidance Revision族のみUNTESTED・古典形はREJECT）に限られる。

---

## §6 三層構造提案（Strategy Registryとの整合）

```
Layer 1: RGP（本書・唯一の分類軸）
    ↓
Layer 2: Strategy Family（RGP内の実装アプローチ族。例: RGP-1 → {PEAD族, Guidance Revision族, Analyst Revision族}）
    ↓
Layer 3: Strategy Specification（versioned・Frozen・REGISTRY登録単位）
```

現行REGISTRYのマッピング（変更は別途ASK_FIRST）:

| Layer 1 (RGP) | Layer 2 (Family) | Layer 3 (Spec) |
|---|---|---|
| RGP-1 Underreaction | PEAD族 | PEAD Classic v1.0 |
| RGP-1 Underreaction | Guidance Revision族 | PEAD Practical v1.0 |
| RGP-2 Overreaction Correction | Event-driven MR族 | EDMR v1.0 |
| RGP-5 Quality | Profitability+Accruals+F-Score複合族 | Quality MF Classic v1.0 / Practical v1.0 |
| RGP-5×RGP-6交差 | Value×Quality族（Piotroski型） | Quality Value SmallMid v1.0 |
| （Planned） Small Growth | → RGP-7★PROVISIONAL or RGP-5適用領域として再分類要（Current Classification: Non-independent・§4）。RGP-7自体が暫定のため再分類はRGP-7のv1.1再評価を待つのが安全 | 未起案 |
| （Planned） Turnaround | → RGP-6×RGP-5交差族として再分類要（§4） | 未起案 |

**注**: Quality Value SmallMidが交差族であることは本Taxonomyで初めて形式化された（発行時のrgp表記「quality premium/junk avoidance」より正確な位置はRGP-5×RGP-6交差）。仕様書本体は不変（Frozen）・REGISTRYの分類表記更新のみ将来検討（ASK_FIRST）。

---

## §7 保留・未分類（明示）

1. **Carry/配当系**: Ilmanenがカバーするが日本個別株現物での独立実装文献がライブラリにない → RGP候補として保留（追加文献待ち）
2. **ロジック原版のRGP-9**: minute/tick Bulk APIインフラは存在するが研究トラック未接続 → 接続判断はユーザー決裁
3. **05_Discretionaryの裁量知見**: 系統化はRGP経由でのみ許可（本Taxonomyの分類原則3）
4. **Analyst Revision族（RGP-1）**: CJL1996のREV6が文献根拠。日本でのコンセンサスデータ調達はJ-Quants Standard外 → データ調達解決までFamily定義のみ
