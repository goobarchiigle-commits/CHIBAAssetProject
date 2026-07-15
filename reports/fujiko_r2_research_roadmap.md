# FUJIKO 2.0 Ground Truth Reconstruction — Research Roadmap v2

**日付**: 2026-07-13（v2改定）
**作成**: v1=Sonnet 5（2026-07-13）/ v2=Fable 5（同日・ユーザー指示「FUJIKO 2.0 Ground Truth Reconstruction」Part1-5に基づく全面改定）
**性格**: Study75A〜75F（暫定命名含む）によりStatic RSR42前提（hindsight選定バイアス+7〜25pp規模）が崩壊した。
売買ロジック自体の standalone エッジも未証明（PITランダム42銘柄への同一ロジック適用 median IS -2.48%）。
**FUJIKO研究体系をゼロベースで再構築する**。目的は即時のCAGR最大化ではなく、
selection bias崩壊後の**正直な研究基盤の再建**である。

**拘束エビデンス**: `reports/final_research_roadmap_2026-07-04.md`（旧正典）/
`reports/complete_execution_roadmap_2026-07-04.md`（実行ログ）/ `src/research_state.md` /
`reports/study75c_e1_validity_audit.md` / `reports/study76_datr_eq_universe_c_rebaseline.md`（暫定75D）/
`reports/study76d_contamination_ablation.md`（暫定75E）/ `reports/study77_dynamic42_path_decomposition.md`（暫定75F）

## 改定履歴

| 版 | 日付 | 変更 |
|---|---|---|
| v1 | 2026-07-13 | 初版。Part A分類（領域別）・Market→Sector→Stock単一アーキテクチャ・Study87-94起案 |
| v2 | 2026-07-13 | ユーザー指示Part1-5準拠へ全面改定。Study単位分類（A/B/C/D）・6仮説事前確率・Candidate A-E比較評価・Study95-97追加・アーキテクチャbake-off構造へ再設計。strategy_review_2026-04-13/2026-06-28のゼロベース再検証を正式化 |

## ユーザー確定事項（v2前提・2026-07-13）

1. **RSR42は選定バイアスが濃厚**（Study75C: 選定バイアス≈+11pp・E1監査後トーン「+7〜12pp・上限寄り」）。
2. **売買ロジックはほぼ白紙**——Entry/Exitロジックのstandaloneエッジは証明されていない前提で再検証する。
3. **strategy_review_2026-04-13・strategy_review_2026-06-28の両方をゼロベースで再検証する**
   （両文書は凍結参考値へ格下げ。以降の意思決定根拠として使用禁止。FUJIKO 2.0確立後に新レビューを起草）。

## ★Study番号（v1から継承・ユーザー決裁待ち）

- 本セッション暫定名「Study76/76D/77」→ **Study75D/E/F**への改番提案（v1提案を維持）。
  canon Study76（Clenow純正ベンチマーク）・Study77（Exit構造置換）は**未実行・予約継続**。
- 新規研究はStudy87以降を採番（Study78-86は旧正典予約済み）。v2はv1のStudy87-94を維持しつつ
  **Study95/96/97を追加起案**する。全番号はユーザー決裁待ち（roadmap-governance原則2）。
- **表記衝突注意**: 旧正典の「ARCH-A〜E」（MN/PEAD/TSMOM/リードラグ/小型）と本書Part3の
  「Candidate A〜E」（FUJIKO 2.0アーキテクチャ候補）は**完全に別物**。

---

# Part 1: 既存Study01〜77の分類（A/B/C/D）

分類基準:
- **A（高信頼で再利用可能）**: 結論がRSR42のリターン水準に依存しない（インフラ・手法・エンジン整合性・バイアス実測そのもの）。
- **B（条件付き再利用）**: RSR42上で測定されたが、定性的・構造的発見の方向は普遍性が高い。**絶対値（pp・閾値）は再較正必須**。
- **C（再検証必須）**: Production採用判定・絶対数値・意思決定根拠。honestユニバース上でfresh run再測定するまで使用禁止。
- **D（廃棄候補）**: 前提が構造的に無効、または上位監査で既に反証済み。**削除はしない**（governance原則: 凍結保持・意思決定根拠としての使用のみ禁止）。

## A: 高信頼で再利用可能

| Study | 内容 | 再利用理由 |
|---|---|---|
| Study75 / 75A | J-Quants基盤・`database/market/`・Universe C生成器・PIT audit | データソース/手法はRSR42バイアスと独立。次世代研究の唯一の共通土台 |
| Study75B（手法・定性のみ） | 生存者バイアス測定手法+「廃止銘柄は同一ルール下で系統的に損失側」 | ※Delta_A=-16.96pp数値自体はD（セマンティクス汚染） |
| Study75C + E1妥当性監査 | 選定バイアス≈+7〜12pp・生存者バイアス≈0の実測 | **本再構築の根拠そのもの**。E1監査で確信度較正済み |
| Study75D/E/F（暫定） | Dynamic42 v1の病理診断（候補枯渇・ffill汚染・warm-upアーティファクト・OOS単一銘柄集中79%） | 診断事実として信頼可（診断対象のDynamic42 v1はD） |
| Study12/13/16/17/18/49 | 執行コスト・デプロイ・権限・信頼性・運用レジリエンス | インフラ検証でありリターン水準非依存 |
| Study37/38 | 再現器収束・エンジンforensics（data vintage確定・CB説明率0%） | エンジン整合性の証明。ユニバース非依存 |
| Study78（手法枠組みのみ） | RoR/Monte Carlo/Bootstrap/感度sweepの計算基盤 | 手法は再利用可。※数値はC |
| 横断資産 | WF5fold・採用ゲート機械判定・容量診断計装（skip_stats/cand_series等）・governanceプロセス | Study75D-F分析で有用性を実証済み。そのまま次世代で使用 |

## B: 条件付き再利用（定性構造○・絶対値×）

| Study | 保持する定性結論 | 再較正が必要な部分 |
|---|---|---|
| Study60/61 | 日足OHLCV由来の情報天井（IC 0.06-0.14）・BigWinner=Day1判別不能・FalseHero率高 | IC絶対値・比率はRSR42銘柄群上の測定 |
| Study62-69 | BW保護のシンプル特徴量限界・PLB構造（RSR系ExitがBWをPeak前に切る）・NEV分解手法 | 全pp値・NEV円額 |
| Study40/63/64 | Exit micro最適化の天井は低い（構造的発見） | 上界+6.37pp等の絶対値 |
| Study21-24 | Exit/Signal failure分類学（FALSE_BREAKOUT最多・tail_capture計測法） | n=22-35の小標本・RSR42上 |
| Study53/54/55/56 | ENTRY_DEFICIT（候補ゼロ日66%）・機会損失構造・Quality score設計 | 候補密度はユニバース構築に強依存→新ユニバースで再測定 |
| Study8系列/25/27/28 | Portfolio geometry/allocation/adaptive CAPのEXHAUSTED判定 | 有利ユニバース上でも棄却→方向は保守的に妥当。ただしパーセンタイル系パラメータ前提が変われば限定的に再訪可 |
| Study29-36 | ATR Risk Sizing=REMOVE・MTF=NEGATIVE・equal weight優位（機能単位棄却） | 新アーキテクチャでの機能寄与は同一とは限らない |
| Study42/43A/44/45/46 | ロット制約力学（lot=83%寄与・最小効率資本概念・cliff構造） | ¥20-30M等の具体額（RSR42価格帯依存） |
| Study10/11 | 資本注入ポリシー（annual優位・injection破壊） | α前提値 |

## C: 再検証必須

| Study | 内容 | 再検証理由 |
|---|---|---|
| Study9・dyn_rsr42_bear_rs0採用・Exit RSR70系（2026-06-04採用群） | 全ADOPT判定 | 採用根拠のベースラインがhindsight RSR42 |
| Study47/48/50/51/52 | D_ATR_EQ Production系譜（IS12.37%/OOS13.48%等） | Production数値の土台そのもの |
| Study57/58A/59 | DPO Case E・QualityReplacement（Shadow稼働中） | 同上。Shadow判定基準も再アンカー要 |
| Study70-73 | Addon REJECT・Provenance監査・CAND_B(rsr_exit=75) CONDITIONAL | 監査手法は健全だが比較基準値が汚染 |
| Study74/74B/80A/81 | 資本スケーリング**BLACK判定はa fortiori維持**（Study75Cが補強） | 判定方向は不変。¥20-30M等の数値はhonestユニバースで再測定 |
| Study78（数値） | RoR/MC結果 | RSR42トレード分布ベース |
| Study19/20 | Limited Live GO判定 | 前提期待値がRSR42。**実弾の即時停止は不要**（CIRCUIT/DD監視は独立に機能）だが、増資・拡大判断の根拠には使用禁止 |
| **strategy_review_2026-06-28** | Production全体レビュー | **ユーザー確定: ゼロベース再検証**。FUJIKO 2.0確立後（Study94後）に新版を起草 |
| M1後Official値（IS12.22%/OOS11.42%） | 公式基準値 | 凍結参考値へ格下げ済み（Study75Dで実施） |

## D: 廃棄候補（凍結保持・意思決定根拠として使用禁止）

| 項目 | 廃棄理由 |
|---|---|
| Dynamic RSR42 v1（trailing 12M composite・月次top42） | Production候補として不適格（ユーザー確定済み）。候補枯渇+測定自体が非クリーン（Study75E: contaminated版もfixed版もアーティファクト持ち） |
| Study75B Delta_A=-16.96pp | パーセンタイル・セマンティクス汚染で無効（Study75Cで立証済み） |
| Study65の Add-on +3.16pp期待 | Study70で反証済み（IS/OOSとも負） |
| Study52キャッシュ由来の旧数値（20.51%等） | 汚染事件で廃止済み（fresh_run_required恒久化の起源） |
| **strategy_review_2026-04-13** | 2世代前の基盤（yfinance価格・Study52汚染期・旧エンジン）。ユーザー確定: ゼロベース対象＝新版で置換。凍結参考のみ |
| 旧正典Phase2-5（Study79/82-86）のCP1-CP4目標改定ロジック | 「RSR42素の実力10-12%」前提の上の緩和プログラム。**一時凍結**（廃棄ではなく、Part5完了後に再設計）。例外: Study80（ARCH-Aスプレッド実測）・Study83（TSMOM）はRSR42非依存のため並行可（ユーザー決裁次第） |

---

# Part 2: 6仮説の事前確率評価

各仮説に事前確率（主観・自前データ+文献ベース）・根拠・検証Studyを付す。
事前確率は**検証の優先順位付けと期待情報量の見積り**に使う（採否判定には使わない——判定は各StudyのKillゲートが行う）。

| # | 仮説 | 事前確率 | 検証Study |
|---|---|---|---|
| 1 | Cross-sectional momentumは依然有効（日本） | **0.45** [0.35-0.55] | **Study95（新設・最優先）** |
| 2 | Market→Sector→Stock階層が存在する | **0.35** [0.25-0.45] | Study88+89 |
| 3 | 日本にセクターモメンタム持続性がある | **0.40** [0.30-0.50] | Study88 |
| 4 | 現行RSR定義は無効 | **0.75** [0.65-0.85] | canon Study76（絶対スコア代替） |
| 5 | 現行Entry/Exitロジックにまだエッジが残る | **0.30** [0.20-0.40] | **Study96（新設）** |
| 6 | Dynamic universeは必須 | 二分割（下記） | Study90/91 |

## 根拠詳細

**仮説1（CS momentum有効・P=0.45）**
- 反証側: 文献上、日本はクロスセクショナル・モメンタムが歴史的に最も弱い市場（Asness系の
  「momentum everywhere except Japan」）。自前データでは Dynamic42 v1（trailing return top42）の
  IS崩壊（CAGR -16〜-25%）——ただし候補枯渇・warm-upアーティファクトの交絡あり（Study75E/F）。
- 支持側: PITランダムユニバースのOOSでRSR42パーセンタイルが70%（median超）に留まる＝モメンタム系
  選抜に窓外でも若干の情報が残る示唆。2010年代以降の日本モメンタム改善報告。トレンドフィルター
  （TSMOM成分）併用時の改善余地。
- **重要**: これまで factor-level（戦略エンジンを通さない素のデシル分析）で一度も検証していない。
  Dynamic42 v1の崩壊は「実装病理」と「ファクター不在」が未分離。Study95がこれを分離する。

**仮説2（階層存在・P=0.35）**
- Study75F: corr(top_sector_share_t, return_t+1)=-0.252（弱い負・n=96）——ただし
  「セクターを狙って設計していないシステム」上の観測であり設計された検証ではない（反証にならない）。
- 文献: Moskowitz-Grinblatt型industry momentumは米国で確立、日本での再現報告は薄い。
- 仮説1が偽なら本仮説の実効確率はさらに低下（銘柄層で拾えないものをセクター層で拾える見込みは薄い）。

**仮説3（セクター持続性・P=0.40）**
- 仮説2より要求が弱い（伝播不要・セクター自己相関のみ）。データは`companies.parquet`のSector17/33で
  即検証可能。検証コスト最小級。

**仮説4（RSR定義無効・P=0.75）**
- 支持側（強い）: RSR=プール相対パーセンタイルはプールサイズにセマンティクスが従属。
  U0→U1→U2→U3の勝率単調劣化（Study75B）・min_rsr=75/Top30がプール規模で別戦略化する実証・
  Dynamic42がプール42固定を強制された事実。**移植可能な定義としては壊れている**。
- 留保: 固定42銘柄プール内では設計通り機能していた（「定義が無効」＝「全文脈で無意味」ではない）。
- 帰結: FUJIKO 2.0のランキングは**プールサイズ非依存の絶対スコア**（Clenow slope×R²等）を第一候補とする。

**仮説5（Entry/Exitロジックにエッジ残存・P=0.30）**
- 反証側（強い）: Study75C E1——同一ロジックをPITランダム42銘柄群に適用した結果 median IS -2.48%
  （同期間TOPIXは正リターン）＝**ロジック単体はパッシブに劣後**。Exit micro最適化の低天井（Study40/63/64）。
  Entry filter系のWF棄却の積み重ね（ET score30・Entry Velocity系）。ユーザー認識「売買ロジックはほぼ白紙」と整合。
- 支持側: ランダム抽選はモメンタム事前選抜を欠く——本ロジックはモメンタム銘柄上で使う設計であり
  「選抜×ロジック」交互作用は未検証。ATRトレイリング等の災害回避成分は選抜と独立に価値を持ちうる。
- 帰結: **「ロジックあり vs 素のモメンタム保有」の帰属分解（Study96）を経ずにFUJIKO 2.0へ
  現行Entry/Exitを持ち込まない**。

**仮説6（Dynamic universe必須）— 二分割して評価**
- 6a: 「PIT/ルールベースのhindsightなしユニバースが必須」= **P≈0.95（実質確定）**。
  Study75Cが+7〜12ppのhindsight優位を実測した以上、静的hindsight選定への回帰はあり得ない。
- 6b: 「月次高回転top-Nローテーションがその正しい形」= **P≈0.25**。
  Dynamic42 v1の病理（avg_candidates 0.27-0.55・turnover 44.6%/月・在籍中央値3ヶ月・warm-up
  アーティファクト）が反証的。年次/四半期PITリバランスや広域固定指数（TOPIX500型）でも6aは満たせる。
- 帰結: 「Dynamic＝月次ローテーション」という思い込みを排し、Study90で回転頻度・プール規模を
  独立の設計変数として比較する。

---

# Part 3+4: FUJIKO 2.0 候補アーキテクチャと評価

全候補に共通の前提: Layer 0 = Market Regime（TOPIX>MA200・既存再利用）/ ユニバース = Study75A
Universe C系のPIT基盤（仮説6a）/ ランキング = プールサイズ非依存の絶対スコア（仮説4への対処）。

## Candidate A: Market → Stock（Clenow style）

| 項目 | 評価 |
|---|---|
| エッジ源泉 | 銘柄レベルCSモメンタム（slope×R²絶対スコア）+ トレンドフィルター。仮説1に全面依存 |
| 想定失敗モード | 仮説1が日本で偽ならエッジなし。モメンタムクラッシュ（2022型）への脆弱性。β相関高 |
| 必要データ | **既存で完備**（database/market OHLCV + Universe C） |
| 計算コスト | **最小**（fresh run 1セット。canon Study76として設計済み） |
| 実装複雑性 | **最小**（層を減らす方向。RSRパーセンタイル機構を除去） |
| 期待頑健性 | **最高**（パラメータ最少・文献裏付け最強・Study75D-Fの病理の大半を構造的に回避） |
| 対応Study | **canon Study76**（予約済・Universe C上で実施。Universe統制ポリシー2026-07-04適用） |

## Candidate B: Market → Sector → Stock

| 項目 | 評価 |
|---|---|
| エッジ源泉 | industry momentum（仮説3）+ セクター内選抜への伝播（仮説2）。分散統制の副次効果 |
| 想定失敗モード | セクター層がラグ・ノイズ源化。二重フィルターが候補枯渇を悪化（Dynamic42の教訓）。日本でのindustry momentum不在 |
| 必要データ | 既存で完備（companies.parquet Sector17/33） |
| 計算コスト | 低→中（Study88/89は純データ分析・BT不要。プロトタイプWFで中） |
| 実装複雑性 | 中（Layer2新規実装。Layer3は既存流用） |
| 期待頑健性 | 未知（H1/H2合格が前提。事前確率0.35は5案中で高くない） |
| 対応Study | Study88→89→92（v1設計を維持） |

## Candidate C: Market → Sector ETF → Stock Rotation

| 項目 | 評価 |
|---|---|
| エッジ源泉 | ETF価格＝取引可能なセクター集約シグナル（構成銘柄集計より低ノイズ・低コストの観測） |
| 想定失敗モード | 日本のTOPIX17連動ETFは流動性薄・履歴短→WF fold数不足。ETF追跡誤差がシグナル汚染。**データ未取得**（`database/market/etf/`はプレースホルダーのみ） |
| 必要データ | **新規取得必須**（J-Quants ETF系 or 手動）。5案中唯一データ障壁あり |
| 計算コスト | 中（データ取得+検証） |
| 実装複雑性 | 中〜高 |
| 期待頑健性 | 低め（履歴制約でWF検証力が構造的に弱い） |
| 対応Study | **Study97（新設・条件付き起案）**: Study88合格∧ユーザーのデータ取得承認の場合のみ |

## Candidate D: Dynamic Top500 momentum

| 項目 | 評価 |
|---|---|
| エッジ源泉 | 広域honestプール（候補枯渇の構造的回避）+ 絶対スコアランキング。プール規模がDynamic42病理の主因だったなら本命 |
| 想定失敗モード | warm-upアーティファクト残存（Study87未解決なら）。Top500末尾の流動性・ロット制約（¥3M・3スロットなら影響限定的だがStudy14/15の力学で要確認）。回転コスト |
| 必要データ | 既存で完備（Universe C生成器の設定変更のみ） |
| 計算コスト | 中（Study90診断は安価・Study91 fresh runで中） |
| 実装複雑性 | 中（Study87 warm-up修正が前提） |
| 期待頑健性 | 中〜高（warm-up修正成立が条件） |
| 対応Study | Study87→90→91（v1設計を維持。Top500をStudy90の筆頭案とする） |

## Candidate E: Hybrid

| 項目 | 評価 |
|---|---|
| エッジ源泉 | A-Dで実証された最強レイヤーの合成 |
| 想定失敗モード | **アーキテクチャ選択自体が新たなselection biasになる**（メタレベルで今回の失敗を反復する最悪ルート）。旧正典統治原則5「ハイブリッド化禁止」と正面衝突 |
| 必要データ/コスト/複雑性 | A-Dの結果依存のため未定 |
| 期待頑健性 | 定義不能（証拠が出るまで設計自体が存在しない） |
| 対応Study | **起案しない**。Study93完了後、複数候補が「異なる独立エッジで」合格した場合に限り、統治原則5の明示的上書きをユーザーへ提起する |

## 検証優先順位（情報価値/コスト比）

**A → D → B → C →（Eは原則なし）**

理由: Aは最安・最単純・文献最強で、かつ仮説1と仮説4の両方を一撃で検証する「基準器」。
Dはデータ完備でAとランキングを共有（差分＝ユニバース構築のみ＝きれいなablation）。
Bは仮説ゲート2段（H1∧H2）を安価な純データ分析で先に済ませられる。Cはデータ障壁で最後。

---

# Part 5: 新research roadmap — FUJIKO 2.0確立まで

**目標の再定義**: CAGRの即時最大化ではない。**(1) honestな地面（PITユニバース+クリーン測定）の上で
(2) エッジの存在をfactor-levelで確認し (3) 最小複雑性のアーキテクチャから順に積み上げ
(4) 全採否をhonestベースライン比較で決める**——research foundationの再建が目標。

## 統治原則（本ロードマップ固有・旧正典原則に追加）

1. **factor-first**: エンジン（3スロットPF・Exit群）を通す前に、素のファクター検証で仮説の生死を判定する。
   エンジン経由の検証は交絡（経路依存・候補枯渇）が大きすぎることをStudy75D-Fが実証した。
2. **honestベースライン必須**: 全候補の合否は (a) Universe Cランダム抽選median (b) TOPIX buy&hold
   (c) 静的RSR42のバイアス補正後推定値 の3基準との比較で判定。単独絶対値での合格宣言禁止。
3. **プールサイズ非依存ランキング必須**: パーセンタイル型パラメータ（min_rsr等）の新規採用禁止（仮説4）。
4. **現行実弾との分離**: 本ロードマップは研究レイヤー。現行Live（RSR42・¥3M）の停止/変更はASK_FIRST
   別決裁であり、Study94まで本ロードマップから実弾変更を派生させない。

## 依存関係図（v2）

```
Study75A-F（完了・前提事実）
   │
   ├─ Phase R0（並行3本・全て安価）────────────────────────────┐
   │    Study95: CSモメンタムfactor-level ground truth ★H0・最優先│
   │    Study87: Warm-up修正版ユニバース/RSR生成器（インフラ）      │
   │    Study90: ユニバース構築代替案ベンチマーク（診断のみ）        │
   │    Study88: セクターモメンタム持続性 ★H1（純データ分析）       │
   │                                                            │
   ├─ [H0不合格 → プログラム分岐: 全Candidate凍結・              │
   │   旧正典ARCH系（PEAD/TSMOM等・モメンタム非依存）へ転進提起]    │
   │                                                            │
   ├─ Phase R1（仮説ゲート第2段）
   │    Study89: セクター→銘柄伝播 ★H2 [Study88合格のみ]
   │    Study96: Entry/Exitロジック帰属分解 ★H5 [Study95合格∧Study87完了]
   │
   ├─ Phase R2（アーキテクチャ・プロトタイプ / clean universe上）
   │    canon Study76: Candidate A（Clenow・Universe C版）← 最初の基準器
   │    Study91: Candidate D（Dynamic v2 fresh run）[Study87∧Study90]
   │    Study92: Candidate B プロトタイプ [Study89合格のみ]
   │    Study97: Candidate C データ実現性 [Study88合格∧ユーザー承認のみ]
   │
   ├─ Phase R3（最終判定）
   │    Study93: 全生存候補 vs honestベースライン3基準の並列比較
   │    Study94: 静的RSR42終了可否の最終決定（Q4正式回答）
   │
   └─ Phase R4（体系確定・研究外の文書作業）
        FUJIKO 2.0仕様書確定 + strategy_review新版起草
        （2026-04-13版・2026-06-28版を正式に凍結参考へ置換）
```

## Study仕様（新設分。Study87/88/89/90/91はv1仕様を維持——本書末尾の注記参照）

### Study95: Cross-Sectional Momentum Factor-Level Ground Truth（★新設・最優先・H0）

- **目的**: 仮説1をエンジン非経由で直接検証する。Universe C（PIT・月次・平均907銘柄）上で
  トレイリングリターン（3/6/12ヶ月・skip-1ヶ月）による五分位/十分位ポートフォリオ
  （月次リバランス・equal weight・コスト込み/なし両方）のスプレッドを2016-2025で測定。
  サブ期間安定性（2018-2021 / 2022-2025）・regime条件付き（TOPIX>MA200時のみ）も併測。
- **成功条件**: Top-Bottomスプレッドが正かつ統計的有意（Newey-West t>2目安）、または
  regime条件付きで明確な正のスプレッド。サブ期間で符号一貫。
- **失敗条件**: スプレッドがゼロ近傍またはサブ期間で符号反転。
- **Kill基準**: 失敗＝**H0棄却 → Candidate A-E全凍結**。モメンタム系FUJIKOの続行根拠が消滅するため、
  旧正典ARCH系のモメンタム非依存案（PEAD=Study82・TSMOM=Study83）への転進をユーザーへ提起する。
  regime条件付きのみ合格の場合はCandidate設計をregime-gated型に限定して続行。
- **想定コスト**: 純データ分析・BTエンジン不要。1日規模。
- **依存**: Study75A（Universe C）のみ。**即時着手可能**。

**★実行結果（2026-07-14・Fable 5実行・完了）**: `reports/study95_cs_momentum_factor_level.md` /
`backtests/study95_cs_momentum_factor_level.json`。Universe C 119ヶ月・panel=108,895行・
12-1モメンタム/Clenow slope90d×R²の2ファクターで検証。
**判定=両ファクターともFAIL（Kill基準機械発動=True）**:
12-1モメンタムは12M年率spread=-1.83%（NW-t=-0.368）だが3M/6M/12MでIC t統計量が有意に負
（-2.77/-3.46/-2.83）——「ゼロ」というよりDecile10（過去勝者）が6M/12Mで明確に最下位となる
**弱い逆転（reversal）**。Clenowは1M/3M/6M horizonで正の単調性・スプレッド（1M=+7.02%等）を
示すが12Mで反転（-1.79%）し「複数期間で一貫」の合格基準を満たさない（Bear regimeで-7.64%・
t=-2.681と有意に悪化）。Sector-neutral・容量（ADV20/turnover）診断で交絡は否定。
**次アクション=ユーザー決裁待ち**: (a) Kill基準通りCandidate A-E全凍結→PEAD/TSMOM転進 or
(b) Clenow短期（1-6M）シグナルをregime-gated型で限定継続検証。詳細は`src/research_state.md`
2026-07-14セクション参照。

### Study96: Entry/Exitロジック帰属分解（★新設・H5）

- **目的**: 仮説5を検証する。同一クリーンユニバース・同一銘柄選抜（Study95合格ファクター）上で、
  (a) naiveベースライン（上位デシル買い・月次リバランス・ロジックなし）
  (b) 現行Entry群のみ追加（SEPA/ブレイクアウト条件）
  (c) 現行Exit群のみ追加（ATRトレイリング/RSR系Exit相当の絶対スコア版）
  (d) フル現行ロジック
  の4構成を比較し、各層の限界寄与（ΔCAGR/ΔCalmar/ΔMaxDD）を分解する。
- **成功条件**: (b)(c)(d)いずれかが(a)をCalmarで有意に上回る層が特定できる（正の寄与層の同定）。
- **失敗条件**: 全層が(a)に対して寄与ゼロまたは負（＝「売買ロジックは白紙」が定量確定）。
- **Kill基準**: 失敗の場合、FUJIKO 2.0のEntry/Exitは**naiveベースライン+災害ストップのみ**から
  再出発する（canon Study77「Exit構造置換」の問題意識をここに統合）。現行ロジックの移植は行わない。
- **想定コスト**: fresh run 4構成×1セット。1-2日規模。
- **依存**: Study95合格 ∧ Study87完了。
- **注記**: 失敗もプログラム前進である——「白紙」の定量確定は、以降の全実装コストを
  ロジック再発明ではなく選抜・ユニバース側へ集中させる根拠になる。

### Study97: Candidate C（Sector ETF）データ実現性調査（★新設・条件付き）

- **目的**: TOPIX17連動ETF等の価格履歴の取得可能性・流動性・履歴長を調査し、Candidate Cが
  WF検証に耐えるデータ基盤を持ちうるか判定する（判定のみ・戦略検証はしない）。
- **成功条件**: 主要セクターETFで2018年以前まで遡れる日次データが取得可能、かつ出来高が
  シグナル用途に足る水準。
- **失敗条件/Kill基準**: 履歴不足・流動性不足 → Candidate C恒久棄却候補としてユーザーへ提起。
- **想定コスト**: データ調査のみ。半日規模。
- **依存**: Study88合格 ∧ ユーザーのデータ取得承認（ASK_FIRST: 新規データソース追加）。

### canon Study76（Candidate A検証・予約済番号の本来内容を実施）

- v1の位置づけを変更しない——旧正典予約通り「Clenow純正ベンチマーク」を、Universe統制ポリシー
  （2026-07-04決裁）に従いUniverse C上で実施する。**Phase R2の最初の1本**とし、以降の全候補の
  「基準器」（最小複雑性リファレンス）とする。
- v2での追加要求: Study95の合格ファクター定義とランキング整合を取ること（slope×R²を第一候補としつつ、
  Study95で優位だったlookback/skipを反映）。判定はhonestベースライン3基準比較（統治原則2）。

### Study92/93/94（v1から判定基準のみ改定）

- **Study92**: 「Candidate Bプロトタイプ」と再定義（v1と実体同じ）。比較対象はStudy91のDynamic v2
  だけでなく**canon Study76（Candidate A）を含む**——「層を足す価値」はA比で測る。
- **Study93**: 「全生存候補（A/D/B/C のうち到達したもの）vs honestベースライン3基準」の並列最終比較。
  勝者==FUJIKO 2.0アーキテクチャ。全候補がベースラインに勝てない場合、
  「現時点でhonestなエッジは実証できない」を正式結論とし、実弾規模の縮退提案を含めてユーザーへ提起。
- **Study94**: 静的RSR42終了可否の最終決定（v1のまま）。加えて**strategy_review新版の起草をトリガー**する。

## 実行順序と概算コスト

| 順 | Study | 種別 | 規模 | ゲート |
|---|---|---|---|---|
| 1 | Study95 (H0) | 純データ分析 | 1日 | なし（即時可） |
| 1' | Study87 / Study90 / Study88 | 実装・診断・純データ分析 | 各0.5-2日 | なし（Study95と並行可） |
| 2 | Study89 (H2) | 純データ分析 | 0.5-1日 | Study88合格 |
| 2' | Study96 (H5) | fresh run 4本 | 1-2日 | Study95合格∧Study87 |
| 3 | canon Study76（Candidate A） | fresh run WF | 1日 | Study95合格 |
| 3' | Study91（Candidate D） | fresh run WF | 1日 | Study87∧Study90 |
| 4 | Study92（Candidate B） | 実装+fresh run WF | 数日 | Study89合格 |
| 4' | Study97（Candidate C調査） | データ調査 | 0.5日 | Study88合格∧ASK_FIRST |
| 5 | Study93 → Study94 | 比較・判定のみ | 1日 | Phase R2完了 |
| 6 | FUJIKO 2.0仕様書 + strategy_review新版 | 文書 | 1日 | Study94 |

Kill発生時の残余価値: どの段階で止まっても「何が存在しないか」のhonestな確定が残る
（Study95失敗＝日本CSモメンタム不在の自前実証・Study96失敗＝ロジック白紙の定量確定——
いずれも以降の資源配分を決める一級の情報）。

## プログラムレベルKill条件

- Study95（H0）失敗 かつ regime条件付きでも救済されない → モメンタム系FUJIKO全体を凍結、
  旧正典ARCH系（PEAD/TSMOM）へ転進提起。
  **★2026-07-14実行済み: 12-1モメンタム/Clenowともに機械的FAIL・regime-gated救済条件も不成立
  （bull_ann両ファクターとも+5%未達）。本条件が形式上発動。ただしClenow短期(1-6M)シグナルの
  扱いはユーザー決裁待ち（詳細=`reports/study95_cs_momentum_factor_level.md`）。全凍結の実行
  （Study87以降着手停止・PEAD/TSMOM転進の正式着手）はユーザー承認後に行う（自動実行しない）。**
- Phase R2の全候補がhonestベースライン3基準に勝てない（Study93失敗）→ 「実証可能なエッジなし」を
  正式結論として実弾規模の縮退を提起。
- 旧正典既定の「連続2四半期新規採用ゼロ→プログラムKill/縮退候補」は本ロードマップにも適用。

---

# 総括

- **Part1**: Study01-77をA（インフラ・手法・バイアス実測=土台として保持）/ B（構造的発見=方向のみ保持）/
  C（Production採用・絶対値=fresh run再測定まで使用禁止）/ D（凍結=意思決定根拠として廃棄）へ分類。
  strategy_review 2026-04-13=D・2026-06-28=C（ユーザー確定のゼロベース対象）。
- **Part2**: 最も確からしいのは「RSR定義の移植不能性」（P=0.75）と「PITユニバース必須」（P≈0.95）。
  最も疑わしいのは「Entry/Exitロジックのエッジ残存」（P=0.30）。**全仮説に検証Studyを割当済み**。
- **Part3/4**: 検証順 **A→D→B→C**（Eは原則起案しない）。Candidate A（Clenow）が最安・最頑健の基準器。
- **Part5**: Phase R0-R4・Study95/96/97新設・factor-first原則・honestベースライン3基準・
  プログラムレベルKill条件を定義。**最初の一手はStudy95（CSモメンタムfactor-level検証・即時着手可・1日）**。

**ユーザー決裁待ち事項（v2時点）**:
1. Study75D/E/F改番の承認（v1から継続）
2. Study95/96/97の採番・起案承認
3. canon Study76をPhase R2基準器として実施する承認（Universe統制ポリシー適用確認込み）
4. 旧正典Phase2-5凍結の正式承認（v1から継続。例外並行: Study80/83）
5. strategy_review_2026-04-13/2026-06-28の凍結参考値格下げの正式承認

---

# 付録: v1継承Study仕様（Study87-91・変更なしで有効）

## Study87: Warm-up修正版ユニバース/RSR生成器

- **目的**: Study75E（旧76D）で確定した「0埋め修正は不連続アーティファクトを生む」問題を解消する、
  真にクリーンなDynamic Universe基盤を実装する。在籍開始直後`mom_period`日間はモメンタム由来の
  エントリーシグナルを抑制する（NaN継続・シグナル生成スキップ）ウォームアップ処理を追加。
- **成功条件**: avg_simultaneous_holdings・avg_candidatesがcontaminated版・0埋め版のいずれとも
  異なる値を示し、かつ月次membership境界での不連続なmomentum spike発生率が実測ゼロに近いこと
  （検証指標: 在籍開始直後21日間のmom_arr分布が在籍4ヶ月目以降の分布と統計的に区別できないこと）。
- **失敗条件**: ウォームアップ処理を入れても依然として境界での異常値が検出される。
- **Kill基準**: 2回の設計修正でも境界アーティファクトが解消しない場合、月次ローテーション型
  ユニバース自体の実装難易度が高すぎると判断し、Dynamic Universe路線全体を再評価する
  （Study90の結果次第でTOPIX固定指数ベース案に切替）。
- **想定コスト**: 実装+単体検証のみ・新規BTなし。数時間規模。
- **依存**: Study75D/E/F（完了済み）。

## Study88: セクターモメンタム持続性（Sector Momentum Persistence）

- **目的**: TOPIX17・TOPIX33の各業種について、月次リターンの自己相関・持続性（1〜3ヶ月ホライズン）
  を実測する。Moskowitz & Grinblatt型の業種モメンタム文献の日本市場での再現性検証。
- **成功条件**: 業種リターンの1ヶ月ラグ自己相関が統計的に有意（p<0.05）かつ符号が正、
  out-of-sample分割（例: 2018-2021 fit / 2022-2025 test）でも同方向の持続性が確認される。
- **失敗条件**: 自己相関がノイズ水準（|IC|<0.05）またはin-sample/out-of-sampleで符号が反転する。
- **Kill基準**: 失敗の場合、H2以降（Study89・Study92）は起案しない。
- **想定コスト**: 純データ分析・新規BT不要。1日規模。
- **依存**: なし（Study75系列のデータ基盤のみ。Study87/95と独立・並行可）。

## Study89: セクター→銘柄伝播検証（Sector→Stock Propagation）

- **目的**: 「強いセクターに属する銘柄」が、銘柄固有モメンタムを統制した後も追加的な予測力を
  持つかを検証する（二重ソート or Fama-MacBeth型回帰）。
- **成功条件**: セクター強度ダミーの回帰係数が、銘柄固有モメンタムを含む多変量モデルでも
  統計的に有意（p<0.05）かつ経済的に意味のある大きさ（月次+0.3%以上）。
- **失敗条件**: セクター強度の限界寄与が非有意、または銘柄固有モメンタムに完全吸収される。
- **Kill基準**: 失敗の場合、Study92（Candidate Bプロトタイプ）は起案しない。
- **想定コスト**: 純データ分析・新規BT不要。半日〜1日規模。
- **依存**: Study88合格。

## Study90: ユニバース構築代替案ベンチマーク（特性比較のみ・BT不要）

- **目的**: Dynamic42 v1の失敗（IS崩壊・候補枯渇）を繰り返さないため、TOPIX100・TOPIX500・
  Prime市場全体・TOPIX17セクター内トップN・ハイブリッドの各ユニバース構築案について、
  **バックテストの前に**候補プールの特性（規模・流動性分布・セクター多様性・月次turnover・
  候補数見込み）をStudy75E/F流の診断手法で比較する。
- **成功条件**: 少なくとも1案が「Dynamic42 v1で確認された病理（avg_candidates<0.5・
  zero_candidate_day>60%）を構造的に回避できる」と診断段階で判断できること。
- **失敗条件**: 全案がDynamic42 v1と同様の候補希薄化リスクを抱える。
- **Kill基準**: 全案失敗の場合、月次ローテーション型ユニバース自体を撤回し、静的だがhindsightの
  ない構築法（年次リバランス・PIT基準のTOPIX500固定型等）を再検討する。
- **想定コスト**: 純データ分析・実装不要。1-2日規模。
- **依存**: Study75A。Study88/95と並行可。

## Study91: 採用ユニバースでのDynamic42 v2 fresh run（Candidate D検証）

- **目的**: Study90で選定した候補ユニバース構築案 + Study87のウォームアップ修正を組み合わせた、
  真にクリーンなDynamic Universe構成でのfresh run（IS/OOS/WF5fold）。
- **成功条件**: WF5foldでの年別分散がStudy75E両版より縮小し（2020/2021/2022年の極端な負値が緩和）、
  avg_candidates>1.0（Dynamic42 v1比2倍以上）。
- **失敗条件**: Study87の修正後も同様の病理（薄い候補・極端なDD）が再現する。
- **Kill基準**: 失敗の場合、月次ローテーション型設計自体を恒久的に棄却し、静的年次リバランス型へ
  方針転換する（旧正典の恒久閉鎖リストへの追加をユーザーへ提案）。
- **想定コスト**: fresh run（IS/OOS/WF5fold）1セット。数十分〜1時間規模。
- **依存**: Study87・Study90。

---

*v1作成: Sonnet 5, 2026-07-13 / v2改定: Fable 5, 2026-07-13。新規BT・コード変更・実弾変更なし。
全数値は既存成果物（Study75A-F・research_state.md）からの引用。v1のStudy87-91仕様は変更なしで有効
（Study92-94は判定基準のみ改定・本文参照）。番号確定・起案承認は全てユーザー決裁事項。*
