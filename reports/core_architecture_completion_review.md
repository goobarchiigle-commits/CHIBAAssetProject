# Core Architecture Completion Review — Final Audit

**日付**: 2026-07-04
**性格**: 第三者視点最終監査。Study01〜81 + Final Research Roadmap (2026-07-04) を対象に「Core Architecture（現行Long Only・固定制約）の研究を終了できるか」のみを判定する。
**制約遵守**: 新規BT=**ゼロ** / Productionコード変更=**ゼロ** / 改善案・新戦略提案=**なし** / 既存成果物のみ引用（research_state.md / reports/*.md / backtests/*.json / MEMORY索引）。
**関連成果物**: `reports/core_open_questions.md`（未解決事項台帳） / `reports/core_evi_matrix.md`（EVI評価行列）

---

## §0 監査範囲と現行公式値

- **Core Architecture** = 現行Long Only・固定制約（max_positions=3 / RSR42系Universe / ¥3M / 日次判定 / 翌日寄付 / COST_ONE_WAY 0.155%）。
- **「研究終了」の定義** = 固定制約内の改善研究の恒久停止。運用・月次decay監視・後継プログラム（Study75〜86）は本判定の対象外。
- **現行公式基準値**（D_ATR_EQ・M1適用後・2026-07-04 fresh run）: IS 12.22% / OOS 11.42% / FULL 11.22% / WF 4/5 (avg 17.99%) / 2022 -2.95% / IS Calmar 0.671 / Bootstrap P(>0)=0.984。
- 補正後の素の実力: **10〜12%**（Survivorship+Selection補正、Production Comprehensive Audit 2026-07-02）。

---

## Part 1: 未解決事項一覧

詳細台帳は `reports/core_open_questions.md`。要約:

| ID | 未解決の問い | 状態 | Arch選択へ影響 | 影響しない理由 |
|---|---|---|---|---|
| OQ1 | 同日競合候補の分散縮小24.8%（リスク相関構造）の因果分解 | 未解決（Study80A副産物・Study81スコープ外） | **なし** | 純効果（枠拡大の限界寄与≦+0.33pp）は4回独立実測済み。因果分解は説明価値のみで意思決定を変えない |
| OQ2 | CAP_MISS矛盾（同品質候補見送り∧枠拡大でCAGR悪化）の因果完全証明 | 部分解明（Study74B-RCA/80A/81） | **なし** | 本監査§3.5で論理的に解消（測定レベルの混同）。残余はOQ1と同一 |
| OQ3 | mom_period=21の過学習疑い（崖+PEAK_AT_DEFAULT） | 未解決（Study78 Part3検出） | **なし**（下方リスクのみ） | 覆る方向は「実力がさらに低い」側。30%不可判定を強化する。Study76/77へ申し送り済み |
| OQ4 | Survivorshipバイアス実幅（推定-1〜3pp・未実測） | 未実測（Study75未実施） | **なし**（下方リスクのみ） | 同上。全緩和軸共通の分母としてStudy75がプログラム既定 |
| OQ5 | Exit構造オラクル+6.37ppの構造的回収可否（回転Exit） | 未検証（Study64上界のみ確定） | **なし** | 全量回収でも16-18%止まり=30%判定不変。回転ExitはStudy77（構造置換=後継プログラム）に予約済み |
| OQ6 | クラスター粒度依存（raw sector p=0.0有意 vs macro 4分類 p=0.0661非有意） | 未解決（Study81解析3） | **なし** | 集中度の統計判定が粒度で揺れるのみ。緩和5軸の優先順位を変えない。Study85相関設計へ |
| OQ7 | CAND_B WF5/5の¥3M固有性（資本増で4/5→3/5） | 測定済み・決裁待ち（Study74） | **なし** | パラメータ選択（S1決裁）でありArchitecture選択ではない |
| OQ8 | Lev Audit(2026-06-13) PF=0.24-0.29 vs Study80A forward_20=+2.8%の測定系不整合 | 監査所見（本書§3.5(a)） | **なし** | 旧エンジン×システムExit損益 vs 現エンジン×生ドリフトの測定系差。両者とも「枠拡大無効」という同一決定を支持 |
| OQ9 | exit_policy="A" defer判定のentry_price結合 | 記録済み（M1-RCA） | **なし** | 実装結合の設計課題。Study77申し送り済み |
| OQ10 | idle-cash日の25%に勝者候補存在（Q1_idle_when_winner） | 測定済み（Study74B） | **なし** | entry頻度=🔴構造限界に分類済み。回収手段（Entryフィルター/枠拡大）は恒久閉鎖#7/#11でREJECT確定 |

**Part1結論**: Architecture選択（緩和5軸の優先順位・Long Only Coreの骨格判断）へ影響する未解決事項は**ゼロ**。

---

## Part 2: Expected Value of Information（要約）

評価行列全体は `reports/core_evi_matrix.md`。指定4項目:

| 項目 | 結論が変わる可能性 | 新意思決定の可能性 | 研究コスト | **EVI (Core)** |
|---|---|---|---|---|
| max_positions=3 | **Low**（独立実測4回: Study8系/41/53/74PartA。全て≦+0.33pp or 負） | Low（恒久閉鎖#11維持のみ） | Medium | **Low** |
| Opportunity Cost | **Low**（Study80Aで607件個別+forward全データセット化済み。総量は既知、回収手段は全て負判定済み） | Low | Low | **Low** |
| Portfolio State（状態依存仮説） | Medium（因果未証明）だが決定影響ゼロ（純効果実測済み） | Low | Medium-High（counterfactual再構築BT要） | **Low** |
| Time Competition（候補到着×スロット解放の時間不整合） | Medium（回転Exitで構造解消の可能性=Study64上界） | **後継プログラムで既に予約済み**（Study77がこの問いそのもの） | High | **Low（Core）/ High（Study77帰属）** |

**Part2結論**: Core内で追加研究にHigh EVIを持つ項目はゼロ。High EVIは全て制約緩和側（Study75/77/80）に存在し、正典ロードマップが既に予約済み。

---

## Part 3: Decision Impact Analysis

**命題**: 「制約固定では30%は困難」は追加研究で覆るか。

**判定: 覆らない**（覆るとすれば困難側への改定のみ）。

### 根拠チェーン（全て実測・Study番号付き）

1. **実測実力**: FULL 11.22% / OOS 11.42%（M1後fresh run、2026-07-04）。補正後素の実力10〜12%（Audit 2026-07-02）。
2. **各改善チャネルの天井（全チャネル測定完了）**:
   - Exitマイクロ: 理論+1.63pp / 現実MC -0.93pp（Study63）
   - Exit構造オラクル（未来知識込み・実現不能上界）: +6.37pp（Study64）
   - BW予測保護: Oracle比11.9%<30% → STOP（Study69）
   - 情報天井: 日足OHLCV IC 0.06〜0.14・安定特徴量ゼロ（Study60/61）
   - 幾何・配分: oracle ΔCalmar+0.069<0.10（Study25/27/28）
   - 枠拡大・機会損失回収: +0.33pp（Study53）/ ¥20M以降は負（Study74 PartA）/ ブロックシグナルPF=0.24-0.29（Lev Audit 2026-06-13）
   - lot丸め: 上限+1.12pp・¥20Mで解消済み（Study74 PartA）
   - 資本スケーリング: ¥3M→¥20Mで+0.89ppのみ・CP1=BLACK（Study74）
   - Entry core: WF通過実績ゼロ（Roadmap 2026-06-13）
3. **オラクル合算上界**: 10-12% + 6.37pp + ≈0 + ≈0 = **16〜18%**。30%との差12pp+を埋める未測定チャネルはCore内に存在しない。Study74B/80A/81により最後の未測定候補（機会損失の質・クラスター分散仮説）も測定完了 — 前者は限界寄与≈0、後者は**棄却**（Study81: 同cluster群forward_20 +3.46% > 別cluster +1.71%、仮説と逆方向）。
4. **資本メカニクス構造天井**: lot/max_pos/symbol_cap全解除でもCAGR 13.1-13.3%収束・MaxDD-27.5%（Study74 PartA）— 制約を外しても30%に接近しないことの直接証明。
5. **下方リスクの残存**: OQ3（mom_period過学習疑い）とOQ4（survivorship未補正-1〜3pp）は、追加測定で覆る場合「実力はさらに低い」方向にしか作用しない。

**副次判定**: CAGR30%∧Calmar1.5同時目標=理論的矛盾（Final Architecture Review 2026-07-03 §6 / Final Research Roadmap Part1）も、Study78実測（レバ1.3xでMaxDD中央値-13.95%・RoR 1.02%はIIDブートストラップの楽観側であり実測FULLとの乖離を明記済み）により反証材料なし。

---

## §3.5 監査上の追加所見 — 「3ポジ最適 ∧ 見逃し候補にα」矛盾の論理的解消

本監査の付加価値として、既存成果物のみで当該矛盾が**解消可能**であることを示す（改善案ではない。事実の整理のみ）。

**結論: これは矛盾ではない。「候補単体のα」と「ポートフォリオ限界寄与」という異なる測定量の混同である。**

**(a) 単体α ≠ 限界寄与**
見送り候補のforward_20=+2.79〜+2.88%（Study80A/81）は候補単体の無条件ドリフト。一方、4枠目解放のポートフォリオ限界寄与は直接実測済み: +0.29pp(¥3M)〜-0.42pp(¥20M)（Study74 PartA）/ max=10で+0.33pp・Sharpe悪化（Study53）。**両命題は同時に真**であり論理的衝突はない。

**(b) 希薄化コストは常時・α取得は散発**
均等配分エンジンでは4枠目が全営業日の既存ポジ目標比重を約33%→25%へ希薄化する（M2'文書化: entry経路cap=MAX_POS_WEIGHT=0.40 / addon headroom=0.375）。一方CAP_MISSはcap_saturation_rate=40.6%の日にしか発生しない（Study74B）。**コストは毎日、便益は4割の日だけ** — 純効果≈0は算術的に整合する。

**(c) 4銘柄目は「新しい賭け」ではなく「保有中の賭けの相関コピー」**
同日競合3候補群の分散縮小率=24.8%のみ（独立なら理論値≈66.7%、日跨ぎ無作為抽出実測67.3%、Study80A）。期待値は加算されてもリスクがほぼ線形加算されるため、分散便益なしにDDが悪化する（全制約解除時MaxDD-27.5%、Study74 PartA）。Study81の逆方向結果（同cluster見送りの方が高forward）はこれと整合 — 「強いクラスターに候補が同時多発する時こそ候補αが高い」= 4枠目は既に保有しているファクターの買い増しである。

**(d) 時間集中**
見送り最多年2023（121件）は採用側も収益+144.9万円・勝率65.4%で最高（Study74B-RCA解析2）。候補は強レジームに同時多発し、その時既存3ポジは最良局面にある。4枠目は「既に乗っている上昇」を希薄化した比重で買い増す行為に等しい。

**∴ 統一的説明**: 見逃しαは実在する。しかしその回収コスト（(b)希薄化 + (c)相関重複 + (d)時間集中）が便益を系統的に相殺し、純効果≦+0.33ppであることが**4系列の独立実測**（Study8系 / Study41 / Study53 / Study74 PartA）で確定している。残る未知は(b)(c)(d)の寄与比率の分解のみであり（OQ1）、どの比率であっても「枠拡大しない」という決定は変わらない → **EVI=Low、Core内で追及する価値なし**。この問いに投じてきた労力は本所見をもって終了してよい。

---

## Part 4: Core Completion Criteria

### 4-1. 十分反証できた仮説（Study番号付き）

| 仮説 | 反証Study | 判定 |
|---|---|---|
| Exitマイクロ最適化で+3pp超可能 | Study40/63（天井+1.63pp・現実-0.93pp） | 反証済 |
| BigWinner予測保護は実装可能 | Study68/69（Oracle比11.9%・n=3.8件/fold） | 反証済 |
| 日足OHLCVから新規MLシグナル抽出可能 | Study60/61（IC≤0.14・安定特徴量ゼロ） | 反証済 |
| 幾何・配分で有意改善可能@¥3M | Study25/27/28（oracle+0.069<0.10） | 反証済 |
| Adaptive CAP/レジームsizing/MSW有効 | Study8系/MSW WF（全REJECT） | 反証済 |
| Add-on拡張で+3pp（Study65の+3.16pp楽観） | Study70（IS-0.44pp/OOS-0.39pp・純ドラッグ） | 反証済 |
| Entryフィルター・タイミング改善可能 | ET score30 (WF1/5) / Entry Velocity (-34pp) / Study24 | 反証済 |
| Conditional RSRポリシー有効 | Study7（全REJECT） | 反証済 |
| Lot cost ratio緩和有効 | Study44（Seg1 -8.49pp） | 反証済 |
| 資本増で¥20-30M・CAGR≥22%再現（Study42/43A/46） | Study74（+0.89ppのみ・CP1=BLACK・旧24.15%は汚染前数値） | 反証済 |
| 4銘柄目が増えないのはクラスター重複のせい | Study81（同cluster +3.46% > 別cluster +1.71%・逆方向・REJECT） | 反証済 |
| 機会損失回収（枠拡大）で改善可能 | Study53 (+0.33pp・Sharpe悪化) / Study74 PartA（¥20M以降負） | 反証済 |

### 4-2. 反証できなかったが、Architecture選択へ影響しない未解決事項

OQ1（リスク相関構造の因果分解）/ OQ2残余 / OQ6（クラスター粒度）/ OQ8（測定系不整合）— いずれも純効果が直接実測済みのため、解明されてもCore内の意思決定（枠拡大しない・恒久閉鎖維持）は不変。帰属先は全て後継プログラム（`reports/core_open_questions.md`参照）。

### 4-3. 今後調査しても意思決定が変わらない事項

- max_positionsの再スイープ・条件付き4枠目（恒久閉鎖#11・4回実測済み）
- Opportunity Costの精密化（回収手段が全滅している以上、精度向上に決定価値なし）
- Hidden Factorの完全分解（§3.5の通り、どの寄与比率でも決定不変）
- symbol_cap調整（全資本水準でΔ=0.00pp・非拘束、Study74 PartA）

### 4-4. 研究終了宣言の可否

**可**。条件3点: ①恒久閉鎖14項（Final Research Roadmap）の維持 ②OQの帰属先（Study75/76/77/85）変更禁止 ③fresh run原則の継続。

---

## Part 5: Final Verdict

# **B. COMPLETE WITH OPEN QUESTIONS**

- **Core研究（現行Long Only・固定制約内の改善研究）は終了可能**。未解決事項は残るが、全件がArchitecture選択に非影響であり、かつ後継プログラム（Study75/76/77/85）に帰属済み。
- Core内に「追加研究で意思決定が変わる」問いは存在しない（EVI High項目ゼロ）。
- 「制約固定では30%は困難」は覆らない。残存する未測定要素（survivorship / mom_period）は困難側にのみ作用する。
- 長年の労力対象だった「3ポジ最適∧見逃しα」矛盾は§3.5をもって**論理的に解消済み**として閉鎖してよい。

### 最終出力（指定4項目）

| 項目 | 結論 |
|---|---|
| 研究終了可否 | **終了可**（Verdict B） |
| 残すべき未解決事項 | OQ1（同日候補のリスク相関構造→Study85）/ OQ3（mom_period過学習→Study76/77）/ OQ4（survivorship幅→Study75）/ OQ5（Exit構造回収→Study77）の4件のみ。他は閉鎖 |
| 今後のArchitecture選択へ影響するか | **影響しない**。緩和5軸の優先順位（資本→L/S MN→Universe→情報源→時間構造）はいずれのOQの帰結でも不変 |
| Core研究を恒久終了してよいか | **恒久終了可**。ただし「研究終了」であって「運用終了」ではない — 運用+月次decay監視は継続、Study76/77はCore骨格の後継検証として正典の枠内で実施 |

---

*作成: 第三者監査, 2026-07-04。新規BT・Productionコード変更・改善提案なし。全数値は既存成果物からの引用。*
