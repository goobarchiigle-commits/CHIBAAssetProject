# Core Research Closure — Long Only Core Architecture

**日付**: 2026-07-04
**性格**: Study01〜81 + Core Architecture Completion Review (Final Audit, 2026-07-04) を統合し、Core Architecture研究の正式終了を宣言する文書。
**制約遵守**: 新規BT=ゼロ / コード変更=ゼロ / 新規仮説=ゼロ / 改善案=ゼロ。既存成果物のみ使用。
**関連成果物**: `reports/core_decision_record.md`（Closed Research一覧・再開条件） / `reports/architecture_handover.md`（Study74-86引き継ぎ）
**拘束エビデンス**: Study01〜81 / Final Research Roadmap (2026-07-04) / Core Architecture Completion Review (2026-07-04) / `reports/core_open_questions.md` / `reports/core_evi_matrix.md`

---

## 1. Core Architectureの最終結論（1文）

> **現行Long Only Core Architecture（max_positions=3・RSR42固定Universe・¥3M・日次判定・翌日寄付執行）は、Study01〜81を通じて実力が補正後10〜12%・オラクル込み理論上界16〜18%であることが実測で確定し、制約固定下でのCAGR30%到達およびCAGR30%∧Calmar1.5同時達成は共に成立しないことが確定した一方、内部の全構造要素（max_positions・Exit・Entry・幾何配分・資本規模）は個別に改善余地を反証済みであり、現行構成は当該制約の下での局所最適である。**

---

## 2. 明文化

### 2-1. 何が証明されたか

| # | 証明事項 | 根拠Study |
|---|---|---|
| 1 | 現行構成の実測実力（M1適用後fresh run） | FULL 11.22% / OOS 11.42% / IS 12.22% / WF 4/5 |
| 2 | 補正後の素の実力レンジ | 10〜12%（Production Comprehensive Audit 2026-07-02） |
| 3 | Exitマイクロ最適化の天井 | 理論+1.63pp・現実MC-0.93pp（Study63） |
| 4 | Exit構造オラクル上界（実現不能な理想値） | +6.37pp（Study64） |
| 5 | BW予測ベース保護は情報不足で実装不能 | Oracle比11.9%<30%（Study69） |
| 6 | 日足OHLCV特徴量の情報天井 | IC 0.06〜0.14・安定特徴量ゼロ（Study60/61） |
| 7 | 幾何・配分の天井@¥3M | oracle ΔCalmar+0.069<0.10（Study25/27/28） |
| 8 | 資本増加の実効上限 | ¥20Mでlot丸め解消分+1.12ppのみ・全解除でも13.1-13.3%収束（Study74 PartA） |
| 9 | 枠拡大（機会損失回収）は無効 | +0.33pp・Sharpe悪化（Study53）、¥20M以降は負（Study74 PartA） |
| 10 | クラスター重複仮説は逆方向 | 同cluster forward_20 +3.46% > 別cluster +1.71%（Study81） |
| 11 | オラクル合算上界 | 16〜18%（10-12%＋6.37pp＋≈0＋≈0） |
| 12 | 「3ポジ最適∧見逃しα」は矛盾ではなく測定レベルの混同 | 単体α(forward_20+2.8%)≠限界寄与(≦+0.33pp)。希薄化常時×α取得散発(cap_saturation40.6%)＋相関コピー(分散縮小24.8%)＋時間集中(2023年同時好調)（Study74B/74B-RCA/80A/81・Final Audit§3.5） |

### 2-2. 何が反証されたか（仮説→反証Study、Final Audit Part4-1の再確認）

Exitマイクロ最適化(+3pp超) / BW予測保護の実装可能性 / 日足OHLCVからの新規MLシグナル / 幾何配分@¥3Mの有意改善 / Adaptive CAP・レジームsizing・MSW / Add-on拡張(+3pp) / Entryフィルター・タイミング改善 / Conditional RSRポリシー / Lot cost ratio緩和 / 資本増による¥20-30M・CAGR≥22%再現(Study42/43A/46の汚染前数値) / クラスター重複による4銘柄目無効説 / 枠拡大による機会損失回収 — **全12件、Study番号付きで反証済み**（詳細: Final Audit Part4-1）。

### 2-3. 何が未解決か

台帳全10件（`reports/core_open_questions.md`）のうち、Architecture選択に影響する残存4件:

| OQ | 内容 | 帰属先 |
|---|---|---|
| OQ1 | 同日競合候補のリスク相関構造（分散縮小24.8%の因果分解） | Study85 |
| OQ3 | mom_period=21の過学習疑い（崖+PEAK_AT_DEFAULT） | Study76/77 |
| OQ4 | Survivorshipバイアス実幅（推定-1〜3pp・未実測） | Study75 |
| OQ5 | Exit構造オラクル+6.37ppの構造的回収可否 | Study77 |

他6件（OQ2/6/7/8/9/10）は本監査または前回Final Auditで論理的に解消済み、または研究ではなく決断待ち。

### 2-4. 未解決が意思決定を変えない理由

1. **OQ1**: 純効果（枠拡大の限界寄与≦+0.33pp）は4系列独立実測済み（Study8系/41/53/74PartA）。因果分解の結果がどの比率であっても「枠拡大しない」という決定は変わらない。
2. **OQ3**: 覆る方向は「実力がさらに低い」側のみ。過学習が確定しても、緩和優先順位（資本→L/S MN→Universe→情報源→時間構造）は変わらない。
3. **OQ4**: 同上、下方リスクのみ。Survivorshipバイアスは全緩和軸共通の分母であり、Core Architecture自体の選択（現行構造の維持/破棄）には影響しない。
4. **OQ5**: 全量回収しても理論上界16-18%止まり＝30%判定は変わらない。回転Exitは「予測せず構造で守る」経路として既にStudy77（後継プログラムPhase2）に予約済みであり、これはCore（固定制約内改善）の問いではなく制約緩和プログラムの問いである。

**共通構造**: 4件全てにおいて、「情報が更新されても現在の決定（現行Core構成を維持し、制約緩和プログラムへ移行する）は不変」という性質を持つ。これが所謂EVI=Low(Core)の定義そのものである。

---

## 3. Core研究の終了条件 確認

正典（Final Research Roadmap 2026-07-04・統治原則3）の起案基準「期待値+2pp未満の研究は起案禁止」を終了条件の基準として適用する。

| 条件 | 判定 | 根拠 |
|---|---|---|
| **EVI High = 0** | ✅ **満たす** | `reports/core_evi_matrix.md`: 指定4項目（max_positions=3/Opportunity Cost/Portfolio State/Time Competition）は全てCore EVI=Low。台帳10件中、Core内EVIがHighまたはMediumの項目はゼロ（High/Mediumは全て後継プログラム側=Study75/76/77/85の勘定） |
| **Architecture Decisionへ影響するOpen Question = 0** | ✅ **満たす** | `reports/core_open_questions.md`集計: 「Architecture選択へ影響する件数」=**0**（10件全てが非影響、または後継プログラムの決定事項） |
| **期待情報価値+2pp以上の未実施研究 = 0（Core内）** | ✅ **満たす** | Core内で未実施かつ+2pp以上を期待できる研究テーマは存在しない。オラクル合算上界16-18%は既に全チャネル測定済みであり、Core内に残る唯一の変数（OQ1〜OQ5相当）はいずれも制約緩和側に帰属し、Core自体の起案対象ではない |

**3条件すべて充足。終了条件は満たされている。**

---

## 4. 最終判定

# **Core Research Closed**

- Long Only Core Architecture（固定制約下の改善研究）は本書をもって正式に研究終了とする。
- 終了は「運用終了」を意味しない。運用・月次decay監視は継続する（正典・恒久タスク）。
- 終了後の全ての新規研究起案は、Final Research Roadmap Part3（Study74〜86統一プログラム）の枠内でのみ有効。Core内の恒久閉鎖14項の再訪は表現を変えても禁止。
- Closed Research全一覧・再開条件は `reports/core_decision_record.md` を正典とする。
- Architecture Program（Study74-86）への引き継ぎ事項は `reports/architecture_handover.md` を正典とする。

---

*作成: Core Research Closure監査, 2026-07-04。新規BT・コード変更・新規仮説・改善提案なし。全数値は既存成果物からの引用。*
