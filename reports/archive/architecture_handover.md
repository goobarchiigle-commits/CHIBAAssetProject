# Architecture Program Handover — Core → Study74-86統一プログラム

**日付**: 2026-07-04
**性格**: Core Research Closure（`reports/core_closure.md`）に伴い、Architecture Program（Final Research Roadmap Part3・Study74〜86）へ引き継ぐ事項を確定する。Coreが確定した事実・残した問い・継承すべき統治原則・資産を一覧化する。
**制約遵守**: 新規BT=ゼロ / コード変更=ゼロ / 新規仮説=ゼロ / 改善案=ゼロ。

---

## 1. Coreから引き継ぐ確定事実（前提として扱ってよい・再検証不要）

| 事実 | 数値 | Architecture Programでの用途 |
|---|---|---|
| Core実測実力（M1後・fresh run） | FULL 11.22% / OOS 11.42% / IS 12.22% / WF 4/5 | Study76のClenow純正ベンチマーク比較基準値 |
| 補正後の素の実力レンジ | 10〜12% | 全緩和軸の起点値 |
| オラクル込み理論上界（Core固定制約下） | 16〜18% | 「制約緩和が必須」の定量的根拠。Study74-86着手の正当性の数値的基盤 |
| 資本メカニクス構造天井（lot/max_pos/symbol_cap全解除でも） | CAGR 13.1-13.3%収束・MaxDD-27.5% | Study74後継（Study79レバ設計）のベースライン |
| max_positions=3は資本非依存の構造限界 | ¥20M以降解除でCAGR悪化 | Study85（ポートフォリオ統合）でのCore枠設計の前提 |
| CAND_B（rsr_exit=75）WF5/5は¥3M固有 | ¥10-30Mで4/5→4/5→3/5 | S1決裁・Study74後継の資本設計への留意事項 |
| mom_period=21の過学習疑い | 崖16→19:+3.01pp,19→21:+3.48pp・PEAK_AT_DEFAULT | Study76の骨格評価で要考慮（現行構成に暗黙のチューニング依存がある） |
| クラスター（macro/factor）ベースの分散仮説 | REJECT（Study81） | Study85の相関設計は「クラスターラベル」ではなく実測相関係数ベースで行うべき（示唆） |
| 同日競合候補の分散縮小率 | 24.8%（独立理論値≈66.7%） | Study85の結合RoR・相関行列設計における最重要インプット |
| RSR等価性（見送り候補=採用候補・中央値81.0） | 完全一致 | Universeフィルター自体は健全（Study75でのUniverse再設計時に「フィルター基準は妥当、枠数が制約」という前提を持ち込んでよい） |

---

## 2. 引き継ぐ未解決事項（Open Questions・Core側では非影響）

`reports/core_open_questions.md` より、Architecture Program側で価値を持つ4件:

| OQ | 内容 | 引き継ぎ先 | Program側でのEVI | 具体的な引き継ぎ内容 |
|---|---|---|---|---|
| OQ1 | 同日競合候補のリスク相関構造（希薄化/相関重複/時間集中の寄与比率） | **Study85** | Medium | ポートフォリオ統合設計時、Satelliteスリーブとの結合相関行列を組む際、Core内部の「見かけの分散24.8%」を過大評価しないこと。結合RoR計算にはこの縮小率を織り込む必要がある |
| OQ3 | mom_period=21の過学習疑い | **Study76/77** | Medium | Clenow純正ベンチマークでmom_period依存の構造が異なる場合、現行D_ATR_EQとの比較でこのパラメータ依存が交絡しないよう注意。純正構成側でも同様の崖がないか感度チェック推奨 |
| OQ4 | Survivorshipバイアス実幅（推定-1〜3pp） | **Study75** | High（正典最上流指定済み） | 全緩和軸（Study76-86）の数値の信頼区間を決める共通分母。Study75完了までは他Study結果に暗黙のバイアス幅を仮定しないこと |
| OQ5 | Exit構造オラクル+6.37ppの構造的回収可否 | **Study77** | High（後継プログラム内） | 「予測せず構造で守る」回転Exit経路は未検証。Study76勝者構成に対する3アームWFとして正典に既に予約済み。BW予測ベース保護（Core閉鎖#2）とは別物である点を明確に区別して評価すること |

---

## 3. 継承する統治原則（変更なし・Architecture Programでも遵守必須）

Final Research Roadmap 統治原則7項をそのまま継承:

1. fresh run必須（Study52汚染事件の恒久対策）
2. 採用ゲート固定: WF5/5 ∧ 2022非悪化 ∧ ΔCAGR≥+1pp ∧ Bootstrap P(>0)≥95% ∧ コストストレス
3. 起案予算制: 四半期あたり新規Study 3本上限。期待値+2pp未満の研究は起案禁止
4. Kill Criteria事前定義必須
5. ハイブリッド化禁止（新アーキテクチャ各案は独立で白になるまで既存系と混載禁止）
6. ASK_FIRST遵守（新規スクリプト・PARAMS_LOCKED変更・口座/データ契約・実弾変更）
7. プログラムレベルKill: 連続2四半期で新規採用ゼロ → 月次メンテに縮退・運用フェーズ移行

**追加継承（Core Closureで確定）**: Closed Research 15項目（`reports/core_decision_record.md`）は表現・命名を変えた再訪も禁止。Architecture Program内の新Study起案時、これらの領域を「別名」で内包していないか起案前チェックを行うこと。

---

## 4. 継承する研究資産（追加BTなしで再利用可能・Program側の起点）

| データセット | 内容 | 再利用先 |
|---|---|---|
| `trade_dataset_v2.json` | 採用309件・v2拡張 | 全Study共通 |
| `missed_candidates_full.json` | 見送り607件個別レコード | Study75/81/85 |
| `forward_return_dataset.json` | forward_5/10/20/40/60・MFE・MAE | Study75/85 |
| `opportunity_cost_dataset.json` | Sector/Regime/Rank/skip_reason別 | Study85 |
| `correlation_dataset.json` | 同日候補集中度 | Study85（OQ1解決の直接インプット） |
| `study81_analysis_template.py` | Mann-Whitney U・KS検定・Permutation Test・Bootstrap CI実装済み | Study76-86全般 |
| `study78_trade_dataset.json` / `study78_risk_summary.json` / `study78_mc_distribution.json` / `study78_sensitivity.json` / `study78_drawdown_analysis.json` / `study78_risk_contribution.json` | RoR/MC/Sensitivity/DD/LossCluster/RiskContrib一式 | Study74/79/81/85/86 |
| `study74_capital_scaling_2026-07-04.json` / `study74_integrated_review_2026-07-04.json` | 資本弾力性4点×2構成の全測定 | Study79 |

---

## 5. Architecture Program起点となるCore側の判断材料

- **CP1（Study74）= BLACK確定**。目標改定（黒→CAGR15-20%/Calmar1.2）はユーザー決裁待ちのまま。Architecture Programはこの決裁を前提とせず、Study74白黒どちらの分岐（Final Research Roadmap CP1参照）にも対応できる設計で進めること。
- **緩和優先順位（Constraint Relaxation Map）は本Closureで変更なし**: 資本(Study74)→Return Structure/L-S MN(Study80)→Universe/Data(Study75/81)→Information Source(Study82/84)→時間構造/Exit(Study76/77)。
- Core側で「良好」と確定した構成要素（Entry: RSR42フィルター品質は健全、Exit: 多層構造はStudy77検証まで現状維持が正しい判断）は、Architecture Program側の新設計でも参照点として使ってよい（ゼロから再発明する必要はない）。

---

## 6. 引き継ぎの完了条件

本Handoverは以下をもって完了とする:
- `reports/core_closure.md`の「Core Research Closed」宣言
- `reports/core_decision_record.md`のClosed Research 15項目の確定
- 本書のOpen Questions 4件の引き継ぎ先確定

以降、Core Architectureに関する新規研究起案は本書記載のStudy74-86の枠内でのみ行う。Architecture Program側での成果・失敗は、それぞれのStudy固有のガバナンス（成功/失敗条件・終了条件・Production採用条件、Final Research Roadmap Part3参照）に従う。

---

*作成: Core Research Closure監査, 2026-07-04。新規BT・コード変更・新規仮説・改善提案なし。全数値は既存成果物からの引用。*
