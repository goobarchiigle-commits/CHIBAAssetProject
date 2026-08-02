# Core Decision Record — Closed Research Registry

**日付**: 2026-07-04
**性格**: Core Architecture研究終了（`reports/core_closure.md`）に伴う正式決定記録。恒久閉鎖領域の一覧・再開条件・再開禁止理由を確定する。本書はFinal Research Roadmap §「絶対にやってはいけない研究」12領域＋追加禁止2項を継承し、Study74B/80A/81で新たに確定した閉鎖事項を統合する。
**拘束力**: 本書記載の閉鎖領域は、表現・命名・入口を変えた再訪も禁止。起案された場合、内容を問わず却下する（Final Research Roadmap統治原則を継承）。

---

## 決定事項（Decision）

**Long Only Core Architecture（固定制約: max_positions=3 / RSR42系Universe / 日本株 / ¥3M / DD予算15%CB / 翌日寄付執行 / 日次判定）の内側で行う改善研究を、2026-07-04付で恒久終了する。**

### 決定の位置づけ
- 対象: PARAMS_LOCKED区画内の全パラメータ・現行Exit/Entry/幾何配分ロジックの改善研究。
- 非対象: 制約緩和プログラム（Study74-86）、運用フェーズの月次decay監視、パラメータ決断（S1等のASK_FIRST事項）。
- 上位文書: Final Research Roadmap (2026-07-04) の完結宣言、Core Architecture Completion Review (2026-07-04) Verdict B、本書付随 `reports/core_closure.md`。

---

## Closed Research 一覧

### A. 恒久閉鎖14項（Final Research Roadmapより継承・変更なし）

| # | 領域 | 閉鎖根拠 | 再開条件 | 再開禁止理由 |
|---|---|---|---|---|
| 1 | Exitマイクロ最適化（閾値・延期・条件分岐） | Study40 EXHAUSTED / Study61-69全STOP / 天井+1.63pp(Study63) | **なし（永久）** | 現実MC効果-0.93pp（負）。上界自体が小さく、微調整の余地は理論・実測両面で消尽 |
| 2 | BigWinner検出・保護（予測ベースExit全般） | Study69 STOP（Oracle比11.9%・n=3.8件/fold） | **なし（永久）**。ただし「予測せず構造で守る」経路はStudy77で別途検証中（本閉鎖の対象外） | サンプル数の物理的制約（BW発生頻度）は研究では解消不能 |
| 3 | 日足OHLCV特徴量からの新規MLシグナル | IC天井0.06-0.14(Study60) / 安定特徴量ゼロ(Study61) | データソース変更時のみ（Study75以降のJ-Quants拡張データ等・これは「新規MLシグナル」ではなく別研究として扱う） | 情報天井は特徴量エンジニアリングでは超えられない構造的制約 |
| 4 | ポートフォリオ幾何・資本配分 @¥3M | Study25/27/28 EXHAUSTED（oracle天井+0.069） | 資本規模が変わった場合のみ（Study74後継＝Study79/85の枠内） | オラクル（未来知識込み）上界が既に閾値0.10を下回る |
| 5 | Adaptive CAP / レジームsizing / MSW | Study8系列・MSW・regime_2 全REJECT | **なし（永久）** | 複数バリアント総当たりで全滅、パラメータ空間は消尽 |
| 6 | Add-on拡張（サイズ・タイミング・適用範囲） | Study70 REJECT（純ドラッグ） | **なし（永久）** | IS/OOS双方で負のΔCAGR実測済み |
| 7 | Entryフィルター・タイミング・閾値 | ET score30 WF1/5 / Entry Velocity -34pp / Study24系列終了 | **なし（永久）** | Entry改変はWF通過実績ゼロ（30%目標下で最大の過学習経路と確定） |
| 8 | Conditional RSRポリシー | Study7 全REJECT | **なし（永久）** | 全ポリシーバリアント既に試行済み |
| 9 | Lot cost ratio緩和 | Study44 全REJECT（Seg1 -8.49pp） | 資本規模拡大時のみ再評価対象（Study74で再検証済み・¥20M以降解消のため事実上closed） | Seg1で大幅悪化実測 |
| 10 | 監査の監査（Study71-73型再帰） | fresh run規則1行で代替 | **なし（永久）** | 統治原則の恒久ルール（fresh_run_required=true）で代替済み |
| 11 | max_positions拡大・機会損失回収 | Study53(+0.33ppのみ) / ブロックシグナルPF=0.24-0.29 / Study74 PartA(¥20M以降負) / Study81(クラスター重複説REJECT) | **なし（永久）**。PARAMS_LOCKED=3を破棄する場合はArchitecture変更（Study85統合設計）としてのみ扱う | 4系列独立実測で純効果≦+0.33ppまたは負。CAP_MISS矛盾も測定レベルの混同と解消済み（本記録§下記C） |
| 12 | Entry core改変全般 | WF通過実績ゼロ | **なし（永久）** | 30%目標下で最大の過学習経路と確定 |

**追加禁止2項（継承）**:
- 検証独立前のハイブリッド化（新アーキテクチャ部品の白確定前の既存系混載）
- 目標を仮説に賭けること（未検証経路を前提とした目標・資本・レバ設定）

### B. 本書で新たに確定する閉鎖事項（Study74B/80A/81・Final Audit由来）

| # | 領域 | 閉鎖根拠 | 再開条件 | 再開禁止理由 |
|---|---|---|---|---|
| 13 | クラスター（factor/macro）ベースの4銘柄目抑制ロジック | Study81 REJECT（同cluster forward_20 +3.46% > 別cluster +1.71%・逆方向） | **なし（永久）**。ただしOQ1（リスク相関構造そのもの）はStudy85の相関行列設計として別枠で継続 | 狭義仮説「同クラスター＝低期待値」は解析4・6で一貫して逆方向に反証 |
| 14 | CAP_MISS矛盾の「解決策探索」（見送り候補を救済するEntry/枠拡大変種） | Final Audit §3.5（測定レベルの混同として論理的に解消） | **なし（永久）**。因果の完全分解（寄与比率）はStudy85のOQ1としてのみ許可 | 見逃しαの存在と回収不能は両立確定事実。回収を試みる設計変更は#11の再訪に等しい |
| 15 | mom_period=21の「感度が滑らか」という前提に基づく追加最適化 | Study78 Part3（崖+PEAK_AT_DEFAULT検出・backtest_summary.json 2026-03-15の旧判定を上書き） | Study76（Clenow純正ベンチマーク）での骨格再設計時のみ、Core内パラメータ最適化としては再開しない | 過学習疑いのある値をさらにチューニングする行為はEVI負 |

---

## Architecture Decisionへの影響 = ゼロ（再確認）

Closed Research 15項目のいずれも、以下のArchitecture Decision（現行構成の骨格）を変更しない:

- max_positions=3を維持する
- 現行Exit多層構造（RSR_EXIT/RSR_MOMENTUM_EXIT/ATR_TRAILING/turtle_exit）を維持する
- 現行Universe（RSR42系）を維持する（Survivorship検証待ちのStudy75着手までは暫定継続）
- 現行資本規模¥3Mでの運用を継続する（Study74 CP1=BLACK・目標改定はユーザー決裁待ちの状態を維持）

---

## 決定の再考トリガー（本書自体の失効条件）

以下いずれかが発生した場合のみ、本Decision Recordは失効し再評価対象となる:
1. Study75（Survivorship-free再構築）でRSR42固定Universeのバイアスが閉鎖根拠を覆す規模（>-3pp）で確定した場合
2. Study76/77でCore骨格そのものが置換候補として白判定された場合（この場合も「Core改善」ではなく「Architecture置換」として扱う）
3. Study85でSatellite統合により資本配分・相関構造の前提が変わった場合

上記以外の理由（会話・思いつき・再検討要望等）による再訪は、本書の拘束力の下で却下する。

---

*作成: Core Research Closure監査, 2026-07-04。新規BT・コード変更・新規仮説・改善提案なし。*
