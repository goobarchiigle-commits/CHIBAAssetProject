---
strategy: Quality MF Classic
version: "1.0"
status: Frozen
verdict: UNTESTED
role: Calibration Benchmark
parent: none
derived_from: [NovyMarx2013(GP/A), Piotroski2000(F-Score), Sloan1996(Accruals), OShaughnessy(コンポジット方式), Berkin&Swedroe(QMJ/Ball et al.)]
rgp: quality premium / junk avoidance（クロスセクション・低回転）
conventions: common_conventions_v1.0
created: 2026-07-24
origin: research/strategies/quality/literature_evidence_2026-07-24.md
---

# Quality Multi-Factor Classic v1.0 — 文献ベースベンチマーク（変更禁止）

**仮説**: 日本株に品質プレミアム（高収益・高会計品質・財務健全企業の超過リターン）が存在する（H0: D10−D1スプレッド ≤ 0）。
**用途**: 較正原点ベンチマーク。**実装禁止・バックテスト禁止のまま仕様のみFrozen**。判定はユーザー承認後のfresh runでのみ記入。FAILでも永続。

## 採用ファクター（文献で個別に支持される3本のみ・追加禁止）

| # | ファクター | 定義（完全定量） | 方向 | 文献 |
|---|---|---|---|---|
| F1 | 粗利益収益性 | **GP/A = 売上総利益 / 総資産**（直近本決算） | 高いほど良 | Novy-Marx 2013 |
| F2 | 会計発生高 | **ACC = (当期純利益 − 営業CF) / 総資産**（直近本決算） | **低い**ほど良 | Sloan 1996（CF法）・Ball et al.（cash-based最強） |
| F3 | F-Score | Piotroski 9項目合計（0-9点）: ①ROA>0 ②CFO>0 ③ΔROA>0 ④CFO/TA>ROA ⑤長期負債/TA低下 ⑥流動比率上昇 ⑦新株発行なし ⑧売上総利益率上昇 ⑨総資産回転率上昇（各1点・前期本決算比） | 高いほど良 | Piotroski 2000 |

## 仕様（固定値）

| 項目 | 仕様 |
|---|---|
| Universe | Study75 PITユニバース − 金融業（33業種: 銀行/証券・商品先物/保険/その他金融の4業種除外=Novy-Marx SIC6除外の日本移植）+ 株価 ≥ 100円・ADV20 ≥ 5,000万円 |
| Scoring | 形成日ごとにF1・F2（符号反転）・F3をユニバース内パーセンタイルランク化し**単純平均=コンポジットスコア**（O'Shaughnessy方式・ウェイト調整禁止） |
| Ranking | コンポジットスコア十分位。D10=最上位 |
| Signal | 研究: D10ロング−D1ショート（**時価総額加重**=Novy-Marx VW忠実）。live検証: D10ロングオンリー |
| 形成日 | **毎年8月最終営業日**（年1回）。使用データ=形成日時点でFYEから5ヶ月以上経過した最新本決算（Piotroski 5ヶ月ラグ規則。米国6月末形成の日本適応——3月期決算中心の日本では8月末形成で全FYEに5ヶ月ラグを確保） |
| Entry | 形成日翌営業日 寄付 |
| Exit | 次回形成日翌営業日 寄付（**保有12ヶ月・年次全面リバランス**） |
| 欠損規則 | F1-F3いずれか計算不能（前期データ欠損含む）の銘柄は当年除外（補完禁止） |
| Primary判定 | D10−D1スプレッド（12M・コスト後・VW）> 0 かつ NW-t ≥ 2.0 → PASS。月次系列 n ≥ 60（conventions n_min上書き・明記） |
| 禁止事項（RGP越境・governance §9準拠） | RSRランキング追加・決算イベント/サプライズ条件追加・テーマ株条件追加・出来高急増条件追加・ブレイクアウト/トレンドフォロー条件追加 |

## データ上の留保（設計時点で確定している事実）

- **売上総利益（F1分子・F3⑧）は`/v2/fins/summary`に存在しない**（COGS非提供）。**長期負債・流動比率（F3⑤⑥）も同様**。本仕様の完全再現には決算短信XBRL/EDINET等の補助データソースが必要。これは仕様の欠陥ではなくデータ調達要件として記録（実装承認時の前提条件）。
- 代替指標への置換は本仕様書では**禁止**（置換した瞬間にClassicではなくなる。置換版=Practical v1.0が別仕様として存在）。
