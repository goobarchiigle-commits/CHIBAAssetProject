# CHANGELOG — quality/（RGP: quality premium / junk avoidance）

## 2026-07-24
- `quality_mf_classic_v1.0.md` 初版Frozen。role=Calibration Benchmark。GP/A(Novy-Marx)+ACC(Sloan)+F-Score(Piotroski)のパーセンタイルコンポジット・年次8月末形成・VW D10−D1。**データ留保: GP/長期負債/流動比率は/v2/fins/summary非提供**（再現に補助データ要）。verdict=UNTESTED。
- `quality_mf_practical_v1.0.md` 初版Frozen。role=Production Candidate。parent=classic。OP/TA(RMW系)+ACC+F-Score-J 8項目（全フィールド実在）・四半期リバランス・EW上位五分位。verdict=UNTESTED。
- `quality_mf_smallmid_v1.0.md` 初版Frozen。parent=classic。時価総額50-1,000億×高B/M tercile×F-Score-J≥6（Piotroski原典構成回帰）・半期・N≥30規則・momentum複合禁止条項。derived_fromに自プロジェクトStudy間接証拠6点を事前登録。verdict=UNTESTED。
- 起案原本: `literature_evidence_2026-07-24.md`（book.pdf/03書籍10冊+02所収論文の定量抽出・フィールド可用性表）。
- 3仕様とも実装・BT未承認（ASK_FIRST）。

## 2026-07-24（同日追記・ユーザーレビュー反映）
- `quality_mf_smallmid_v1.0.md` → **`quality_value_smallmid_v1.0.md` へ改名**（strategy名: Quality MF SmallMid → Quality Value SmallMid）。理由: 内容の核心がPiotroski原典のB/M条件（Value）であり"SmallMid"のみでは曖昧との指摘。role=Research Hypothesis追加。禁止事項をgovernance §9のRGP越境禁止リストに統一表記。
- classic/practicalにも role フィールド追加・禁止事項（RGP越境）行を追加。
- governance文書に role 3値定義（Calibration Benchmark/Production Candidate/Research Hypothesis）+ §9 RGP越境禁止を新設。REGISTRY.mdをRGP-first構造+Role列に再編。
