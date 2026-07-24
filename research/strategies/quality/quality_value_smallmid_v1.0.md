---
strategy: Quality Value SmallMid
version: "1.0"
status: Frozen
verdict: UNTESTED
role: Research Hypothesis
parent: quality_mf_classic_v1.0
derived_from: [Piotroski2000(高B/M×quality・小型/低カバレッジで効果最大), Study82E(効果のSmall/Mid集中・Large≈0), Study95/99/83/82D(継続型シグナル全滅→momentum複合禁止), Study110A(Top10%集中55-56%→分散必須), Study100(静的名簿FATAL→PIT必須), Study101(旧手法TOPIX全面劣後→判定基準=TOPIX B&H)]
rgp: quality premium / junk avoidance（Value条件付き・Small/Mid限定）
conventions: common_conventions_v1.0
created: 2026-07-24
origin: research/strategies/quality/literature_evidence_2026-07-24.md
---

# Quality Value SmallMid v1.0 — 自プロジェクト実測適応型（Value×Quality×Small/Mid）

**命名注記（2026-07-24）**: 初版名"Quality MF SmallMid"はPiotroski原典のB/M条件を含意せず曖昧との指摘を受け改名。本仕様はQuality単体ではなく**Value（高B/M）×Quality（F-Score-J）の交差**が核心のため、両条件を名称に明示する。RGPフォルダは変更なし（`quality/`のまま。方向反転を伴わないため`edmr/`分離のような別RGP扱いにはしない）。

**仮説**: 品質シグナルの超過リターンは、日本市場の非効率が実測で確認された領域（Small/Mid-cap）×文献で効果最大とされる領域（高B/M・低カバレッジ）の交差で最も検出されやすい。
**自プロジェクト実測根拠（事前登録・全て間接証拠であることを明示）**:
- Study82E: 決算イベント効果はSmall/Mid集中（Size寄与1.06pp=最大）・Large-capほぼゼロ（t=0.34）→ Large排除
- Study95/99/83/82D: momentum系継続シグナルは日本で軒並み逆転 → **momentum/RS系の複合・オーバーレイを本仕様で明示的に禁止**（Study101「何が効いたか不明」の再発防止）
- Study110A: Top10%銘柄への利得集中55-56%・persistence弱 → 集中ポートフォリオ禁止・N≥30規則
- Study100: 静的名簿=hindsight FATAL → PITユニバース必須
- **品質ファクター自体の直接実測は本プロジェクトに存在しない**（quality系Study未実施）。verdict=UNTESTEDの根拠

**親版とのdiff**: ①Universe をSmall/Mid帯に限定 ②コンポジット十分位→**高B/M分位×F-Score-Jフィルタ**（Piotroski原典構成への回帰）③年次→半期リバランス。

## 仕様（固定値）

| 項目 | 仕様 |
|---|---|
| Universe | Study75 PITユニバース ∩ **時価総額 50億〜1,000億円**（Study82E帯域と同一）− 金融4業種 + ADV20 ≥ 1億円・株価 ≥ 100円・Eq > 0・監理/整理銘柄除外 |
| Value条件 | **B/M 上位1/3**（B/M = `Eq` / (`ShOutFY`×形成日前日終値)・ユニバース内tercile。絶対閾値でなく分位=時代不変） |
| Quality条件 | **F-Score-J ≥ 6**（8点制・Practical v1.0 §F3と同一定義・同一欠損規則） |
| 補助条件 | ACC = (NP−CFO)/TA がユニバース中央値未満（Sloan・発生高の悪い高B/M=バリュートラップ排除） |
| Selection | 3条件全通過銘柄を**等ウェイト**全数ロング。**N < 30 の期は通過全銘柄+残余現金**（レバレッジ/補充禁止・N不足は結果に記録） |
| 形成日 | **5月・11月の最終営業日**（年2回。本決算+中間決算の出揃い後・低回転）。使用データ=`DiscDate` ≤ 形成日の開示済み最新値のみ（PIT厳守） |
| Entry | 形成日翌営業日 寄付 |
| Exit | 次回形成日翌営業日 寄付でリバランス（**保有6ヶ月**） |
| 禁止事項（RGP越境・governance §9準拠） | RSRランキング追加・決算イベント/サプライズ条件追加・テーマ株条件追加・出来高急増条件追加・ブレイクアウト/トレンドフォロー条件追加・銘柄数の上位絞り込み（集中化）・B/M閾値の絶対値化 |
| Primary判定 | EWポートフォリオのコスト後月次超過リターン（vs TOPIX B&H）> 0 かつ NW-t ≥ 2.0 → PASS。月次 n ≥ 60・平均N ≥ 30（conventions上書き・明記） |
| Secondary（診断のみ） | ① 時価総額50-300億 vs 300-1,000億のサブ帯域差（82E予測: 小さい側が強い）② F-Score-J各項目の寄与分解 ③ Value条件なし版との差分（qualityの独立寄与） |

## リスク（事前明示）

- 高B/M×Small/Mid×低ADVはバリュートラップ・流動性テールを内包。イベント単位でなくポートフォリオのMaxDD分布を必須記録。
- 品質プレミアムの日本市場実測は本プロジェクト初——先行Studyの間接証拠は方向を保証しない。Primary判定が唯一の判定。
- 実装・バックテストはユーザー承認後（ASK_FIRST）。
