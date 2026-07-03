# CRO Decision Memo — 2026-07-03

**前提**: Study01〜Study73 + Productionコード + Comprehensive Audit（2026-07-02）を統合。目的関数=「CAGR30%以上・Calmar1.5以上を長期維持」に対する批判的評価と意思決定。

## 結論

**目的「CAGR30%・Calmar1.5長期維持」と検証済み事実の間には約2.5倍の乖離があり、この乖離はStudy53-73型のマイクロ研究では埋まらない。** 検証済みの現実: IS 12.37% / OOS 13.48% / Full期間 11.35% / Calmar IS 0.683（Study52、integrity fix後）。サバイバルバイアス-1〜3ppを引けば実力は**素の10〜12%**。目的への唯一の実証候補経路（資本¥20-30M+レバ1.3x → 26-30%、Roadmap 2026-06-13）は、**全て2026-06-28 integrity fix前の数値に立脚しており、現在は未検証仮説に格下げされている**。

## 致命的問題（批判的評価）

1. **研究とゴールの断絶**: Study53以降の21研究（6日間）でProduction採用ゼロ。全てExit/BW/Add-onのマイクロ最適化と監査の再帰。結論は一貫して「天井が低い」（Study63: 理論天井+1.63pp<3pp、Study69: STOP、Study70: REJECT）。**+1〜2pp級の研究を積んでも30%には構造的に届かない**のに、リソースの9割がそこに投下された。
2. **30%経路の証拠が汚染懸念**: Study42「天井24.15%」/Study43A「Sweet ¥20-30M」/Study46「+6.07pp」は全てintersectionバグ修正前（06-26/27実行）。Study50はStudy46の+6.07ppを「バグ+corona除外の産物」と公式否定済み。**資本スケーリング系だけ再検証を免れている**。
3. **監査ループの再帰**: Study71→72→73は「監査の監査」。Study52キャッシュ汚染の発見は価値があったが、プロセス欠陥（fresh run必須化）を直せば1研究で済んだ。
4. **Calmar1.5の未達構造**: Calmar1.5 @ CAGR30% はMaxDD≤20%を要求。現行DD-18.12%は資本スケールで%不変（Study43A）だが、レバ1.3xでDD≈23%に拡大 → **レバ経路はCalmar目標と直接衝突**。DD構造の69%はアルファ源泉（動的ユニバース）と表裏一体で除去不能（Study34）。**CAGR30%とCalmar1.5は現構成では同時達成不能**であり、これを明言した研究が一本もない。

## 意思決定

### ① 今後絶対にやるべき研究（この4本以外は原則凍結）

| # | 研究 | 理由 |
|---|---|---|
| 1 | **Study74: 資本スケーリング再検証（fix後エンジン・¥3M/10M/20M/30M・D_ATR_EQ&RSR75両構成）** | 30%目標の成立可否を決める唯一の研究。白なら目標続行、黒なら目標改定 |
| 2 | **Study75: Survivorship-free検証**（上場廃止込みユニバース） | 全数値の信頼区間を確定。-3ppなら全戦略判断の前提が動く |
| 3 | **CAND_B（rsr_exit 70→75）移行**（研究済・決断のみ） | WF5/5・2022年+5.02pp・Bootstrap P(>0)=100%。将来レバを掛けるならworst-year正転は前提条件 |
| 4 | **Quality Replacement Phase9判定**（shadow稼働中・自動発火） | WF5/5・Calmar+0.075検証済み。追加コストほぼゼロ |

### ② やってはいけない研究（証拠付き閉鎖）

| 領域 | 閉鎖根拠 |
|---|---|
| Exit micro最適化 | Study40 EXHAUSTED宣言、Study61-69チェーン全てSTOP/REJECT |
| BigWinner検出・保護 | Study69 STOP（Oracle比11.9%<30%、n不足） |
| 新規MLシグナル/特徴量 | Study60 IC天井0.06-0.14、Study61安定特徴量ゼロ、Study62 FalseHero検出不能 |
| ポートフォリオ幾何/配分 | Study25/27/28 EXHAUSTED（oracle確認、天井+0.069<0.10） |
| Adaptive CAP/レジームsizing | Study8系列全REJECT、MSW REJECT、regime_2 WF失敗 |
| Add-on拡張 | Study70 REJECT（純ドラッグ） |
| Entryフィルター/タイミング | Study24系列終了、ET score30 REJECT（WF1/5）、Entry Velocity全REJECT |
| **監査の監査（Study71-73型再帰）** | プロセス修正（fresh run必須）で代替 |

### ③ Production維持か変更か

**維持、ただし2点変更を決定**（実行はASK_FIRSTでユーザー承認）:
1. **rsr_exit 70→75採用（CAND_B）**。平均-2ppの代償はあるが、①70の採用根拠はINVALID確定（Study72）②目標30%はrsr_exitでは決まらない（寄与±2pp）③長期維持・将来レバの前提はworst-year生存 — で正当化。
2. **Addon執行価格の整合PATCH**（翌日close→翌日open）。BT/Live parityは実弾の前提。

### ④ 残りロードマップ

```
Phase A（〜2週）: Study74資本再検証 + CAND_B移行 + Addon PATCH
Phase B（〜1ヶ月）: Study75 survivorship / QR Phase9判定（8月中旬）
Phase C（条件付き）: Study74白 → 入金計画¥20M+レバ1.3x設計（capital_scaling層は実装済み）
             Study74黒 → 目標を「CAGR15-20%・Calmar1.2」に公式改定し研究縮小・運用フェーズ移行
Kill criteria: Phase A/Bで新規採用ゼロが続く場合、研究は月次メンテのみに縮退
```

### ⑤ ROI順ランキング

| 順位 | 施策 | 期待値 | コスト |
|---|---|---|---|
| 1 | Study74 資本再検証 | **目標成立可否そのもの**（±15pp級の意思決定情報） | 中 |
| 2 | CAND_B移行 | 2022年+5.02pp・WF5/5・レバ前提確立 | 極小（決断のみ） |
| 3 | Study75 survivorship | 全数値の±1-3pp信頼性 | 中 |
| 4 | QR Phase9判定 | Calmar+0.075 | ほぼゼロ（自動発火） |
| 5 | Addon PATCH | parity（±0.3pp実測要） | 小 |
| 6 | MC/感度sweep | 頑健性の穴埋め（防御的） | 中 |
| 7以下 | ②の全領域 | **負ROI確定** | — |

## 最終決定 — 責任者として次にやる一つ

**Study74: integrity fix後エンジンでの資本スケーリング再検証。**

これが唯一「目的関数を動かせる」研究である。rsr_exitもaddonも±2ppの話であり、30%との18pp差を埋める候補は資本×レバ経路しか存在しない。そしてその経路の証拠は全てバグ修正前のもの — **現在この目標は検証済み根拠を一つも持っていない**。Study74が白なら入金とレバ設計に進み、黒なら目標を改定して運用フェーズに移る。どちらに転んでも研究プログラムの出口が確定する。

実行にはスクリプト新規作成（ASK_FIRST）が必要。
