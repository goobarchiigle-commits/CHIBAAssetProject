---
strategy: PEAD Practical
version: "1.0"
status: Frozen
verdict: UNTESTED
role: Production Candidate
parent: pead_classic_v1.0
derived_from: [Brandt2008(EAR), CJL1996(ABR), Study82Phase0(EarnForecastRevision実在確認)]
rgp: post-earnings underreaction / drift（日本固有情報源=会社予想）
conventions: common_conventions_v1.0
created: 2026-07-24
origin: research/studies/Study082/pead_v1_spec_3proposals_2026-07-24.md（2案）
---

# PEAD-J Practical v1.0 — 日本市場実戦型（ガイダンス修正イベント）

**仮説**: 会社予想の大幅上方修正+ファンダ確認+市場初動確認の複合イベントに正のドリフトが存在する。
**親版とのdiff**: サプライズ定義をSUE型→会社予想修正+EAR型に置換（MAJOR相当の構造差だが初版のため v1.0 として独立起案・parentは系譜表示用）。SUE型とEAR型は文献上低相関（CJL: 0.115）=直交検証。

## 仕様（固定値）

| 項目 | 仕様 |
|---|---|
| Universe | 東証プライム+スタンダード（`MarketCode ∈ {0111, 0112}`）・時価総額 ≥ 50億円（`ShOutFY`×前日終値）・ADV20 ≥ 1億円・株価 ≥ 100円 |
| Event | 次のいずれか（OR）:<br>**(a)** `DocType = EarnForecastRevision` かつ ΔFOP ≥ +10%<br>**(b)** 決算短信同時修正: 短信の`FOP`が直前開示`FOP`比 +10%以上<br>ΔFOP = (FOP_new − FOP_prev) / \|FOP_prev\|、**FOP_prev > 0 必須** |
| Filter（全て必須） | ① Sales YoY > 0（直近実績四半期累計）② OP YoY > 0 ③ NP > 0（赤字除外）④ FEPS > 0 ⑤ Eq > 0（債務超過除外） |
| 初動確認（EAR型） | T0+1寄付ギャップ: `Open(T0+1)/Close(T0) − 1` − TOPIX同ギャップ **> 0** の場合のみエントリー。live執行: 寄前気配（kabuステーションAPI板情報・08:55時点）でギャップ正確認→寄付成行。BT proxy=実際の始値ギャップ |
| Entry | T0+1営業日 寄付（ギャップ確認と同時執行） |
| Exit | エントリーから**40営業日**後 寄付（Study82 PhaseD台帳と比較可能） |
| Primary判定 | イベント群コスト後CAR(40d) vs TOPIX > 0 かつ NW-t ≥ 2.0 かつ n ≥ 300 → PASS |

## 留保（事前登録）

- (a)+(b)の年間イベント数は未実測（Phase0では54件中13件が予想修正）。**n < 300 は INCONCLUSIVE**（FAILではない）。
- (b)検出には全銘柄FOP時系列の新規構築が必要——(a)のみで開始する縮小版はユーザー決裁事項（3案仕様書§6-2）。

## 再現手順（API）

1. `/fins/summary`全収集 → `DocType=EarnForecastRevision`抽出 + 短信FOP時系列差分で(b)検出
2. Filter①-⑤をイベント時点の最新実績短信から適用（PIT: 開示済みデータのみ）
3. T0+1ギャップ計算 → 正のみ採用 → 40営業日CAR測定
