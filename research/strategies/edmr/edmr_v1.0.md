---
strategy: EDMR (Event-Driven Mean Reversion)
version: "1.0"
status: Frozen
verdict: UNTESTED
role: Production Candidate
parent: pead_classic_v1.0（方向反転=MAJOR分岐・RGP再分類）
derived_from: [Study82PhaseD, Study82E, Study82F(Case C)]
rgp: post-earnings overreaction correction / mean reversion
conventions: common_conventions_v1.0
created: 2026-07-24
origin: research/studies/Study082/pead_v1_spec_3proposals_2026-07-24.md（3案）
---

# EDMR v1.0 — Event-Driven Mean Reversion（自プロジェクト実測直系）

**仮説**: 日本Small/Mid-cap株の決算後ネガティブサプライズは過剰反応であり、0-20営業日で訂正（リバウンド）する。
**位置づけ**: research_state.md記載「Event-driven Mean Reversion独立研究（ASK_FIRST・未着手）」の起案実体。Study82F Case C確定の直接後継。

**事前登録済み効果量（実測根拠）**:
- Study82 PhaseD: Negative群40dコスト後 +1.157%（t=7.9）> Positive群 +0.619%（t=5.3）
- Study82E: 逆転はSmall/Mid集中（Size寄与1.06pp=最大）・Large-capスプレッド≈0（t=0.34）・regime不変
- Study82F: 0-20d spread −0.687%（95%CI[−0.991,−0.384]・NW-t=−4.44）→ 保有20営業日=効果最大窓

## 仕様（固定値）

| 項目 | 仕様 |
|---|---|
| Universe | Study75 PITユニバース ∩ **時価総額 50億〜1,000億円**（Large-cap除外=Study82E根拠）∩ ADV20 ≥ 1億円 ∩ 株価 ≥ 100円 |
| Event | 四半期決算短信（Classic v1.0と同一規則）で **YoY EPS変化が負**（EPS_q − EPS_{q−4} < 0）。binary符号分類は既存台帳30,952件との直接比較のため意図的維持（連続量化=v1.1候補・本版では禁止） |
| Filter | ① Eq > 0（債務超過除外）② 監理・整理銘柄除外 ③ 大赤字除外は**しない**（過剰反応仮説の対象はバッドニュース側）④ FEPS欠損可 |
| Entry | T0+1営業日 寄付 |
| Exit | エントリーから**20営業日**後 寄付（Study82F 0-20d窓） |
| 方向 | **ロングオンリー**（ネガティブサプライズ群を買う・逆張り） |
| Primary判定 | Negative群コスト後CAR(20d) vs TOPIX > 0 かつ NW-t ≥ 2.0 → PASS |
| Secondary（診断のみ） | ① Positive群との0-20dスプレッド再現（−0.687%±CI内）② Small/Mid限定でスプレッド拡大（82E予測: する）③ 61-80d再出現窓の構造（82F未解明点） |

## リスク（事前明示）

- 逆張り固有テール: 悪材料継続銘柄の下方continuation。v1.0はSLなし（タイムイグジットのみ）のためイベント単位MaxDD分布を必須記録。
- 82F効果量は**スプレッド**であり、ロング片脚のTOPIX超過が正である保証はない（PhaseD前例: Neg群絶対+でも単一スロットCAGR≈4%<TOPIX 12.76%）。Primary判定が「vs TOPIX」なのはこのため。
- 実装・BT実行はユーザー承認後（ASK_FIRST）。
