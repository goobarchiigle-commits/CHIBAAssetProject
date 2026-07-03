# strategy_snapshot_manifest.md — Single Source for Snapshot Archaeology Backtest

作成日: 2026-06-25
方式: コードベース直接抽出 + 既存研究ドキュメント引用のみ。推測禁止。
対象: S1=2026-03-15 / S2=2026-04-15 / S3=2026-05-15 / S4=2026-06-15 / S5=2026-06-25(Current)

## 0. 表記規約（必読）

- **CONFIRMED**: file:line または日付明記のドキュメントで直接確認。
- **LIKELY**: 確認済みの変更日の前後関係から推論可能（変更日が確認済みで、スナップショット日がその前/後にあることが確定している）。
- **UNKNOWN(carry-forward)**: 当該スナップショット日時点の直接証拠なし。直前の CONFIRMED/LIKELY 状態を実行用デフォルトとして引き継ぐ（これは推測ではなく「最後に確認された状態を変更未確認の間維持する」という明示的な方法論。実行可能性のため必須）。
- **N/A**: 概念自体が当時のアーキテクチャに存在しない。

S1 は S2-S5 と**別アーキテクチャ**（TEMPORAL24 + top_k ランク方式、`src/backtest/topk_live_equivalent.py`）。S2-S5 は統一 RSR42 + 動的ユニバースアーキテクチャ（`src/backtest/composite_alpha_bt.py`）。両エンジンとも同一コスト構造（slippage=0.1%/commission=0.055%）・同一データソース（yfinance経由 `download_universe`）を使用するが、シグナル生成ロジック自体が異なる（これは推測ではなく、当時実際に異なるアーキテクチャが稼働していたという研究記録上の事実: `research_state.md:1482-1486`）。

---

## 1. Universe

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| TEMPORAL24, top_k=4 ランク選出（min_rsr=0.0で閾値無効化）— **CONFIRMED** `research_state.md:1482-1486`, `topk_live_equivalent.py` docstring | RSR42 + dyn_rsr42_bear_rs0（Bull Top30/Bear Top20 rs>0）— **CONFIRMED** 採用日2026-04-05, `strategy.yaml:98-117` | 同S2 — **CONFIRMED**（変更記録なし） | 同S2 — **CONFIRMED** | 同S2 — **CONFIRMED** |

## 2. turtle_entry

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| 20d — **CONFIRMED** | 20d — **CONFIRMED** | 20d | 20d | 20d |

(変更記録なし。全期間で唯一不変のパラメータ)

## 3. turtle_exit

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| **20d** — CONFIRMED（03-31変更前。`strategy.yaml:12`コメント「20→55に変更2026-03-31」） | 55d — CONFIRMED | 55d | 55d | 55d |

## 4. min_rsr (entry)

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| **0.0**（top_kランクで代替、閾値無効）— CONFIRMED `topk_live_equivalent.py:18` | 75.0 — CONFIRMED | 75.0 | 75.0 | 75.0 (PARAMS_LOCKED) |

## 5. rsr_exit（メカニズム）

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| N/A（topk engineはRANK_EXIT/TIME_STOP/STRATEGY_EXITのみ、RSR exit閾値概念なし） — CONFIRMED コード読解 | **z-score exit_1のみ（RSR_Z<1.1）、絶対閾値メカニズム自体が未導入** — CONFIRMED `strategy_review_2026-04-13.md` §2.3「RSR-z<1.1」 | 同S2（UNKNOWN(carry-forward); 06-05に絶対閾値追加と確認、05-15はその前） | **z-score多層(4条件OR) + 絶対閾値70.0のOR結合** — CONFIRMED 絶対閾値追加日=2026-06-05 (`strategy.yaml:9`), 多層4条件は06-24時点で確認済みコードに存在(`fujiko_strategy_spec_v202606.md`§4注記1) | 同S4 — CONFIRMED（06-24のATR/MTF除去では変更されていない、`signal_bridge.py`差分確認済み） |

## 6. trailing_stop

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| N/A（topk engineに該当機構なし） — CONFIRMED | パラメータ存在はCONFIRMED（**固定 -2.5% from HWM**, `trail_stop=0.025`, `strategy_review_2026-04-13.md`§2.3/§8）だが**実行時は無効化**。理由: 機械的再現（日次HWM-2.5%判定、turtle/RSRモメンタムより低優先）を実装し検証した結果、実測トリガー率76%が04-13レポート実績(219トレード中trailing起因 ほぼ0件、turtle105+RSRモメンタム110が主要)と著しく不整合。当時の正確なコード実体は非保存(gitなし)のため、推測実装での代替を回避し**無効化**を選択（推測禁止ルール優先）。 | 同S2（無効化） | **ATR-based 3.0×ATR20**（`signal_bridge.py:2064-2082`） — LIKELY。切替日自体はUNKNOWN | 同S4 — CONFIRMED |

## 7. MTF (filter)

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| 存在せず（実装前） — CONFIRMED 実装日=2026-03-24 (`research_state.md:2216`) | **ON** — CONFIRMED | ON | ON | **OFF（除去済み 2026-06-24 20:50）** — CONFIRMED `signal_bridge.py:1634,2085` + mtime |

## 8. max_positions

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| 4（=top_k, 別アーキテクチャの定義） — CONFIRMED | 3 — CONFIRMED（PARAMS_LOCKED, 違反は04-18開始) | **4 — PARAMS_LOCKED違反** CONFIRMED（変更日04-18, `research_state.md:925-926`） | 3 in config だが **CDOS実行時オーバーライドで実質4-5枠** — CONFIRMED config値, LIKELY実効値（`2026-06-11.md:33`「Full Audit」、修正日は06-24仕様書で「解消済み」と確認のみ、修正の正確な日付はUNKNOWN） | 3、トリプルクランプ修正済み — CONFIRMED `fujiko_strategy_spec_v202606.md` §8 |

**注**: S4 は max_positions=3（config通り）と max_positions=5（CDOS override上限相当）の **2バリアントを実行**し、影響を上下に挟む（後述 §runner）。

## 9. sizing

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| topk_live_equivalent固有（cash/effective_slots, capped by max_single_weight） — CONFIRMED コード読解 | **"existing"**（cash÷残候補数の動的分配） — CONFIRMED `april13_baseline_identification_and_rollback_plan.md:91-93` | UNKNOWN(carry-forward)。ATR導入の最早証拠は06-03のため05-15はexisting継続と推定 | **ATR risk-based**（capital×1.25%÷ATR20, min(qty_risk,qty_cap)） — CONFIRMED 06-11時点で稼働確認(`2026-06-11.md:32`), 導入日はUNKNOWN(範囲: 05-31〜06-03) | **"existing"に復帰**（06-24 20:50パッチ） — CONFIRMED `signal_bridge.py:3312`コメント「4/13仕様」 |

## 10. sector gate

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| なし — CONFIRMED | **ON**（同一セクター1銘柄/35%上限） — CONFIRMED 採用日04-09 (`strategy.yaml:76-81`) | ON | ON | ON |

## 11. shock exit

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| なし — CONFIRMED（topk engineに当該ロジックなし） | **composite**（TOPIX≤-5% AND 個別≤-8%） — CONFIRMED 採用日04-05 | composite | composite | composite |

## 12. bear filter

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| なし — CONFIRMED | **ON**（7セクター除外） — CONFIRMED 採用日04-07 | ON | ON | ON |

## 13. addon (Winner Confirmation / Continuation Boost)

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| なし | なし — CONFIRMED（04-13レポートに記載なし） | UNKNOWN(carry-forward=OFF)。導入日の直接証拠なし、05-31監査で初めて「使用中」と確認 | **ON**（1.15x Continuation Boost） — CONFIRMED `strategy_review_2026-05-31.md` §1.2 | ON |

## 14. entry timing

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| なし | なし | なし — CONFIRMED（導入日05-31、05-15はその前） | **ON**（boost_weight=0.06） — CONFIRMED 導入日2026-05-31 (`strategy.yaml:212-216`) | ON |

## 15. capital（参考、固定条件外）

| S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| 実際値¥2,000,000（資金注入前） — CONFIRMED `topk_live_equivalent.py:63`, 実口座統一は03-30 | ¥3,000,000 — CONFIRMED | ¥3,000,000 | ¥3,000,000 | ¥3,000,000 |

> 本タスクの比較では全スナップショットを capital=¥3,000,000 に正規化する（ユーザー指定の固定条件）。S1の実歴史値2,000,000はここに記録するのみで、比較実行では使用しない。

---

## 既知の限界（推測ではなく構造的限界の明示）

1. S1とS2-S5は別エンジンであり、「同一バックテストエンジンバージョン」要件は **コスト構造・データソースのみ** で満たされ、シグナル生成ロジックの実装は意図的に異なる（archaeology上の事実）。
2. UNKNOWN(carry-forward) 項目は実行可能性のため直前確定状態を採用しているが、真の値が異なる可能性を排除しない。該当: S3のtrailing_stop/sizing/rsr_exit機構、S3のaddon、S4のtrailing_stop切替正確日、S4のCDOS修正正確日。
3. S4の max_positions 実効値はCDOS override分を上限バリアント(5)で挟むのみであり、実際の動的範囲（4 or 5、発生頻度）は再現していない。
