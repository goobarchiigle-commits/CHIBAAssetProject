# Strategy Registry — 戦略台帳（唯一の索引）
# 規則: research/governance/strategy_specification_governance.md §6, §9, §10
# RGP定義の正典: research/governance/rgp_taxonomy_v1.0.md（Frozen・2026-07-24決裁。RGP-7/Small GrowthのみPROVISIONAL）
# 追加=Draft起案時 / Verdict更新=fresh run実測後のみ（Study52規則）
# 構造バージョン: v1.1 — RGP → Family → Specification の3階層閲覧構造（2026-07-24 Research Entry Gate導入に合わせ再編）

最終更新: 2026-07-24

## 閲覧順序（本台帳の読み方）

本台帳は必ず **RGP → Family → Specification** の順で閲覧する（governance §1の4層構造・§10 Research Entry Gateと対）。
「PEADを直すか/Qualityを直すか」ではなく「Underreactionをどう捉えるか/Qualityをどう捉えるか」で考える——RGP見出し内をFamily小見出しで細分し、各Familyの下にSpecification表を置く。

---

## RGP: Underreaction（post-earnings drift）

### Family: PEAD族（決算サプライズドリフト・SUE/ABR型）

| Strategy | Version | Role | Status | Verdict | Evidence | Parent | File |
|---|---|---|---|---|---|---|---|
| PEAD Classic | 1.0 | Calibration Benchmark | Frozen | UNTESTED（事前予測=FAIL登録） | Bernard&Thomas1989 / FOS1984 / CJL1996 | — | `pead/pead_classic_v1.0.md` |

### Family: Guidance Revision族（会社予想修正ドリフト・日本固有情報源）

| Strategy | Version | Role | Status | Verdict | Evidence | Parent | File |
|---|---|---|---|---|---|---|---|
| PEAD Practical | 1.0 | Production Candidate | Frozen | UNTESTED | Brandt2008 / CJL1996 / Study82Phase0 | PEAD Classic 1.0 | `pead/pead_practical_v1.0.md` |

### Family: Analyst Revision族（アナリスト予想改定ドリフト）

*Specification未起案*（データ調達要件=コンセンサスデータがJ-Quants Standard外。rgp_taxonomy_v1.0.md §7-4参照）。

---

## RGP: Overreaction correction（mean reversion）

### Family: Event-driven MR族（イベント起点・短期）

| Strategy | Version | Role | Status | Verdict | Evidence | Parent | File |
|---|---|---|---|---|---|---|---|
| EDMR | 1.0 | Production Candidate | Frozen | UNTESTED（事前登録効果量あり） | Study82 PhaseD / 82E / 82F | PEAD Classic 1.0 | `edmr/edmr_v1.0.md` |

### Family: Long-term Reversal族（長期反転）

*Specification未起案*。

---

## RGP: Quality premium / junk avoidance

### Family: Profitability+Accruals+F-Score複合族

| Strategy | Version | Role | Status | Verdict | Evidence | Parent | File |
|---|---|---|---|---|---|---|---|
| Quality MF Classic | 1.0 | Calibration Benchmark | Frozen | UNTESTED（データ留保: GP等要補助ソース） | NovyMarx2013 / Piotroski2000 / Sloan1996 / O'Shaughnessy | — | `quality/quality_mf_classic_v1.0.md` |
| Quality MF Practical | 1.0 | Production Candidate | Frozen | UNTESTED | FamaFrench2015(RMW) / Ball et al. / Piotroski2000 / HXZ2015 | Quality MF Classic 1.0 | `quality/quality_mf_practical_v1.0.md` |

### Family: Value×Quality族（Piotroski型・RGP-5×RGP-6交差）

| Strategy | Version | Role | Status | Verdict | Evidence | Parent | File |
|---|---|---|---|---|---|---|---|
| Quality Value SmallMid | 1.0 | Research Hypothesis | Frozen | UNTESTED（間接証拠のみ・quality直接実測なし） | Piotroski2000 + Study82E/95/99/100/101/110A | Quality MF Classic 1.0 | `quality/quality_value_smallmid_v1.0.md` |

**注**: 本Familyの正確な位置はRGP-5単独ではなくRGP-5×RGP-6交差（rgp_taxonomy_v1.0.md §6で形式化）。フォルダは`quality/`のまま（方向反転を伴わないためフォルダ分離はしない）。

---

## RGP: 未分類（Research Entry Gate Step1＝NO の候補・分類確定までSpecification起案不可）

| Strategy候補 | 現在の分類状況 | Evidence（未整理） |
|---|---|---|
| Small Growth | **Current Classification: Non-independent**（RGP-7★PROVISIONAL or RGP-5/6適用領域・rgp_taxonomy_v1.0.md §4）。全文献精査未了 | O'Neil等（未精査） |
| Turnaround | RGP-6×RGP-5交差族（Piotroski型・rgp_taxonomy_v1.0.md §4で確定判定）。**Value×Quality族への合流候補**（上記Family参照） | Piotroski2000 |

---

## 系譜（RGP内・RGP間の親子関係）

```
Underreaction
  PEAD Classic v1.0 (Calibration Benchmark・較正原点・事前予測FAIL)              [Family: PEAD族]
      └── PEAD Practical v1.0 (Production Candidate)                           [Family: Guidance Revision族]
              ← サプライズ定義を会社予想修正+EAR型に置換

Overreaction correction
  EDMR v1.0 (Production Candidate)                                             [Family: Event-driven MR族]
      ← parent=PEAD Classic・方向反転=MAJOR分岐・RGP再分類→edmr/

Quality premium / junk avoidance
  Quality MF Classic v1.0 (Calibration Benchmark・較正原点・GP/A+ACC+F-Score)   [Family: Profitability+Accruals+F-Score複合族]
      ├── Quality MF Practical v1.0 (Production Candidate)                     [Family: 同上]
      │       ← 全ファクターを/v2/fins/summary実在フィールドに置換・四半期化
      └── Quality Value SmallMid v1.0 (Research Hypothesis)                    [Family: Value×Quality族]
              ← Small/Mid×高B/M×F-Score-J（Piotroski原典回帰+実測適応）
```

## Role定義（governance §3準拠・3値固定）

| Role | 意味 |
|---|---|
| Calibration Benchmark | 文献忠実複製。勝つことを期待しない。改善Δの測定原点 |
| Production Candidate | 実装データ充足・live候補になりうる仕様 |
| Research Hypothesis | 自プロジェクト実測（間接証拠含む）ベースの仮説段階・直接検証未実施 |

## RGP越境禁止（governance §9）

各仕様書は自身のRGPに属さない条件（RSRランキング・決算イベント条件・テーマ株条件・出来高急増・ブレイクアウト等）を追加してはならない。複合検証は新規RGP戦略として別途起案（暗黙混入禁止）。

## Research Entry Gate（governance §10・要約）

新規研究依頼は必ず: **Step0 Research Questionを1文で固定** → Step1 既存RGPへ分類可能か（YES→Family へ／NO→新RGP提案=例外） → Step2 Familyは存在するか（YES→Specification作成／NO→Family追加起案） → Step3 Specificationは存在するか（YES→Step4／NO→Specification v1.0起案） → **Step4 Study起案（Study Role=Calibration/Validation/Replication/Improvement/Explorationを1つ付与・governance §11）**。詳細はgovernance §10・§11。

## 共通規約

- 全仕様書は `common_conventions_v1.0.md` を参照（データソース・執行コスト・検証ゲート）。
- 検証推奨順: EDMR → PEAD Practical → PEAD Classic（既存台帳再利用の軽量さ順）。Quality系は未着手（優先順位はユーザー決裁）。
- 実装・BT実行は全戦略とも未承認（ASK_FIRST）。
- 新RGP追加・新Family追加（Small Growth/Turnaroundの分類確定含む）はユーザー決裁（ASK_FIRST・governance §10）。
