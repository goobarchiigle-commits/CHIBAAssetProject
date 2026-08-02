# Study82 Phase0 — 決算発表日時精度監査 結果報告（実データ・小サンプル）

**日付**: 2026-07-20
**性格**: 実行結果（実データ検証・小サンプル・監査のみ）。`study82_phase0_design.md`v1.3 §2-3準拠。
**目的（厳守）**: `Can PEAD be researched without look-ahead bias or timestamp leakage?`——「PEADは機能するか」ではない。アルファ・イベントリターン・サプライズ率は**一切計算していない**。
**スクリプト**: `src/scripts/study82_phase0_audit.py` / `src/jquants/provider.py::get_fins_summary()`（新規実装）
**出力**: `backtests/study82_phase0_raw_sample_2026-07-20.json`（83件生データ）/ `backtests/study82_phase0_audit_2026-07-20.json`（監査結果）

---

## 最終判定

```
PASS
```

---

## 1. サンプル概要

| 区分 | コード | 期間指定 | 実取得件数 | 実データ範囲 |
|---|---|---|---|---|
| 流動性大型株 | 86970（1コードで50件到達のため打ち切り） | 2025-01-01〜2025-12-31 | 54件 | **2016-07-28〜2026-04-28**（下記§4参照） |
| 廃止銘柄プローブ | 44490（`data/jquants/metadata/universe_events.parquet`実測REMOVEイベントより抽出・削除日2026-06-29） | 2024-01-01〜2026-06-01 | 29件 | 2019-11-14〜2026-05-13 |
| **合計** | | | **83件** | — |

エンドポイント: `/v2/fins/summary`（Phase0.1で存在確認済み・本審査で疎通成功を実測確認）。

---

## 2. Audit1-6 結果

| Audit | 項目 | 判定 | 根拠 |
|---|---|---|---|
| **Audit1** | Endpoint availability | **PASS** | `/v2/fins/summary`が200を返し83件取得 |
| **Audit2** | DiscDate / DiscTime existence | **PASS** | 両フィールドとも実データに存在（`DiscDate`="2016-07-28"型・`DiscTime`="12:00"/"15:07:00"等） |
| **Audit3** | Missing ratio | **PASS** | DiscDate欠損率0.0%・DiscTime欠損率0.0%（n=83） |
| **Audit4** | Correction disclosure existence | **PASS** | `DocType`+`DiscNo`フィールドが存在し訂正/修正区別が構造上可能。**実データにも修正開示が実在**（54件中13件が`EarnForecastRevision`＝業績予想修正、1件が`DividendForecastRevision`＝配当予想修正）。詳細は§5の留保参照 |
| **Audit5** | Leakage possibility | **PASS** | `DiscTime`は定型値ではない（12種の異なる時刻値・08:30〜15:38に分布）。場中判定（9:00-15:00）50件・引け後/前場判定33件 — 場中/引後を機械的に判別可能 |
| **Audit6** | Delisted stock coverage | **PASS** | 実在の廃止銘柄（コード44490・2026-06-29上場廃止）で29件の決算データを取得。最終開示日2026-05-13は上場廃止の約1.5ヶ月前で自然な推移 |

**総合判定の機械算出根拠**（`study82_phase0_design.md`§3準拠）: PASS/FAIL判定はAudit1・Audit2（DiscDate/DiscTime実在）・Audit5・Audit6の4項目のANDで決定。Audit3・Audit4は付随所見（単独ではFAILとしない設計）——今回は4項目全てPASSかつAudit3・4も良好だったため、総合PASSに疑義なし。

---

## 3. 実フィールド名（v1ドキュメントとの乖離・重要）

Web公式ドキュメント（`jpx-jquants.com`）調査時点（Phase0.1）で想定していたフィールド名`TypeOfDocument`/`DisclosureNumber`は**実在しなかった**。実際のv2 APIは略記名`DocType`/`DiscNo`を使用——他エンドポイント（`daily_quotes`のO/H/L/C等）と同型の「v1想定ドキュメント↔v2実装」の乖離パターンが本エンドポイントでも再現した（`research_state.md` 2026-07-10ログの先例と整合）。当初の候補リストでこの乖離を検知できず`Audit4=FAIL`と誤判定したため、実データのフィールド名を確認しスクリプトを修正・再実行して確定判定に至った（本監査プロセス自体がAudit4の意義を実証する経過になった）。

---

## 4. 留保事項・未解決の論点（正直に報告）

1. **`from`/`to`パラメータが本エンドポイントで機能していない可能性**: `date_from=2025-01-01`/`date_to=2025-12-31`を指定したが、実際に返ったレコードは2016-07-28〜2026-04-28の**約10年分全期間**だった。`/v2/equities/bars/daily`では同名パラメータが実測で機能する（provider.py既存コメント参照）が、`/fins/summary`では無視されている可能性がある。**Phase0の判定（leakage/PIT安全性）には影響しない**（全期間データが返っても各レコード自体のDiscDate/DiscTime精度は変わらない）が、将来Study82 Alpha Study（Phase D）で大規模データ収集する際は日付フィルタが機能しない前提で設計する必要がある——1コード=全履歴が返る仕様なら、date範囲指定ではなくコード単位での取得+事後フィルタが必要になる。
2. **Audit4の留保**: 観測された修正開示（`EarnForecastRevision`/`DividendForecastRevision`）は「業績予想の修正」であり、「確定した決算数値そのものの訂正（restatement）」とは異なる概念。後者（一度確定発表した決算を後日訂正するケース）が本エンドポイントでどう表現されるかは、今回の小サンプル（83件）には該当レコードが含まれておらず**未確認のまま**。Phase D（アルファ実測）で件数が増えた際に再確認が望ましい（新規audit不要・観測ベースの継続監視で足りる）。
3. **時刻表記の不統一**: `DiscTime`のサンプル値に`"12:00"`と`"12:00:00"`の両表記が混在（意味は同じだが文字列長が異なる）。将来のパース処理でこの表記ゆれに対応する必要がある（軽微・監査結果には影響しない）。
4. **Audit6は単一銘柄プローブ**: 廃止銘柄カバレッジは1コードのみの確認（母集団規模の欠落率は未測定）。Phase Dで廃止銘柄を含むサンプルを拡大する際、より広いプローブでの再確認が望ましい。

---

## 5. 禁止事項の遵守確認

本監査では以下を一切実施していない（`study82_phase0_design.md`目的節・`roadmap_v15_governance_layer.md`§8A-7と整合）:
```
実施していない: alpha計算・PEADバックテスト・イベントリターン測定・
                ポートフォリオ最適化・Study83実装・Route順位議論・仮定調整
```
取得した業績数値（Sales/OP/NP/EPS等）はraw JSONに含まれるが、これはエンドポイントの標準レスポンス構造をそのまま保存したもの（raw/processed分離原則）であり、本レポートでの分析対象には一切していない。

---

## 6. 決定木の適用（`roadmap_v15_governance_layer.md`§8A-1A・タスク指定）

```
PASS
  ↓
Study103 rerun unnecessary.
Proceed only to Study83 Proposal.
```

Study82は完了（PASS）。§8A-2 Research Freeze Ruleの解除条件（Study82完了）を満たした。
次段階は`study83_proposal.md`（既存・Proposal onlyの起案書）——**Study83実装への着手は本レポートの範囲外であり別途ユーザー判断が必要**。

---

## Sources（Phase0.1で参照した公式ドキュメント・再掲）
- [財務情報(/fins/summary)](https://jpx-jquants.com/ja/spec/fin-summary)
- [契約ごとに利用可能なAPIとデータ格納期間](https://jpx-jquants.com/ja/spec/data-spec)

*作成: CLD (Fable 5)・2026-07-20。実データ検証実施済み（小サンプルn=83・API疎通確認済み）。*
