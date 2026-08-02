# Study75 Universe Event Source比較 — Option A (listed/master) vs Option B (daily bars)

**日付**: 2026-07-10
**目的**: ASK_FIRST③（Universe復元本実行）着手前に、Universe ADD/REMOVEイベントの導出元を
listed/masterの日次ポーリングから daily bars の日次スナップショット由来に置き換えるべきかを評価する。
**実行**: ダウンロード・大規模API実行なし。既存の実測値（ASK_FIRST②スモークテスト・Download
Strategy Validation・2026-07-09/10）とアーキテクチャ分析のみに基づく。

---

## 対象

- **A（listed/master日次ポーリング）**: `universe.rebuild_universe_events()`。営業日ごとに
  `/v2/equities/master` を叩き、前営業日との差分でADD/REMOVEを判定する。
- **B（daily bars日次スナップショット由来）**: `universe.rebuild_universe_events_from_daily_bars()` /
  `rebuild_universe_events_from_staged_bars()`。`study75_downloader.py`（Strategy C）が価格取得のために
  既に取得済みの `/v2/equities/bars/daily?date=X` 応答のCode列からADD/REMOVEを導出する。

## 測定

| 項目 | A（listed/master） | B（daily bars） | 判定 |
|---|---|---|---|
| **リクエスト数**（全期間・約2,440営業日） | 専用に約2,440件 | Strategy Cと同時実行なら**限界コストゼロ**（価格取得と同じ応答を再利用）。単独実行時のみAと同等 | **B優位** |
| **完全性** | 正式な上場ステータスを取得（取引停止・出来高ゼロの日でも「上場中」を正しく捕捉） | その日「実際に取引された」コードのみ捕捉。取引停止・極端な低流動性で出来高ゼロの日は見逃す可能性 | **A優位**（ただし後述の理由で実務上の影響は限定的） |
| **短命上場の捕捉力** | 営業日単位でスナップショットするため、1営業日でも存在すれば捕捉 | 同じ営業日粒度。ただし「上場はしたが一度も取引されなかった」超短命ケースはBでは捕捉不可 | **ほぼ互角**（実際に取引されたか否かが唯一の差） |
| **再現性** | 外部APIへの再照会が必要。過去分の再構築も毎回ライブ通信に依存 | `cache/daily/day_*.parquet`（Strategy Cが既に保存済み・検証済み・不変）から**完全オフラインで決定論的に再構築可能**。外部APIの応答が将来変わっても影響を受けない | **B優位（決定的）** |

## 結論: **B が A を支配する**

- リクエスト数: Strategy C（価格ダウンロード）と統合すれば実質ゼロ。単独実行でも最大でA同等（劣化しない）。
- 完全性の差はStudy75の実際の用途（モメンタム戦略のsurvivorship-free universe構築）において
  実務上の影響が限定的: 「取引されなかった日」は取引不可能日でもあり、戦略が売買判断を行う対象になり得ない。
- 再現性はBが決定的に優位。オフライン・決定論的な再構築は、研究の再現性要件（本プロジェクトの
  検証機構の中核原則）と直接整合する。

**採用決定**: B を正本（canonical）とする。A（`rebuild_universe_events`）はレガシー・比較用として
削除せず残置（`--rebuild-universe-legacy`）。listed/master は日次スナップショットの用途から外し、
**銘柄単位・1回限りのメタデータ補完専用**（会社名・セクター・市場区分）に役割を縮小する
（`enrich_universe_reference_with_listed_info()` / `--enrich-universe`）。

## 副次的な運用上の含意

Bの採用により、**実行順序の推奨が変わる**: 従来案（Universe復元→Full Download）ではなく、
**Full Download（Strategy C）を先に（または同時に）実行し、その副産物としてUniverseイベントを
オフライン導出する**方が効率的。`--rebuild-universe`（オフライン正本）は、ステージング未済の
営業日に到達すると安全に処理を打ち切り、次回Full Downloadが進んでから再開する設計にした。

## 実装

- `src/jquants/universe.py`:
  `derive_codes_from_daily_bars()` / `rebuild_universe_events_from_daily_bars()`（ライブAPI版）/
  `rebuild_universe_events_from_staged_bars()`（完全オフライン・正本）/
  `enrich_universe_reference_with_listed_info()`（メタデータ補完専用）。
  `rebuild_universe_events()`（A）はレガシーとして残置。
- CLI: `--rebuild-universe`（B・オフライン正本）/ `--rebuild-universe-live`（B・ライブAPI）/
  `--rebuild-universe-legacy`（A）/ `--enrich-universe`。
- テスト: `TestDeriveCodesFromDailyBars` / `TestRebuildUniverseEventsFromDailyBarsLive` /
  `TestRebuildUniverseEventsFromStagedBars`（チェックポイント再開・未検証ファイルの拒否・
  ギャップでの安全停止を含む）/ `TestEnrichUniverseReferenceWithListedInfo`
  （`tests/jquants/test_jquants_infra.py`）。全てモック・ネットワーク不要。

## 未実施（本評価のスコープ外）

実際のUniverse復元・Full Downloadの実行（ASK_FIRST③/④）。本書はアーキテクチャ決定のみ。
