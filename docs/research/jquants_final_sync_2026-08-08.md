# J-Quants解約前（8/9）最終差分取得 2026-08-08

## 0. 前提

- J-Quants契約は2026-08-09解約予定（本セッション実行時点で解約前日）。
- 前回同期基準日: 2026-07-31（`database/market/README.md`旧記載）。
- 本セッションの目的: 前回同期以降に取得可能な全データの確保・新規API 2件の実装可否確認・
  既知の重大バグ（ISSUE-001）の修正・全件整合性検証。

## 1. 実行内容サマリ

| # | 内容 | 実行方式 |
|---|---|---|
| 1 | OHLCV/master/universe/classifications 差分更新 | `src/database/sync.py`（既存・未取得営業日のみ） |
| 2 | 分足/Tick/財務summary/信用/空売り/投資部門別（当月+前月） | `src/database/sync_bulk.py sync_all()`（既存・既にdedupe-on-key安全設計） |
| 3 | TOPIX/TOPIX-17/その他指数（26系列） | **新規実装**の安全差分マージ版（ISSUE-001対策） |
| 4 | ETF派生ビュー再構築 | `src/database/etf.py`（追加API呼び出しなし・ohlcv/companies派生） |
| 5 | 決算発表予定日API | **新規実装** `src/database/earnings_calendar.py` |
| 6 | 財務追加項目（ShEq/ROE/NCShEq/NCROE） | 実データ確認 → 全期間force再取得で遡及統合 |
| 7 | bulk_catalog／metadata／snapshot再生成 | 既存スクリプト一式 |
| 8 | 整合性検証 | `verify_database.py`（通常 + `--full-archive-check`） |

## 2. データセット別: 追加期間・追加行数・date_max（同期前後比較）

| データセット | 行数(同期前) | 行数(同期後) | 追加行数 | date_max(前) | date_max(後) |
|---|---:|---:|---:|---|---|
| ohlcv | 10,151,606 | 10,173,833 | +22,227 | 2026-07-31 | 2026-08-07 |
| minute | 224,493,045 | 234,936,073 | +10,443,028 | period=202606 | period=202607(ok)+202608(partial) |
| tick | 1,885,799,082 | 1,997,970,366 | +112,171,284 | period=202606 | period=202607(ok)+202608(partial) |
| fundamentals_summary | 192,064 | 193,583 | +1,519 | 2026-07-31 | 2026-08-07 |
| margin_interest | 2,084,004 | 2,088,264 | +4,260 | 2026-07-24 | 2026-07-31 |
| margin_alert | 448,514 | 449,606 | +1,092 | 2026-07-31 | 2026-08-07 |
| short_ratio | 83,708 | 83,878 | +170 | 2026-07-31 | 2026-08-07 |
| short_sale_report | 1,405,986 | 1,411,507 | +5,521 | 2026-07-31 | 2026-08-07 |
| investor_types | 2,402 | 2,406 | +4 | 2026-07-30 | 2026-08-06 |
| etf_prices | 643,968 | 646,353 | +2,385 | 2026-07-31 | 2026-08-07 |
| index_prices（26系列合計） | 56,046 | 56,216 | +170 | 2026-07-31 | 2026-08-07 |
| 決算発表予定日（新規） | 0 | 186 | +186 | — | 2026-08-08（翌営業日分スナップショット1件のみ） |

補足:
- `margin_interest`の2026-08分は`status=empty`（週末残高データのため8/3-8/7の日次live файлには該当日がなかった。仕様通り・欠損ではない）。
- `minute`/`tick`/`fundamentals_summary`/`markets/*`の2026-08分は`status=partial`（月内進行中・仕様通り。9月以降のロールアップはJ-Quants解約により発生しない）。

## 3. 財務情報API追加項目（ShEq/ROE/NCShEq/NCROE）

- 公式仕様（`/v2/fins/summary`）上の正式フィールドコードであり、**Free+プランで取得可能**
  （`/fins/details(statements)` = Premium限定 とは別物）。
- 2026-08-08時点の実データで初めて非NULL値が確認された（従来保存していた列一覧・103列には
  含まれていなかった）。
- 過去分（テスト実施: 2020-01）を`force=True`で再取得したところ、**過去の開示についても遡及的に
  値が返る**ことを確認（例: 2020-01分でShEq非NULL 1,146/1,385件）。
- そのため2016-07〜2026-08の**全122か月をforce再取得**し、既存Parquetへ列追加の形で統合済み
  （`pd.concat`のouter結合により、既存行は新規4列がNaN→今回の再取得で実値に更新。他の既存列・
  既存行数は無変更）。
- 結果: 全11年分parquetでdShEq/NCShEq/ROE/NCROEが非NULL値を持つことを確認（ROE/NCROEは
  開示書類種別により母数が少ないが、これはJ-Quants側の開示仕様——特定のDocTypeのみ算出される
  ——であり取得漏れではない）。

## 4. 新API実装状況

| API | 状態 | 備考 |
|---|---|---|
| `/equities/earnings-calendar`（決算発表予定日） | **実装・取得済み** | Bulk非対応・REST専用・Free+。前方参照専用（翌営業日分のみ返却・REIT除外）のため、解約前に取得できるのは実行時点1スナップショットが限界（過去日は提供されない仕様）。`database/market/fundamentals/earnings_calendar/2026-08-08.parquet`（186行・列: Date/Code/CoName/FY/SectorNm/FQ/Section）。 |
| `/fins/summary` 追加項目 ShEq/NCShEq/ROE/NCROE | **取得済み・全期間統合済み** | 上記§3参照。コード変更なし（既存のpass-through設計により自動反映）。 |

## 5. 取得できなかったもの（契約上の制約・変更なし）

`database/market/metadata/source_catalog.json`より（2026-07-23時点から変化なし）:

- `/fins/details`（財務諸表本表 BS/PL/CF） — Premium限定
- `/fins/dividend`（配当金情報） — Premium限定
- `/equities/bars/daily/am`（前場四本値） — Premium限定・直近のみ
- `/markets/breakdown`（売買内訳データ） — Premium限定
- `/derivatives/bars/daily/futures,/options`（先物/オプション全般） — Premium限定
- `/derivatives/bars/daily/options/225`（日経225オプション） — Standard+だが未実装（今回スコープ外・時間的優先度により見送り）
- `/edinet/major-shareholders,cross-shareholdings,large-volume-shareholders`（EDINET系） — Standard+だが未実装（同上）
- `/markets/calendar`（取引カレンダー） — `src/market/jpx_calendar.py`で代替済みのため未実装のまま据え置き
- `/td/list,/td/files,/td/bulk`（TDnet適時開示） — 別Add-on・未契約
- 日経225指数そのもの — J-Quants非提供（Nikkei Inc独自ライセンス）
- ETF構成銘柄 — J-Quants非提供

## 6. ISSUE-001修正（index_prices.py 上書きバグ）

`database/market/KNOWN_ISSUES.md`参照。`fetch_and_save_index_safe()`他を新設し、
`python -m src.database.index_prices --safe --other --end 2026-08-07`で26系列すべてを
安全差分マージ方式で更新。全系列で`existing_rows <= combined_rows`・date_min後退ゼロを確認
（fail-closedのshrinkエラーは一度も発生せず）。旧関数は後方互換のため残置するが新規呼び出し禁止。

## 7. verify_database.py 結果

通常実行・`--full-archive-check`（archive/bulk全件SHA256再計算）とも:

```json
{
  "bulk_state_file_existence": [],
  "archive_checksum_mismatches": [],
  "schema_conformance": [],
  "duplicates": [],
  "pass": true
}
```

**PASS**（`database/market/metadata/verify_report_2026-08-08.json`）。

## 8. データ消失が起きていないことの証拠

同期実行前後で全11ドメインの行数・date_min・date_maxをスナップショット比較（§2の表がその実測値）。
全データセットで:

- 行数: 単調増加のみ（減少ゼロ）
- date_min: 全ドメインで不変（後退ゼロ）
- date_max: 更新対象ドメインすべてで前進（未更新分は不変）
- symbol_count: 減少ゼロ

index_prices（過去に実際の事故が発生した領域・ISSUE-001）についても、
`fetch_and_save_index_safe()`内蔵のfail-closedチェック（shrink時は例外送出・書き込み拒否）を
26系列全てで通過し、かつ独立スナップショット比較（§2）でも裏付け取得済み。

## 9. 残課題・今後の判断が必要な事項

- 日経225オプション・EDINET系（大株主/政策保有/大量保有報告）はStandard+で取得可能だが
  今回未実装（時間的制約により優先度を下げた）。解約後は再取得不可能になる点、ユーザーに
  改めて確認が必要。
- `data/jquants/`（Legacy・study75_downloader.py系）は本セッションのスコープ外
  （`database/market/`とは独立経路・plan「確定方針8」により無関係）。
