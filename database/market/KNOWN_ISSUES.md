# database/market — Known Issues

記録専用（コード修正はここでは行わない。修正は別途ASK_FIRSTで起案）。

---

## ISSUE-001: `index_prices.py` が既存Parquetを無条件上書きする（データ消失リスク）

- **発見日**: 2026-08-01（Study111 CFLM Phase1実行中に発覚）
- **深刻度**: High（J-Quants解約後は再取得不能なデータの永久消失に直結）
- **対象コード**: `src/database/index_prices.py` の `fetch_and_save_all_topix17()` および
  `fetch_and_save_other_indices()`

### 内容

```python
df = fetch_index_series(code, start, end, provider=provider)
out_path = INDEX_PRICES_DIR / f"{code}.parquet"
df.to_parquet(out_path)   # ← 既存ファイルとのマージ・追記処理が一切ない単純上書き
```

`--start`/`--end`で指定した期間のデータを取得し、そのまま`to_parquet()`で書き込む。既存ファイルに
既に保存されていた期間のデータとマージしない。**狭い`--start`（例: 直近数営業日のみ）で実行すると、
それ以前に保存されていた全履歴が消失する。**

### 実際に発生した事故

2026-07-31 20:34、`database/market/index/prices/{0000,0080-0090}.parquet`（TOPIX本体＋
TOPIX-17業種指数、計18ファイル）が2016年〜2026年の約10年分（各2,443行）から直近5営業日分
（2026-07-27〜2026-07-31）へ縮小した状態で上書きされているのを2026-08-01のStudy111実行中に発見。
他の指数ファイル（0070/0075/0500-0504/B507）は同時期に更新されておらず無事だった——被害範囲が
`fetch_and_save_all_topix17()`の対象コード（`0000`+`0080`-`0090`）と完全一致しており、狭い
`--start`でこの関数が実行されたことが直接原因と推定される。

2026-08-01、ユーザー承認の上で`python -m src.database.index_prices --start 2016-08-01 --end 2026-07-31`
により10年分を再取得し復旧（`docs/research/jquants_final_audit_2026-08-01.md` §1で検証済み）。

### なぜ重大か

- `index/prices/`はJ-Quants Bulk API対象外（通常REST経由）のため、`archive/bulk/`（CSV.GZ恒久
  保存）に一次データが存在しない。**Parquetファイルそのものが唯一のコピー。**
- J-Quants契約は解約前提で進行中（`docs/research/market_snapshot_data_source_decision_memo_2026-07-31.md`）。
  解約後は同じ事故が起きても再取得不可能——**次回同種の事故は永久データ消失になる。**
- J-Quants Standardプランの指数データは10年ローリング窓——`--start`のデフォルトを過去に固定して
  いても（現状`2016-07-01`）、日が経つにつれAPI側の実際の下限が後退するため、常にフル履歴を
  再取得できるとは限らない（2026-08-01時点で`2016-07-01`指定は既に`400 subscription covers
  2016-08-01~`で拒否された）。

### 推奨される修正方針（実装はしない・起案のみ）

1. `df.to_parquet(out_path)`の前に既存ファイルを読み込み、`pd.concat`で日付重複を除去して
   マージしてから書き込む（他のドメイン取り込みスクリプトが採用しているincremental更新パターンに
   合わせる）。
2. 上書き前に既存ファイルの行数・期間をログ出力し、新規取得データが既存より狭い場合は警告・
   確認プロンプトを出す。
3. 恒久対策として`index/prices/`もアーカイブ境界（`archive/bulk/`相当）に含めるか、独立した
   バックアップ機構を設ける。

修正は本Issueを起案根拠として、別セッションでASK_FIRSTの上コード変更を行うこと。
