# Study82 Phase0.1 — J-Quants API疎通確認 結果（ドキュメント調査段階）

**日付**: 2026-07-20
**性格**: 実行結果（読み取り調査のみ・APIコード実行なし・データ取得なし）。`study82_phase0_design.md`§1A手順1-2に対応。
**判定**: **CONNECTABLE（暫定）** — 手順3（小サンプル疎通テスト）は新規スクリプトにつき別途ASK_FIRST。

---

## 1. 調査結果

### エンドポイントの存在確認

J-Quants APIには決算関連エンドポイントが複数存在する:

| エンドポイント | 内容 | 必要プラン |
|---|---|---|
| **`/fins/summary`** | 決算短信サマリ（発表日・**発表時刻**・業績数値） | **Standard（契約済みプラン内・過去10年）** |
| `/fins/details` | 財務諸表（BS/PL/CF）詳細 | Premium（契約プラン外） |
| `/fins/dividend` | 配当情報 | Premium（契約プラン外） |
| `/corporate_action/delisting` | 上場廃止情報 | 未確認 |

**結論**: PEAD研究に必要な最小要件（発表イベント・発表タイミング・業績サプライズ算出用の数値）は`/fins/summary`でカバーされ、これは**現行契約プラン（Standard）の範囲内**（過去10年）。`/fins/details`（Premium限定）は本研究では不要——PEADはサマリレベルの業績サプライズで十分。

### `/fins/summary`のフィールド確認（6項目監査の予備評価）

| 監査項目 | 予備所見 | 確度 |
|---|---|---|
| 1. 発表日時粒度 | `DiscDate`（開示日）+ `DiscTime`（開示時刻・サンプル表記"12:00:00"）の2フィールドが存在 | ドキュメント確認済み。**真に分単位の実測精度か（実データがHH:MM:SSで意味のある値を持つか、定型値でないか）は未検証** — Phase0本審査で実データ確認が必要 |
| 2. 場中/引後区別 | ドキュメントに明示的な区別フラグの記載なし。ただし`DiscTime`が実測時刻ならザラ場(9:00-15:00)/引後で機械的に判別可能 | 未確定（実データ検証待ち） |
| 3. 訂正開示の扱い | `TypeOfDocument`（開示書類種別）+ `DisclosureNumber`（開示番号）フィールドの存在を確認 — 書類種別による訂正判別・開示番号による個別イベント追跡が構造上可能に見える | 良好な兆候。実データでの訂正レコードの実在確認が必要 |
| 4. 配信遅延 | ドキュメントに記載なし | 未確認 |
| 5. PIT保証 | `DiscDate`/`DiscTime`により理論上は構成可能（発表時刻より前のデータのみ使用する設計は可能） | 実データでの検証が必要 |
| 6. 廃止銘柄整合性 | 別エンドポイント`/corporate_action/delisting`の存在を確認したが、`/fins/summary`が廃止銘柄の財務データを継続提供するかは未確認 | 未確認 |

## 2. Phase0.1 判定

```
CONNECTABLE（暫定）
```

根拠: (a) 決算発表日時・業績データを含むエンドポイント`/fins/summary`が実在し公式ドキュメントで確認済み、(b) 現行契約プラン（Standard）の範囲内、(c) 発表時刻フィールド（`DiscTime`）が存在し項目1の前提条件を満たす見込みが立った。

**「暫定」とする理由**: 本調査はドキュメントベースであり、実際のAPI疎通・実データ取得は未実施。`DiscTime`が真に意味のある時刻精度を持つか（定型値でないか）・訂正レコードの実在・廃止銘柄カバレッジの3点は**Phase0本審査（実データ検証）でのみ確定可能**。

## 3. 次アクション（ASK_FIRST）

```
Phase0.1手順3（小サンプル疎通テスト）を兼ねてPhase0監査を実施:
  1. src/jquants/provider.py へ /fins/summary 用のメソッド追加（新規実装）
  2. 数銘柄・直近1-2四半期のサンプル取得
  3. study82_phase0_design.md §2 の6項目監査を実データで実施
  4. PASS/FAIL/UNKNOWN確定
```

新規スクリプト作成・既存スクリプト改修に該当するため、着手には別途承認が必要。

## Sources
- [財務情報(/fins/summary)](https://jpx-jquants.com/ja/spec/fin-summary)
- [財務諸表(BS/PL/CF)(/fins/details)](https://jpx-jquants.com/ja/spec/fin-details)
- [契約ごとに利用可能なAPIとデータ格納期間](https://jpx-jquants.com/ja/spec/data-spec)
- [財務情報(/fins/statements) | J-Quants Pro（日本語）](https://jpx.gitbook.io/j-quants-pro-ja/api-reference/statements)
- [開示書類種別](https://jpx-jquants.com/ja/spec/fin-summary/typeofdocument)
- [上場廃止(/corporate_action/delisting) | J-Quants Pro（日本語）](https://jpx.gitbook.io/j-quants-pro-ja/api-reference/corporate_action/delisting)

*作成: CLD (Fable 5)・2026-07-20。読み取り調査のみ・コード実行なし・データ取得なし。*
