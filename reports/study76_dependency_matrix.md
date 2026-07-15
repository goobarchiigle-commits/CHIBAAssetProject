# Study76 依存関係マトリクス

**作成日**: 2026-07-04
**性格**: Study76実行に関わる依存関係の一覧化のみ。新規Backtest・コード変更ゼロ。
**正典**: `reports/complete_execution_roadmap_2026-07-04.md` §Section4/8、`reports/final_architecture_review_2026-07-03.md` Task7。

---

## 1. 依存関係全体図

```
Study75（J-Quants survivorship-free universe）
   │  [BLOCKING] 規則ユニバース(構成銘柄・期間・バイアス幅)
   ▼
Study76（Clenow純正ベンチマークWF）
   │  [BLOCKING] 76の勝者構成（成功時のみ77が成立）
   ▼
Study77（Exit構造置換WF・3アーム）
```

並行関係: Study74（資本スケーリング）は完了済み（CP1=BLACK確定・ユーザー決裁待ち）でありStudy76とは変数分離（本書スコープ外）。Study78/80A/81は完了済みでStudy76に対しては「参照可能な既存資産」の位置づけ（新規依存ではない）。

---

## 2. 上流依存（Study76が必要とするもの）

| 依存先 | 依存内容 | 現状 | ブロッキング度 |
|---|---|---|---|
| Study75 | 規則ユニバース（TOPIX500∩流動性∩lot制約フィルター）の構成銘柄・価格データ・月次再適用ロジック | **未完了（J-Quants契約待ち）** | **Hard Block**（新規BT実行不可） |
| Study75 | Survivorshipバイアス実測値（-1.5pp〜-3pp想定） | 未完了 | Soft（Study76の合否判定には直接使わないが、D_ATR_EQ比較値の解釈に影響） |
| J-Quants API認証 | `src/.env`のJQUANTS_MAIL_ADDRESS/JQUANTS_API_PASSWORD | 未設定 | Hard Block（Study75自体のブロッカー、Study76からは間接） |
| `composite_alpha_bt.py` | shift(1)/翌日寄付/コスト計算ロジックの継承元 | ✅利用可能 | なし |
| D_ATR_EQ公式ベースライン値（RSR42ベース・旧Universe） | 比較対象数値（M1適用後） | ✅存在するが**旧Universe参考値に格下げ・比較使用禁止**（2026-07-04ユーザー決裁） | なし（判定には使わないため） |
| Study75新Universe上でのD_ATR_EQ再測定 | **確定必須**（2026-07-04ユーザー決裁: 全比較対象は同一Universeで再測定） | **未実施（Study75完了後の新規BT）** | **Hard Block**（これなしではΔCAGR算出不可） |
| ASK_FIRST承認（新規スクリプト作成） | `study76_clenow_benchmark_wf.py`相当の新規作成許可、および`D_ATR_EQ`のStudy75 Universe再測定用スクリプトの新規作成許可 | 未取得（本タスクのスコープ外） | Hard Block（実行段階） |

---

## 3. 下流依存（Study76に依存するもの）

| 依存元 | 依存内容 | 影響 |
|---|---|---|
| Study77 | Study76の勝者構成（純正構成 or 現行維持）を前提に3アームExit比較を実施 | Study76失敗時、Study77は「76勝者構成に対する」定義のまま起案不可 — 別定義への差し替えはユーザー決裁が必要 |
| Stage1 M6 DISCARD候補（turtle_exit=55/fraction.bull=0.0/entry_timing.boost_weight=0.06/vol_adj残置コード） | Study76/77決着後に処遇判断 | Study76失敗（現行維持確定）なら「DISCARD候補は現状維持」で確定。成功ならFUJIKO-R側で構造ごと不要になり同時に整理可能 |
| FUJIKO-R骨格の実在性判定 | Study76が唯一の実証機会 | 成功→shadow並走6ヶ月へ。失敗→骨格放棄・恒久閉鎖相当の扱い |
| architecture_handover.md OQ3（mom_period=21過学習疑い） | Study76/77が感度チェック先として指定済み | 純正構成側でも同様の崖がないか確認する義務あり（正典申し送り事項） |

---

## 4. 変数分離の確認（交絡防止）

Study76は「複雑性の対価」を測定する研究であり、以下の変数はStudy76の外側に固定し、Study76の結果に混入させない:

| 変数 | Study76での扱い | 理由 |
|---|---|---|
| 資本水準（Study74） | ¥3,000,000固定（PARAMS_LOCKED） | Study74は資本弾力性の研究であり別軸。混在させるとΔCAGRの帰属先が不明になる |
| rsr_exit（S1/CAND_B） | 比較対象はrsr_exit=70（CURRENT）を使用。CAND_B(75)はS1決裁未完了のため比較対象に含めない | 正典が指定する比較対象はD_ATR_EQ(CURRENT)のみ |
| **Universe（Study75⇔旧RSR42）** | **純正構成・D_ATR_EQ比較対象の両方をStudy75新Universe上でfresh run再測定（2026-07-04ユーザー決裁で確定）。旧RSR42ベースの数値との比較は禁止** | Universe差とArchitecture差の交絡を排除するための恒久統制。Study76に限らずStudy77以降にも適用 |
| mom_period=21等パラメータ感度（Study78発見・OQ3） | Study76側でも同一崖がないか確認するが、Study76自体のパラメータ（<10個）は事前固定・スイープ禁止 | 感度確認とパラメータ探索は別行為。後者は0.4ゲート違反・恒久閉鎖14項#9抵触リスク |

---

## 5. ブロッキング状態サマリ（2026-07-04時点）

| 項目 | 状態 |
|---|---|
| Study76着手可否 | **不可**（Study75規則ユニバース未生成のためHard Block） |
| Study76計画策定可否 | 可（本タスクで完了） |
| Study76新規スクリプト実装可否 | 不可（Study75完了 + Universe統制ポリシーのASK_FIRST確認 + 新規スクリプトASK_FIRST承認が前提） |
| Study77計画策定可否 | 不可（Study76結果確定が前提。本タスクではStudy77計画書は作成していない） |

**Universe統制ポリシー（2026-07-04 ユーザー決裁・確定事項）**: Study75完了時点で規則ユニバースを新基準Universeと定義し、Study76・Study77の全比較対象（D_ATR_EQ含む）は同一Universe上で再測定した値のみを使用する。旧Universe値との比較は禁止。Study75終了後、本ポリシーの適用をASK_FIRSTで確認してからStudy76へ進む（詳細→`study76_execution_plan.md`§2.4/§5）。
