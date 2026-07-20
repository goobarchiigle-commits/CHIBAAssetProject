# Study83 Proposal — 指数TSMOM 情報価値評価（起案書・実装なし）

**日付**: 2026-07-20（v1.1: §8A-2 Research Freeze Rule反映）
**★実装凍結告知（v1.1）**: `roadmap_v15_governance_layer.md`§8A-2「Research Freeze Rule」により、
**Study82完了（PASS/FAIL確定）までStudy83の実装（新規BT・新規スクリプト）は着手しない**。
本書§3で示した「データ独立のため並行着手も選択肢」は本ルールにより**上書き・凍結**。
本Proposal文書自体（改訂含む）はFreeze Rule対象外（BT・コード・データ取得を伴わないため）。
**性格**: **起案書のみ**。「まだ実装しない」——本書はStudy83実装可否をユーザーが判断するための事前評価。BT・コード・データ取得は一切実施しない。
**正典**: `roadmap_v15_governance_layer.md`§8A Phase B / `alternative_architectures_5x_2026-07-03.md`§ARCH-C / `study103_design.md`§3.4（TSMOM仮定表）。
**位置づけ**: Route Bのもう一方の柱。Study103仮定表（文献レンジからの機械配分）を実装判断可能な精度まで具体化する。

---

## 1. 目的

```
Determine: Can TSMOM improve Route B's frontier?
Output   : 実装コスト・相関見積り・Calmar改善余地・仮定差分の4点評価
```

Study82と異なりPASS/FAIL二値ではない——TSMOMはARCH-C原典で既に「弱点も含め構造が明確」（原文§46-52）なため、Phase0型の生死監査は不要。必要なのは「実装する価値があるか」の定量評価。

---

## 2. 必須出力（5点・固定）

### 2.1 実装コスト

| 項目 | 現状評価 |
|---|---|
| データ | 指数先物日足（ロール調整済）。無料圏（Stooq等）+ J-Quants指数(`/v2/indices/bars/daily`=**疎通確認済み**)で代替可能 — Study82と異なりデータ調達は低コスト |
| 執行系 | kabuステーションAPI先物対応要確認（現行`src/kabusapi/`は現物専用の可能性——執行アダプタ層の新規実装が必要か調査が要る） |
| 証拠金管理 | 現物¥3M運用と別枠管理が必要（`alternative_architectures_5x`原文「証拠金¥30-60万のレイヤー追加」）。既存`portfolio_state.json`の資産クラス設計が先物証拠金を表現できるか要確認 |
| ルール数 | 20/60/120d固定グリッドのみ（原典規定）。パラメータ探索余地が最初から極小=過学習リスクが構造的に低い |
| 総合難易度 | **低**（`alternative_architectures_5x`原文「実装難易度: 低」を維持。対象1-3銘柄・毎日1判定） |

### 2.2 Core相関見積り

Study103仮定表（`study103_design.md`§3.4）: Core-TSMOM相関 Conservative0.35/Base0.25/Optimistic0.15。根拠は`alternative_architectures_5x`比較総括表「現行との収益相関（推定）＝中」の定性評価を機械配分したのみ——**実測ゼロ**。Study83実装時は指数40年データとCore実トレード（Study78 trade_dataset）の実測相関を最優先で算出し、Study103仮定の妥当性を検証すること。

### 2.3 PEADとの相関見積り

未定義（`study103_design.md`§9C(a)でPEAD-TSMOM=Cons0.25/Base0.15/Opt0.05と設定済みだが根拠は「両者ともbull期にlongエクスポージャ」という定性推論のみ）。Study82 Phase0がFAILの場合、この相関は無意味になる（Route B構成がCore+TSMOMのみになるため）——**実装着手順序としてStudy82 Phase0の結果を待ってから本格実装するのが合理的**（Study103 Phase A→Bの順序はこの理由による）。

### 2.4 Calmar改善余地

Study103 MC結果（`reports/study103_portfolio_feasibility.md`）でBase水準の最適配分は「PEAD70%/TSMOM10%/SG20%」——**TSMOMの寄与度は現仮定では小さい（10%配分）**。Conservative水準では「MN35%/PEAD40%/TSMOM20%/SG5%」とやや厚めに配分される。これは「TSMOMが低相関・低volのため、他スリーブの信頼度が下がるほど分散価値が上がる」という自然な帰結——**実測でTSMOM単独Sharpeが仮定(0.6-1.0)より低ければ、この分散価値ごと縮小する**点に注意。

### 2.5 Study103仮定との差分（実装時に測定すべき項目）

```
仮定: Sharpe 0.6-1.0（文献） / Vol 20-25%（vol-target） / CAGR単独15-25%
実測すべき: 40年データでのSharpe・実効Vol・ロールコスト控除後CAGR・
           Bear/Bull regime別の挙動・whipsawコスト（原典弱点②③の定量化）
```

---

## 3. Proposal段階の結論（実装判断材料）

**実装推奨度: 中〜高（ただしStudy82 Phase0の結果待ちが合理的）**

理由:
- 実装コストは5案中最低（データ調達済み同然・パラメータ数最小）
- Study103 MCで既にRoute B主要構成要素として組み込まれている（Conservative水準で20%配分）
- ただし相関仮定が全て未実測のため、Study83実装の主目的は「アルファの発見」ではなく「**Study103仮定の検証**」に近い——期待情報価値は「Route B frontierがどれだけ動くか」に直結する

**実装時期（v1.1確定・§8A-2 Research Freeze Ruleにより裁量なし）**: Study82完了（PASS/FAIL確定）後のみ。~~ただしTSMOMはデータ独立のため並行着手も妨げない~~ → **この選択肢は凍結**。理由: Phase0結果によりRoute B構成そのもの（3スリーブか2スリーブか）が変わり得るため、TSMOM実装の優先度・検証設計（PEADとの相関測定を含むか否か）が変わる。データ独立性はコスト面の利点であって着手順序を早める根拠にはならない——優先順位表（v15§8A・5段固定）通りStudy82 Phase0.1→Phase0→Viability Review→Study83Proposal→Study83実装の順を厳守する。

---

## 4. 次アクション

本Proposalの承認 → Study83実装起案書（正典定義: 固定グリッド{20,60,120d}・現行との相関測定・成功条件=Sharpe≥0.8∧Core相関<0.5∧ロールコスト補正後正）→ 新規スクリプト作成は別途ASK_FIRST。

*作成: CLD (Fable 5)・2026-07-20。BT・コード変更・データ取得なし。*
