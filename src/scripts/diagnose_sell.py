"""
diagnose_sell.py
kabuステーション SELL 注文の診断ツール。

API の生レスポンスを raise_for_status なしで取得し、
失敗原因を特定する。DRY RUN のみ（実際に注文は発行しない）。

使い方:
    cd C:/ai-trading
    python -m src.scripts.diagnose_sell
"""
import sys
import json
import os
import requests
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

_ENV = Path(__file__).resolve().parents[2] / "src" / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=_ENV, override=False)

PORT      = int(os.getenv("KABU_API_PORT", "18080"))
BASE_URL  = f"http://localhost:{PORT}/kabusapi"
API_PASS  = os.getenv("KABU_API_PASSWORD", "")
TRADE_PASS = os.getenv("KABU_TRADE_PASSWORD", "")
ACCOUNT_TYPE = int(os.getenv("KABU_ACCOUNT_TYPE", "4"))


# ────────────────────────────────────────────────────────────────────
# 1. トークン取得
# ────────────────────────────────────────────────────────────────────
def get_token() -> str:
    resp = requests.post(f"{BASE_URL}/token", json={"APIPassword": API_PASS})
    resp.raise_for_status()
    token = resp.json()["Token"]
    print(f"[TOKEN] OK: {token[:8]}...")
    return token


# ────────────────────────────────────────────────────────────────────
# 2. 保有ポジション確認（kabu Station が認識している株を確認）
# ────────────────────────────────────────────────────────────────────
def get_positions(token: str) -> list[dict]:
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}
    resp = requests.get(f"{BASE_URL}/positions", headers=headers)
    resp.raise_for_status()
    positions = resp.json() or []
    print(f"\n[POSITIONS] kabu Station 認識ポジション: {len(positions)}件")
    for p in positions:
        print(
            f"  Symbol={p.get('Symbol')}  Exchange={p.get('Exchange')}"
            f"  LeaveQty={p.get('LeaveQty')}  Price={p.get('CurrentPrice')}"
            f"  Side={p.get('Side')}  AccountType={p.get('AccountType')}"
        )
    return positions


# ────────────────────────────────────────────────────────────────────
# 3. SELL 注文テスト（実際には発行しない — DRY RUN）
#    生レスポンスを表示して原因を特定する
# ────────────────────────────────────────────────────────────────────
SELL_CANDIDATES = [
    # (symbol_4digit, qty, exchange, description)
    ("5401", 100, 1, "5401 TSE=1"),
    ("5411", 200, 1, "5411 TSE=1"),
]

# 試すペイロードバリアント（Price / DelivType / FundType の組み合わせ）
VARIANTS = [
    # 現在の実装（Price なし）
    dict(name="A: Price欠如 DelivType=0 FundType=空白2",
         extra={}),
    # Price=0 明示
    dict(name="B: Price=0(int) DelivType=0 FundType=空白2",
         extra={"Price": 0}),
    # Price=0.0 float
    dict(name="C: Price=0.0(float) DelivType=0 FundType=空白2",
         extra={"Price": 0.0}),
    # DelivType=2 に変更
    dict(name="D: Price欠如 DelivType=2 FundType=AA (BUY系)",
         extra={"DelivType": 2, "FundType": "AA"}),
    # DelivType=1
    dict(name="E: Price欠如 DelivType=1 FundType=空白2",
         extra={"DelivType": 1}),
]


def test_sell_raw(token: str, symbol: str, qty: int, exchange: int,
                  variant: dict) -> None:
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    # ベースペイロード（Price なし）
    payload = {
        "Password":       TRADE_PASS,
        "Symbol":         symbol,
        "Exchange":       exchange,
        "SecurityType":   1,
        "Side":           "1",           # SELL
        "CashMargin":     1,             # 現物
        "DelivType":      0,
        "FundType":       "  ",          # 2 スペース
        "AccountType":    ACCOUNT_TYPE,
        "Qty":            int(qty),
        "FrontOrderType": 10,            # 成行
        "ExpireDay":      0,
    }
    # バリアントで上書き
    payload.update(variant.get("extra", {}))

    print(f"\n  --- {variant['name']} ---")
    print(f"  payload (Password masked): {json.dumps({k: ('***' if k=='Password' else v) for k, v in payload.items()})}")

    resp = requests.post(f"{BASE_URL}/sendorder", headers=headers, json=payload)
    print(f"  HTTP status: {resp.status_code}")
    try:
        body = resp.json()
        print(f"  Response body: {json.dumps(body, ensure_ascii=False)}")
    except Exception:
        print(f"  Response text: {resp.text[:500]}")

    if resp.status_code < 400:
        print(f"  ★★★ 成功候補 — OrderId: {body.get('OrderId', '?')} ★★★")
        print("  !! 実際に発注された可能性があります。注文一覧を確認してください !!")


# ────────────────────────────────────────────────────────────────────
# 4. FrontOrderType 総当たりマトリクス
# ────────────────────────────────────────────────────────────────────
ORDER_TYPE_MATRIX = [
    (10, "成行"),
    (13, "寄成前場"),
    (14, "引成前場"),
    (15, "寄成後場"),
    (16, "引成後場"),
    (17, "不成"),
]


def run_order_type_matrix(token: str, symbol: str, qty: int, exchange: int) -> None:
    """
    Price なし固定・DelivType=0・FundType='  '・Side='1' で
    FrontOrderType だけを変えて全件テストする。
    通った場合は実際に注文が入るため要注意。
    """
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    print()
    print("=" * 60)
    print("[ORDER TYPE MATRIX]")
    print(f"  Symbol={symbol}  Exchange={exchange}  Qty={qty}")
    print("  Side='1'(SELL)  CashMargin=1  DelivType=0  FundType='  '")
    print("  Price=なし（全ケース共通）")
    print()
    print("  !! 警告: HTTP 2xx が返った場合は実際に発注されます !!")
    print("  !! 直ちに kabu Station の注文一覧を確認してください !!")
    print("=" * 60)

    results = []
    for front, label in ORDER_TYPE_MATRIX:
        import time
        time.sleep(1)

        payload = {
            "Password":       TRADE_PASS,
            "Symbol":         symbol,
            "Exchange":       exchange,
            "SecurityType":   1,
            "Side":           "1",
            "CashMargin":     1,
            "DelivType":      0,
            "FundType":       "  ",
            "AccountType":    ACCOUNT_TYPE,
            "Qty":            int(qty),
            "FrontOrderType": front,
            "ExpireDay":      0,
        }
        # Price は意図的に含めない

        resp = requests.post(f"{BASE_URL}/sendorder", headers=headers, json=payload)
        status = resp.status_code
        try:
            body = resp.json()
            body_str = json.dumps(body, ensure_ascii=False)
        except Exception:
            body_str = resp.text[:300]

        success = status < 400
        marker  = "★ ACCEPTED" if success else "  rejected"
        print(f"  FrontOrderType={front:2d} ({label:6}) -> HTTP {status}  {marker}  body={body_str}")

        results.append({
            "FrontOrderType": front,
            "label":          label,
            "http_status":    status,
            "success":        success,
            "body":           body_str,
        })

    print()
    accepted = [r for r in results if r["success"]]
    if accepted:
        print(f"[MATRIX RESULT] 受理された FrontOrderType: {[r['FrontOrderType'] for r in accepted]}")
    else:
        print("[MATRIX RESULT] 全ケース拒否 — FrontOrderType 以外に根本原因あり")

    return results


# ────────────────────────────────────────────────────────────────────
# 5. CashMargin 総当たりマトリクス
# ────────────────────────────────────────────────────────────────────
CASH_MARGIN_MATRIX = [
    (1, "現物"),
    (2, "信用新規"),
    (3, "信用返済"),
]


def run_cash_margin_matrix(token: str, symbol: str, qty: int, exchange: int) -> list:
    """
    FrontOrderType=10 固定・Price なし・DelivType=0・FundType='  ' で
    CashMargin だけを変えて全件テストする。
    通った場合は実際に注文が入るため要注意。
    """
    import time
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    print()
    print("=" * 60)
    print("[CASHMARGIN MATRIX]")
    print(f"  Symbol={symbol}  Exchange={exchange}  Qty={qty}")
    print("  Side='1'(SELL)  FrontOrderType=10  DelivType=0  FundType='  '")
    print("  Price=なし（全ケース共通）")
    print()
    print("  !! 警告: HTTP 2xx が返った場合は実際に発注されます !!")
    print("  !! 直ちに kabu Station の注文一覧を確認してください !!")
    print("=" * 60)

    results = []
    for cm, label in CASH_MARGIN_MATRIX:
        time.sleep(1)

        payload = {
            "Password":       TRADE_PASS,
            "Symbol":         symbol,
            "Exchange":       exchange,
            "SecurityType":   1,
            "Side":           "1",
            "CashMargin":     cm,
            "DelivType":      0,
            "FundType":       "  ",
            "AccountType":    ACCOUNT_TYPE,
            "Qty":            int(qty),
            "FrontOrderType": 10,
            "ExpireDay":      0,
        }

        resp = requests.post(f"{BASE_URL}/sendorder", headers=headers, json=payload)
        status = resp.status_code
        try:
            body_str = json.dumps(resp.json(), ensure_ascii=False)
        except Exception:
            body_str = resp.text[:300]

        success = status < 400
        marker  = "★ ACCEPTED" if success else "  rejected"
        print(f"  CashMargin={cm} ({label:6}) -> HTTP {status}  {marker}  body={body_str}")

        results.append({
            "CashMargin":  cm,
            "label":       label,
            "http_status": status,
            "success":     success,
            "body":        body_str,
        })

    print()
    accepted = [r for r in results if r["success"]]
    if accepted:
        print(f"[MATRIX RESULT] 受理された CashMargin: {[r['CashMargin'] for r in accepted]}")
    else:
        print("[MATRIX RESULT] 全ケース拒否 — CashMargin 以外に根本原因あり")

    return results


# ────────────────────────────────────────────────────────────────────
# 6. DelivType × FundType 組み合わせマトリクス
#    FrontOrderType=10 / CashMargin=1 / Price なし 固定
# ────────────────────────────────────────────────────────────────────
DELIV_FUND_MATRIX = [
    # (DelivType, FundType,  label)
    (0,  "  ",  "DelivType=0  FundType='  '(2sp)  ← 現行"),
    (0,  "AA",  "DelivType=0  FundType='AA'"),
    (0,  "02",  "DelivType=0  FundType='02'"),
    (0,  "0",   "DelivType=0  FundType='0'"),
    (0,  "",    "DelivType=0  FundType=''(空文字)"),
    (2,  "  ",  "DelivType=2  FundType='  '(2sp)"),
    (2,  "AA",  "DelivType=2  FundType='AA'   ← BUY と同設定"),
    (1,  "  ",  "DelivType=1  FundType='  '(2sp)"),
    (1,  "AA",  "DelivType=1  FundType='AA'"),
]


def run_deliv_fund_matrix(token: str, symbol: str, qty: int, exchange: int) -> list:
    """
    DelivType と FundType の組み合わせを総当たりする。
    FrontOrderType=10 / CashMargin=1 / Price なし 固定。
    """
    import time
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    print()
    print("=" * 60)
    print("[DELIVTYPE × FUNDTYPE MATRIX]")
    print(f"  Symbol={symbol}  Exchange={exchange}  Qty={qty}")
    print("  Side='1'(SELL)  CashMargin=1  FrontOrderType=10  Price=なし")
    print()
    print("  !! 警告: HTTP 2xx が返った場合は実際に発注されます !!")
    print("  !! 直ちに kabu Station の注文一覧を確認してください !!")
    print("=" * 60)

    results = []
    for deliv, fund, label in DELIV_FUND_MATRIX:
        time.sleep(1)

        payload = {
            "Password":       TRADE_PASS,
            "Symbol":         symbol,
            "Exchange":       exchange,
            "SecurityType":   1,
            "Side":           "1",
            "CashMargin":     1,
            "DelivType":      deliv,
            "FundType":       fund,
            "AccountType":    ACCOUNT_TYPE,
            "Qty":            int(qty),
            "FrontOrderType": 10,
            "ExpireDay":      0,
        }

        resp = requests.post(f"{BASE_URL}/sendorder", headers=headers, json=payload)
        status = resp.status_code
        try:
            body_str = json.dumps(resp.json(), ensure_ascii=False)
        except Exception:
            body_str = resp.text[:300]

        success = status < 400
        marker  = "★ ACCEPTED" if success else "  rejected"
        print(f"  {label}")
        print(f"    -> HTTP {status}  {marker}  body={body_str}")

        results.append({
            "DelivType":   deliv,
            "FundType":    fund,
            "label":       label,
            "http_status": status,
            "success":     success,
            "body":        body_str,
        })

    print()
    accepted = [r for r in results if r["success"]]
    if accepted:
        print("[MATRIX RESULT] 受理された組み合わせ:")
        for r in accepted:
            print(f"  DelivType={r['DelivType']}  FundType={r['FundType']!r}  ({r['label']})")
    else:
        print("[MATRIX RESULT] 全ケース拒否 — DelivType/FundType 以外に根本原因あり")
        # エラーコードの内訳を表示
        from collections import Counter
        codes = []
        for r in results:
            try:
                codes.append(json.loads(r["body"]).get("Code", "?"))
            except Exception:
                codes.append("?")
        print(f"  エラーコード内訳: {dict(Counter(codes))}")

    return results


# ────────────────────────────────────────────────────────────────────
# 7. Price / FrontOrderType 組み合わせマトリクス
#    DelivType=0 / FundType='  ' / CashMargin=1 固定
#    ★ 限注ケースは実際に約定する可能性があります ★
# ────────────────────────────────────────────────────────────────────
def run_price_matrix(token: str, symbol: str, qty: int, exchange: int,
                     limit_price: int) -> list:
    """
    Price フィールドの有無・値 と FrontOrderType の組み合わせを検証する。
    DelivType=0 / FundType='  ' / CashMargin=1 固定。

    limit_price: 現在値より低い指値（例: 現在値 598 → 550 程度）
                 実際に約定する可能性があるため注意。
    """
    import time
    headers = {"X-API-KEY": token, "Content-Type": "application/json"}

    BASE = {
        "Password":    TRADE_PASS,
        "Symbol":      symbol,
        "Exchange":    exchange,
        "SecurityType": 1,
        "Side":        "1",
        "CashMargin":  1,
        "DelivType":   0,
        "FundType":    "  ",
        "AccountType": ACCOUNT_TYPE,
        "Qty":         int(qty),
        "ExpireDay":   0,
    }

    cases = [
        # (FrontOrderType, Price_or_ABSENT, exchange, label, risk)
        (10, "ABSENT", 1, "Exchange=1 Price=なし（旧来）",                        "low"),
        (10, 0,        1, "Exchange=1 Price=0(int)",                             "low"),
        (10, 0,        9, "Exchange=9(SOR) Price=0(int) ← FIX候補",             "HIGH"),
        (13, 0,        9, "Exchange=9(SOR) FrontOrderType=13 Price=0",           "HIGH"),
        (20, limit_price, 9, f"Exchange=9(SOR) FrontOrderType=20 Price={limit_price}", "HIGH"),
    ]

    print()
    print("=" * 60)
    print("[PRICE MATRIX]")
    print(f"  Symbol={symbol}  Exchange={exchange}  Qty={qty}")
    print("  DelivType=0  FundType='  '  CashMargin=1")
    print(f"  limit_price={limit_price}")
    print()
    print("  !! 警告: HTTP 2xx が返った場合は実際に発注されます !!")
    print(f"  !! Price={limit_price} のケースは約定する可能性が高い !!")
    print("  !! 直ちに kabu Station の注文一覧を確認してください !!")
    print("=" * 60)

    results = []
    for front, price_val, exch, label, risk in cases:
        time.sleep(1)
        payload = dict(BASE, FrontOrderType=front, Exchange=exch)
        if price_val != "ABSENT":
            payload["Price"] = price_val

        resp = requests.post(f"{BASE_URL}/sendorder", headers=headers, json=payload)
        status = resp.status_code
        try:
            body_str = json.dumps(resp.json(), ensure_ascii=False)
        except Exception:
            body_str = resp.text[:300]

        success = status < 400
        marker  = "★★★ ACCEPTED ★★★" if success else "    rejected   "
        print(f"  {label}")
        print(f"    -> HTTP {status}  {marker}  body={body_str}")

        results.append({
            "FrontOrderType": front,
            "Price":          price_val,
            "http_status":    status,
            "success":        success,
            "body":           body_str,
        })

    print()
    accepted = [r for r in results if r["success"]]
    if accepted:
        print("[PRICE MATRIX RESULT] 受理されたケース:")
        for r in accepted:
            print(f"  FrontOrderType={r['FrontOrderType']}  Price={r['Price']}")
        print("  !! kabu Station の注文一覧を今すぐ確認してください !!")
    else:
        print("[PRICE MATRIX RESULT] 全ケース拒否")
        try:
            codes = [json.loads(r["body"]).get("Code") for r in results]
            print(f"  エラーコード一覧: {codes}")
        except Exception:
            pass

    return results


def run():
    print("=" * 60)
    print("kabu Station SELL 診断ツール")
    print("=" * 60)

    token     = get_token()
    positions = get_positions(token)

    # portfolio_state との照合（シンボルの .T を除いて比較）
    from src.paths import RUNTIME_DIR
    import json as _json
    ps_path  = RUNTIME_DIR / "portfolio_state.json"
    ps       = _json.loads(ps_path.read_text(encoding="utf-8")) if ps_path.exists() else {}
    held_syms  = {s.replace(".T", "") for s in ps.get("position_entry_dates", {}).keys()}
    api_syms   = {p.get("Symbol") for p in positions}
    api_leaves = {p.get("Symbol"): p.get("LeavesQty") for p in positions}

    print(f"\n[CROSS-CHECK]")
    print(f"  portfolio_state 保有 (4桁): {held_syms}")
    print(f"  kabu API 認識:             {api_syms}")
    print(f"  kabu API LeavesQty:        {api_leaves}")
    missing = held_syms - api_syms
    if missing:
        print(f"  !! 不一致: {missing}")
    else:
        print("  一致 OK")

    # ── Price マトリクス ─────────────────────────────────────────────
    sym, qty, exch, _desc = SELL_CANDIDATES[0]

    # 現在値を positions から取得して指値を算出（現在値の -3% を目安）
    current_price = next(
        (p.get("CurrentPrice") or p.get("Price", 0) for p in positions
         if p.get("Symbol") == sym),
        0,
    )
    # 指値は現在値より低い整数値（下位 10 円単位で切り捨て）
    # 約定リスクあり: 実行前に必ず kabu Station の注文一覧を確認すること
    limit_price = int((current_price * 0.97) // 10) * 10 if current_price > 0 else 550
    print(f"\n[INFO] {sym} 現在値={current_price}  算出指値={limit_price}")

    run_price_matrix(token, sym, qty, exch, limit_price)

    print("\n[DONE]")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    run()
