"""
kabusapi/client.py
kabuステーション REST API クライアント

前提:
  - auカブコム証券の口座を開設済みであること
  - kabuステーション（Windows アプリ）を起動していること
  - .env に KABU_API_PASSWORD / KABU_TRADE_PASSWORD を設定済みであること

公式ドキュメント:
  https://kabucom.github.io/kabusapi/reference/index.html
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path as _Path

import requests
from dotenv import load_dotenv

# client.py は src/kabusapi/client.py → src/.env は parents[1]/.env
_ENV_PATH = _Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
_PORT = int(os.getenv("KABU_API_PORT", "18080"))
BASE_URL = f"http://localhost:{_PORT}/kabusapi"

# 取引所コード
class Exchange:
    TSE   = 1   # 東証（板情報取得用）
    NSE   = 3   # 名証
    FSE   = 5   # 福証
    SSE   = 6   # 札証
    SOR   = 9   # SOR（最良執行）← sendorder には必ずこれを使う

# 売買区分
class Side:
    SELL = "1"
    BUY  = "2"

# 注文タイプ
class OrderType:
    MARKET        = 10   # 成行
    LIMIT         = 20   # 指値
    MARKET_OPEN   = 13   # 寄成（前場）
    MARKET_CLOSE  = 14   # 引成（前場）

# 現物/信用
class CashMargin:
    CASH   = 1   # 現物
    MARGIN = 2   # 信用新規
    CLOSE  = 3   # 信用返済

# 口座種別
class AccountType:
    GENERAL  = 1
    SPECIFIC = 4   # 特定口座（デフォルト推奨）
    NISA     = 2


# ------------------------------------------------------------------ #
# データクラス
# ------------------------------------------------------------------ #
@dataclass
class Board:
    """株価板情報"""
    symbol:         str
    symbol_name:    str
    exchange:       int
    current_price:  float
    bid_price:      float   # 売気配値
    ask_price:      float   # 買気配値
    trading_volume: float
    trading_value:  float
    raw:            dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_response(cls, data: dict) -> "Board":
        def _f(key: str, default: float = 0.0) -> float:
            v = data.get(key)
            return float(v) if v is not None else default
        return cls(
            symbol        = data.get("Symbol", ""),
            symbol_name   = data.get("SymbolName", ""),
            exchange      = data.get("Exchange", 0),
            current_price = _f("CurrentPrice"),
            bid_price     = _f("BidPrice"),
            ask_price     = _f("AskPrice"),
            trading_volume= _f("TradingVolume"),
            trading_value = _f("TradingValue"),
            raw           = data,
        )


def parse_order_response(data: dict) -> dict:
    """
    kabuステーション sendorder レスポンスを解析して成否を判定する。

    kabu Station の正常レスポンスは OrderId のみを返し Result キーを含まない場合がある。
    （2026-03-16 の実運用で確認: {"OrderId": "20260316A02N80057402"}）

    Returns:
        {
            "success":     bool,
            "order_id":    str,
            "result_code": int,   # 正常時 0、エラー時は API の値
        }
    """
    order_id    = data.get("OrderId") or data.get("order_id") or data.get("ID") or ""
    result_code = data.get("Result", data.get("ResultCode", None))

    logger.debug("[ORDER_RESPONSE] full=%s", data)

    if order_id and result_code is None:
        # OrderId が存在して Result キーが欠落 = kabu Station の正常レスポンス
        logger.info("[ORDER] OrderId=%s (Result key absent, treating as success)", order_id)
        return {"success": True, "order_id": order_id, "result_code": 0}
    elif result_code == 0:
        return {"success": True, "order_id": order_id, "result_code": 0}
    elif result_code is not None:
        return {"success": False, "order_id": order_id, "result_code": result_code}
    else:
        # OrderId も Result も存在しない = 空レスポンスまたは不明エラー
        return {"success": False, "order_id": "", "result_code": -1}


@dataclass
class OrderResult:
    """注文送信結果"""
    order_id:    str
    result_code: int
    raw:         dict = field(repr=False, default_factory=dict)

    @property
    def success(self) -> bool:
        return self.result_code == 0

    @classmethod
    def from_response(cls, data: dict) -> "OrderResult":
        parsed = parse_order_response(data)
        return cls(
            order_id    = parsed["order_id"],
            result_code = parsed["result_code"],
            raw         = data,
        )


# ------------------------------------------------------------------ #
# process 内 singleton token キャッシュ
# ------------------------------------------------------------------ #
class _TokenCache:
    """
    process 内 singleton: kabu Station token を TTL 期間再利用する。

    同一プロセス内で複数の KabuClient インスタンスが生成されても
    token 取得は最大 1 回に抑える。
    """

    _token: str = ""
    _fetched_at: float = 0.0
    _fetch_count: int = 0
    _TTL_SEC: int = 3000  # 50 分（kabu Station 有効期間 60 分 - 10 分マージン）

    def is_valid(self) -> bool:
        return bool(self._token) and (time.time() - self._fetched_at) < self._TTL_SEC

    def set(self, token: str) -> None:
        self._token = token
        self._fetched_at = time.time()
        self._fetch_count += 1

    def age(self) -> float:
        return time.time() - self._fetched_at

    def remaining(self) -> float:
        return max(0.0, self._TTL_SEC - self.age())

    def invalidate(self) -> None:
        self._token = ""
        self._fetched_at = 0.0
        logger.debug("token cache invalidated")


_CACHE = _TokenCache()


# ------------------------------------------------------------------ #
# APIクライアント本体
# ------------------------------------------------------------------ #
class KabuClient:
    """
    kabuステーション REST API クライアント

    使い方:
        client = KabuClient()
        client.fetch_token()

        board = client.get_board("7203", Exchange.TSE)
        print(f"トヨタ: ¥{board.current_price:,}")

        result = client.send_order(
            symbol       = "7203",
            exchange     = Exchange.TSE,
            side         = Side.BUY,
            qty          = 100,
            order_type   = OrderType.MARKET,
        )
    """

    def __init__(self) -> None:
        self._api_password   = os.getenv("KABU_API_PASSWORD", "")
        self._trade_password = os.getenv("KABU_TRADE_PASSWORD", "")
        self._account_type   = int(os.getenv("KABU_ACCOUNT_TYPE", "4"))
        self._token: str     = ""
        self._session        = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------ #
    # 認証
    # ------------------------------------------------------------------ #
    def fetch_token(self) -> str:
        """
        APIトークンを取得してセッションに設定する。
        TTL 内はキャッシュを再利用し HTTP リクエストを発行しない。
        kabuステーション起動後・取引時間中のみ有効。
        """
        if not self._api_password:
            raise ValueError(
                "KABU_API_PASSWORD が未設定です。.env を確認してください。"
            )

        if _CACHE.is_valid():
            self._token = _CACHE._token
            self._session.headers.update({"X-API-KEY": self._token})
            logger.debug(
                "token reused: age=%.0fs remaining=%.0fs fetch_count=%d",
                _CACHE.age(), _CACHE.remaining(), _CACHE._fetch_count,
            )
            return self._token

        # キャッシュ miss → 実際に取得
        resp = self._session.post(
            f"{BASE_URL}/token",
            json={"APIPassword": self._api_password},
        )
        resp.raise_for_status()
        self._token = resp.json()["Token"]
        _CACHE.set(self._token)
        self._session.headers.update({"X-API-KEY": self._token})
        logger.info("トークン取得成功")
        logger.debug("token fetched: fetch_count=%d", _CACHE._fetch_count)
        return self._token

    # ------------------------------------------------------------------ #
    # 内部ヘルパ
    # ------------------------------------------------------------------ #
    def _request_with_token_retry(self, method: str, url: str, **kwargs) -> "requests.Response":
        """
        HTTP リクエストを発行し、401 を受けた場合はキャッシュを破棄して
        fetch_token() を再実行した上で 1 回だけリトライする。
        無限ループ防止のため最大リトライ回数は 1 回。
        """
        resp = getattr(self._session, method)(url, **kwargs)
        if resp.status_code == 401:
            logger.warning("401 received, invalidating token cache and retrying once")
            _CACHE.invalidate()
            self.fetch_token()
            resp = getattr(self._session, method)(url, **kwargs)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------ #
    # 株価取得
    # ------------------------------------------------------------------ #
    def get_board(self, symbol: str, exchange: int = Exchange.TSE) -> Board:
        """
        リアルタイム株価板を取得する。

        Args:
            symbol:   銘柄コード（例: "7203"）
            exchange: 取引所コード（Exchange.TSE など）

        Returns:
            Board オブジェクト
        """
        resp = self._request_with_token_retry("get", f"{BASE_URL}/board/{symbol}@{exchange}")
        return Board.from_response(resp.json())

    # ------------------------------------------------------------------ #
    # 注文
    # ------------------------------------------------------------------ #
    def send_order(
        self,
        symbol:      str,
        exchange:    int  = 9,  # SOR（最良執行）: TSE=1はsendorderで拒否される
        side:        str  = Side.BUY,
        qty:         int  = 100,
        order_type:  int  = OrderType.MARKET,
        price:       float = 0.0,
        cash_margin: int  = CashMargin.CASH,
        expire_day:  int  = 0,
    ) -> OrderResult:
        """
        株式注文を送信する。

        Args:
            symbol:      銘柄コード
            exchange:    取引所コード
            side:        Side.BUY / Side.SELL
            qty:         注文数量（株）
            order_type:  OrderType.MARKET（成行）/ OrderType.LIMIT（指値）
            price:       指値価格（成行の場合は 0）
            cash_margin: CashMargin.CASH（現物）/ CashMargin.MARGIN（信用）
            expire_day:  有効期限（0=当日中）

        Returns:
            OrderResult オブジェクト

        !! 注意 !!
            本関数は実際の発注を行います。
            テスト環境（auカブコム証券 模擬取引）での確認を推奨します。
        """
        if not self._token:
            raise RuntimeError("fetch_token() を先に呼び出してください。")

        payload: dict[str, Any] = {
            "Password":        self._trade_password,
            "Symbol":          symbol,
            "Exchange":        exchange,
            "SecurityType":    1,            # 1=株式
            "Side":            side,
            "CashMargin":      cash_margin,
            "DelivType":       2,            # 2=お預り金
            "FundType":        "AA",         # AA=特定預り（特定口座）
            "AccountType":     self._account_type,  # 4=特定口座
            "Qty":             qty,
            "FrontOrderType":  order_type,
            "Price":           price,
            "ExpireDay":       expire_day,
        }
        resp = self._request_with_token_retry("post", f"{BASE_URL}/sendorder", json=payload)
        result = OrderResult.from_response(resp.json())
        if result.success:
            logger.info("注文送信成功: OrderId=%s", result.order_id)
        else:
            logger.warning("注文送信失敗: %s", result.raw)
        return result

    # ------------------------------------------------------------------ #
    # 照会
    # ------------------------------------------------------------------ #
    def get_positions(self) -> list[dict]:
        """現在の保有ポジション一覧を取得する。"""
        resp = self._request_with_token_retry("get", f"{BASE_URL}/positions")
        return resp.json() or []

    def get_wallet_cash(self) -> dict:
        """現物買付余力を取得する。"""
        resp = self._request_with_token_retry(
            "get", f"{BASE_URL}/wallet/cash",
            params={"symbol": "", "exchange": Exchange.TSE},
        )
        return resp.json()

    def get_orders(self, only_open: bool = True) -> list[dict]:
        """
        注文一覧を取得する。

        Args:
            only_open: True の場合、未約定・一部約定のみ返す
        """
        params = {"details": "true"}
        if only_open:
            params["product"] = "0"
        resp = self._request_with_token_retry("get", f"{BASE_URL}/orders", params=params)
        return resp.json() or []

    def get_filled_orders(self) -> list[dict]:
        """
        約定済み注文一覧を取得する（state=5: 完了）。

        kabuステーション API パラメータ:
          product: 1=株式
          state:   5=完了（全部約定 / 訂正取消後約定）
          details: true=約定明細を含む

        Returns:
            約定済み注文リスト（各要素に Details: [{Price, Qty, ...}] が含まれる）
        """
        resp = self._request_with_token_retry(
            "get", f"{BASE_URL}/orders",
            params={"product": "1", "state": "5", "details": "true"},
        )
        return resp.json() or []

    def cancel_order(self, order_id: str) -> dict:
        """注文をキャンセルする。"""
        payload = {
            "OrderId":  order_id,
            "Password": self._trade_password,
        }
        resp = self._request_with_token_retry("put", f"{BASE_URL}/cancelorder", json=payload)
        return resp.json()
