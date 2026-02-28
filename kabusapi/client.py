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

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
_PORT = int(os.getenv("KABU_API_PORT", "18080"))
BASE_URL = f"http://localhost:{_PORT}/kabusapi"

# 取引所コード
class Exchange:
    TSE   = 1   # 東証
    NSE   = 3   # 名証
    FSE   = 5   # 福証
    SSE   = 6   # 札証
    SOR   = 9   # SOR（最良執行）

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
        return cls(
            symbol        = data.get("Symbol", ""),
            symbol_name   = data.get("SymbolName", ""),
            exchange      = data.get("Exchange", 0),
            current_price = data.get("CurrentPrice", 0.0),
            bid_price     = data.get("BidPrice", 0.0),
            ask_price     = data.get("AskPrice", 0.0),
            trading_volume= data.get("TradingVolume", 0.0),
            trading_value = data.get("TradingValue", 0.0),
            raw           = data,
        )


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
        return cls(
            order_id    = data.get("OrderId", ""),
            result_code = data.get("ResultCode", -1),
            raw         = data,
        )


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
        kabuステーション起動後・取引時間中のみ有効。
        """
        if not self._api_password:
            raise ValueError(
                "KABU_API_PASSWORD が未設定です。.env を確認してください。"
            )

        resp = self._session.post(
            f"{BASE_URL}/token",
            json={"APIPassword": self._api_password},
        )
        resp.raise_for_status()
        self._token = resp.json()["Token"]
        self._session.headers.update({"X-API-KEY": self._token})
        logger.info("トークン取得成功")
        return self._token

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
        resp = self._session.get(f"{BASE_URL}/board/{symbol}@{exchange}")
        resp.raise_for_status()
        return Board.from_response(resp.json())

    # ------------------------------------------------------------------ #
    # 注文
    # ------------------------------------------------------------------ #
    def send_order(
        self,
        symbol:      str,
        exchange:    int,
        side:        str,
        qty:         int,
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
            "FundType":        "  ",
            "AccountType":     self._account_type,
            "Qty":             qty,
            "FrontOrderType":  order_type,
            "Price":           price,
            "ExpireDay":       expire_day,
        }
        resp = self._session.post(f"{BASE_URL}/sendorder", json=payload)
        resp.raise_for_status()
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
        resp = self._session.get(f"{BASE_URL}/positions")
        resp.raise_for_status()
        return resp.json() or []

    def get_wallet_cash(self) -> dict:
        """現物買付余力を取得する。"""
        resp = self._session.get(
            f"{BASE_URL}/wallet/cash",
            params={"symbol": "", "exchange": Exchange.TSE},
        )
        resp.raise_for_status()
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
        resp = self._session.get(f"{BASE_URL}/orders", params=params)
        resp.raise_for_status()
        return resp.json() or []

    def cancel_order(self, order_id: str) -> dict:
        """注文をキャンセルする。"""
        payload = {
            "OrderId":  order_id,
            "Password": self._trade_password,
        }
        resp = self._session.put(f"{BASE_URL}/cancelorder", json=payload)
        resp.raise_for_status()
        return resp.json()
