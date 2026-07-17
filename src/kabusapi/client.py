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

import hashlib as _hashlib
import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from pathlib import Path as _Path

import requests
from dotenv import load_dotenv

# client.py は src/kabusapi/client.py → src/.env は parents[1]/.env
_ENV_PATH = _Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))


def _log_token_forensic(
    resp: "requests.Response",
    pw: str,
    sess_headers: dict,
) -> None:
    """[TOKEN_FORENSIC] 非200応答の詳細をログ出力。FAIL_OPEN。パスワード平文出力禁止。"""
    try:
        _safe_sess = {
            k: "***" if k.lower() in {"x-api-key", "authorization"} else v
            for k, v in sess_headers.items()
        }
        logger.warning(
            "[TOKEN_FORENSIC] timestamp=%s pid=%d status=%d"
            " body=%s response_headers=%s request_url=%s"
            " session_headers=%s pw_len=%d pw_sha8=%s",
            datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            os.getpid(),
            resp.status_code,
            resp.text[:500],
            dict(resp.headers),
            getattr(resp, "url", "unknown"),
            _safe_sess,
            len(pw),
            _hashlib.sha256(pw.encode("utf-8")).hexdigest()[:8] if pw else "",
        )
    except Exception:
        pass


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
    MARKET_OPEN_PM  = 15   # 寄成（後場）
    MARKET_CLOSE_PM = 16   # 引成（後場）
    MARKET_CLOSE_FORCED = 17  # 不成（指値不成立時に成行）

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
        if not resp.ok:
            _log_token_forensic(resp, self._api_password, dict(self._session.headers))
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
    # 注文 バリデーション
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_order_payload(payload: dict) -> None:
        """
        sendorder 直前の payload 整合性チェック。
        不整合は ValueError を raise して API 送信を阻止する（再送禁止）。
        """
        _market_types = {10, 13, 14, 15, 16, 17}

        front = payload.get("FrontOrderType")
        required_base = {
            "Password", "Symbol", "Exchange", "SecurityType",
            "Side", "CashMargin", "DelivType", "FundType",
            "AccountType", "Qty", "FrontOrderType", "Price", "ExpireDay",
        }
        missing = required_base - payload.keys()
        if missing:
            raise ValueError(f"sendorder payload 必須キー不足: {missing}")

        price = payload["Price"]
        if front in _market_types:
            # 成行系: Price=0（int）必須
            if price != 0:
                raise ValueError(
                    f"MARKET系注文 (FrontOrderType={front}) は Price=0 必須, got {price!r}"
                )
            if not isinstance(price, int):
                raise ValueError(
                    f"MARKET系注文の Price は int(0) 必須（float 0.0 は 500 を誘発）, got {type(price)}"
                )
        else:
            # 指値: Price>0 必須
            if price is None or price <= 0:
                raise ValueError(
                    f"LIMIT注文 (FrontOrderType={front}) は Price>0 必須, got {price!r}"
                )

    # ------------------------------------------------------------------ #
    # 注文
    # ------------------------------------------------------------------ #
    def send_order(
        self,
        symbol:      str,
        exchange:    int  = 9,  # BUY: SOR=9（最良執行）/ SELL: TSE=1 を signal_bridge 側で指定
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
                         BUY  → Exchange.SOR (9): 最良執行
                         SELL → Exchange.TSE (1): 保有取引所（東証）を明示
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

        DelivType / FundType の side 別仕様（kabu API 公式）:
            BUY  (Side="2"): DelivType=2（お預り金）, FundType="AA"（特定預り）
            SELL (Side="1"): DelivType=0（自動/受渡なし）, FundType="  "（出金）
        """
        if not self._token:
            raise RuntimeError("fetch_token() を先に呼び出してください。")

        # ── ENTRY FREEZE 最終ガード（資産保全・2026-07-17・defense-in-depth）──
        # 全BUY発注経路（signal_bridge._send_orders / broker_worker.py 子プロセス /
        # run_morning_signal.py）が最終的に収束するこの関数の先頭で、payload構築・
        # HTTP送信の一切手前でBUYを遮断する。上流ゲート（_build_orders等）が
        # 万一バイパスされても、ここが最後の砦として機能する。
        if side == Side.BUY:
            try:
                from src.config_loader import load_strategy_config
                _ef = load_strategy_config().entry_freeze
            except Exception as _ef_err:
                _ef = None
                logger.warning("[ENTRY_FREEZE_GUARD] config読込失敗・freeze判定不能: %s", _ef_err)
            if _ef is not None and _ef.enabled:
                logger.critical(
                    "[ENTRY_FREEZE_GUARD] BUY blocked at sendorder boundary: "
                    "symbol=%s qty=%s reason=%s — API call NOT made",
                    symbol, qty, _ef.reason,
                )
                return OrderResult(
                    order_id="", result_code=-1,
                    raw={"rejected": "entry_freeze", "reason": _ef.reason,
                         "symbol": symbol, "qty": qty},
                )

        is_sell = (side == Side.SELL)

        # ------------------------------------------------------------------
        # BUY / SELL で DelivType・FundType が異なる（kabu API 仕様）
        #   BUY : DelivType=2(お預り金),  FundType="AA"(特定預り)
        #   SELL: DelivType=0(受渡なし),  FundType="  "(出金/空白2文字)
        # DelivType=2 のまま SELL を送ると 400 Bad Request になる
        # ------------------------------------------------------------------
        deliv_type = 0    if is_sell else 2
        fund_type  = "  " if is_sell else "AA"

        # kabu API は Qty を整数で要求する。
        # positions["qty"] が LeavesQty 由来で float になる場合があるため強制変換する。
        # float のまま送ると 500 Internal Server Error になる。
        qty_int = int(qty)

        payload: dict[str, Any] = {
            "Password":        self._trade_password,
            "Symbol":          symbol,
            "Exchange":        exchange,
            "SecurityType":    1,                   # 1=株式
            "Side":            side,
            "CashMargin":      cash_margin,          # 1=現物
            "DelivType":       deliv_type,
            "FundType":        fund_type,
            "AccountType":     self._account_type,  # 4=特定口座
            "Qty":             qty_int,
            "FrontOrderType":  order_type,
            "Price":           price,
            "ExpireDay":       expire_day,
        }

        # kabu API 仕様（diagnose_sell.py 総当たり検証 2026-04-15 確定）:
        #   成行系（FrontOrderType 10/13/14/15/16/17）は Price=0（int）必須。
        #   Price を省略すると 4002017「値段指定エラー」。
        #   Price=0.0（float）は不可（500 を誘発する kabu Station 内部バグ）。
        #   int(0) を明示することで float serialization を回避する。
        _market_types = {
            OrderType.MARKET,
            OrderType.MARKET_OPEN,
            OrderType.MARKET_CLOSE,
            OrderType.MARKET_OPEN_PM,
            OrderType.MARKET_CLOSE_PM,
            OrderType.MARKET_CLOSE_FORCED,
        }
        if order_type in _market_types:
            payload["Price"] = 0   # int 0 → JSON "Price":0（float 0.0 は不可）

        self._validate_order_payload(payload)

        logger.debug(
            "SEND_ORDER_PAYLOAD qty=%s qty_type=%s payload=%s",
            qty_int,
            type(qty_int).__name__,
            payload,
        )

        try:
            resp = self._request_with_token_retry("post", f"{BASE_URL}/sendorder", json=payload)
        except requests.HTTPError as e:
            _status = e.response.status_code if e.response is not None else "N/A"
            _body   = e.response.text        if e.response is not None else "N/A"
            logger.error("SEND_ORDER_ERROR status=%s body=%s", _status, _body)
            # ターミナルに直接出力（log level 設定に依存しない）
            import sys as _sys
            print(
                f"[KABU API ERROR] HTTP {_status}\n"
                f"  payload_symbol={payload.get('Symbol')} side={payload.get('Side')}"
                f" front={payload.get('FrontOrderType')} price_in_payload={'Price' in payload}\n"
                f"  response_body={_body}",
                file=_sys.stderr,
            )
            raise

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
