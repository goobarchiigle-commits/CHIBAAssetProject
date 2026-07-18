"""
tests/test_startup_check_auth.py — API 認証プリフライト [7] テスト

Coverage:
  - _check_api_token: 成功 / 空token / 401 / 503 / ConnectionError / Timeout / ValueError
  - run_startup_check(is_live=True):  auth失敗 → issues → ok=False
  - run_startup_check(is_live=False): auth失敗 → warnings → ok=True
  - run_startup_check: auth成功 → issue/warning なし
  - ポート到達不可の場合は [7] スキップ
  - return dict に api_auth_ok / api_auth_msg キーが存在する
  - 6/8インシデント再現: port=OK + 401 → LIVE ok=False
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.startup_check import _check_api_token, run_startup_check


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_http_error(status_code: int):
    """requests.exceptions.HTTPError をモック生成する。"""
    import requests
    resp = MagicMock()
    resp.status_code = status_code
    err = requests.exceptions.HTTPError(response=resp)
    return err


def _patched_run(
    is_live: bool,
    auth_side_effect=None,
    auth_return: tuple = (True, "API token OK"),
    port_ok: bool = True,
) -> dict:
    """
    run_startup_check をポート / 認証 / portfolio / lock をすべてモックして実行。
    auth_side_effect が指定されたら _check_api_token が例外を送出する。
    """
    import src.startup_check as sc

    dummy_state = {
        "cb_state":       "NORMAL",
        "equity_peak":    3_000_000.0,
        "available_cash": 2_500_000.0,
        "last_equity":    2_800_000.0,
    }

    def _fake_auth(_out=None):
        if auth_side_effect is not None:
            raise auth_side_effect
        return auth_return

    port_msg = f"API port 18080 {'OK' if port_ok else '到達不可: mock'}"
    with (
        patch.object(sc, "_check_api_port",        return_value=(port_ok, port_msg)),
        patch.object(sc, "_check_api_token",        side_effect=_fake_auth),
        patch.object(sc, "_check_portfolio_state",  return_value=(True, [], [], dummy_state)),
        patch.object(sc, "_check_lock_file",        return_value=(True, "OK")),
        patch.object(sc, "_check_snapshot_date",
                     return_value=(False, "snapshot OK", "2026-06-06", "2026-06-05")),
        patch("src.startup_check.datetime") as mock_dt,
    ):
        mock_dt.now.return_value.strftime.return_value = "2026-06-09 08:43:00 JST"
        mock_dt.now.return_value.date.return_value = __import__("datetime").date(2026, 6, 9)
        return sc.run_startup_check(is_live=is_live)


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckApiToken — ユニットテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckApiToken(unittest.TestCase):

    def _run(self, side_effect=None, token_return: str = "valid_token_abc123"):
        with patch("src.kabusapi.client.KabuClient") as mock_cls:
            mock_instance = mock_cls.return_value
            if side_effect is not None:
                mock_instance.fetch_token.side_effect = side_effect
            else:
                mock_instance.fetch_token.return_value = token_return
            return _check_api_token()

    def test_success_returns_true(self):
        ok, msg = self._run()
        assert ok is True
        assert "OK" in msg

    def test_empty_token_returns_false(self):
        ok, msg = self._run(token_return="")
        assert ok is False
        assert "empty" in msg

    def test_http_401_returns_false_with_detail(self):
        ok, msg = self._run(side_effect=_make_http_error(401))
        assert ok is False
        assert "401" in msg
        assert "kabu Station" in msg, f"Expected 'kabu Station' in msg: {msg}"

    def test_http_401_message_not_same_as_503(self):
        _, msg_401 = self._run(side_effect=_make_http_error(401))
        _, msg_503 = self._run(side_effect=_make_http_error(503))
        assert msg_401 != msg_503, "401 と 503 のメッセージは区別されるべき"

    def test_http_503_returns_false_without_login_detail(self):
        ok, msg = self._run(side_effect=_make_http_error(503))
        assert ok is False
        assert "503" in msg
        assert "未ログイン" not in msg

    def test_http_500_returns_false(self):
        ok, msg = self._run(side_effect=_make_http_error(500))
        assert ok is False
        assert "500" in msg

    def test_connection_error_returns_false(self):
        import requests
        ok, msg = self._run(side_effect=requests.exceptions.ConnectionError("refused"))
        assert ok is False
        assert "接続エラー" in msg

    def test_timeout_returns_false(self):
        import requests
        ok, msg = self._run(side_effect=requests.exceptions.Timeout())
        assert ok is False
        assert "タイムアウト" in msg

    def test_value_error_no_password_returns_false(self):
        ok, msg = self._run(side_effect=ValueError("KABU_API_PASSWORD が未設定"))
        assert ok is False
        assert "パスワード未設定" in msg

    def test_unexpected_exception_returns_false(self):
        ok, msg = self._run(side_effect=RuntimeError("unexpected"))
        assert ok is False
        assert "RuntimeError" in msg


# ─────────────────────────────────────────────────────────────────────────────
# TestRunStartupCheckApiAuth — 統合テスト
# ─────────────────────────────────────────────────────────────────────────────

class TestRunStartupCheckApiAuth(unittest.TestCase):

    # ── LIVE + auth fail ───────────────────────────────────────────────────────

    def test_live_auth_fail_sets_ok_false(self):
        result = _patched_run(is_live=True, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        assert result["ok"] is False, f"LIVE+auth失敗は ok=False であるべき: {result['issues']}"

    def test_live_auth_fail_adds_api_auth_issue(self):
        result = _patched_run(is_live=True, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        auth_issues = [i for i in result["issues"] if "API_AUTH" in i]
        assert auth_issues, f"[API_AUTH] issue がない: {result['issues']}"

    def test_live_auth_fail_issue_contains_live_prefix(self):
        result = _patched_run(is_live=True, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        auth_issues = [i for i in result["issues"] if "API_AUTH" in i]
        assert any("LIVE発注停止" in i for i in auth_issues)

    def test_live_auth_fail_no_warning(self):
        result = _patched_run(is_live=True, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        auth_warns = [w for w in result["warnings"] if "API_AUTH" in w]
        assert not auth_warns, "LIVE+auth失敗は warning を出さない（issue のみ）"

    def test_live_auth_fail_api_auth_ok_false(self):
        result = _patched_run(is_live=True, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        assert result["api_auth_ok"] is False

    # ── DRY + auth fail ────────────────────────────────────────────────────────

    def test_dry_auth_fail_keeps_ok_true(self):
        """
        Broker-as-Sole-SSOT (2026-07-18): [7]認証チェック自体は従来通りDRYでは
        issueでなくwarning扱い（test_dry_auth_fail_no_issue参照）。ただし認証失敗時は
        _compute_startup_equity()のbroker snapshot取得も同じ理由で失敗するため、
        LIVE/DRY問わずok=Falseとなる（DRYが「エラーを警告に格下げして継続する」
        設計は資産計算の欠落を許容しない）。
        """
        result = _patched_run(is_live=False, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        assert result["ok"] is False, "DRY+auth失敗はbroker snapshot取得も失敗するためok=False"
        broker_issues = [i for i in result["issues"] if "BROKER_UNAVAILABLE" in i]
        assert broker_issues, f"[BROKER_UNAVAILABLE] issue が必要: {result['issues']}"

    def test_dry_auth_fail_adds_api_auth_warning(self):
        result = _patched_run(is_live=False, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        auth_warns = [w for w in result["warnings"] if "API_AUTH" in w]
        assert auth_warns, f"DRY+auth失敗は [API_AUTH] 警告を出すべき: {result['warnings']}"

    def test_dry_auth_fail_warning_contains_dry_prefix(self):
        result = _patched_run(is_live=False, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        auth_warns = [w for w in result["warnings"] if "API_AUTH" in w]
        assert any("DRY継続" in w for w in auth_warns)

    def test_dry_auth_fail_no_issue(self):
        result = _patched_run(is_live=False, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        auth_issues = [i for i in result["issues"] if "API_AUTH" in i]
        assert not auth_issues, "DRY+auth失敗は issue を出さない"

    def test_dry_auth_fail_api_auth_ok_false(self):
        result = _patched_run(is_live=False, auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)"))
        assert result["api_auth_ok"] is False

    # ── auth 成功 ──────────────────────────────────────────────────────────────

    def test_live_auth_ok_no_issue(self):
        result = _patched_run(is_live=True, auth_return=(True, "API token OK"))
        auth_issues = [i for i in result["issues"] if "API_AUTH" in i]
        assert not auth_issues

    def test_dry_auth_ok_no_warning(self):
        result = _patched_run(is_live=False, auth_return=(True, "API token OK"))
        auth_warns = [w for w in result["warnings"] if "API_AUTH" in w]
        assert not auth_warns

    def test_auth_ok_api_auth_ok_true(self):
        result = _patched_run(is_live=True, auth_return=(True, "API token OK"))
        assert result["api_auth_ok"] is True

    def test_auth_ok_msg_preserved(self):
        result = _patched_run(is_live=True, auth_return=(True, "API token OK"))
        assert result["api_auth_msg"] == "API token OK"

    # ── ポート到達不可 → [7] スキップ ─────────────────────────────────────────

    def test_port_fail_auth_check_skipped(self):
        result = _patched_run(is_live=True, port_ok=False)
        # port fail → issues にポートエラー, auth_ok=True (skip=not failure)
        assert result["api_auth_ok"] is True
        assert "skipped" in result["api_auth_msg"]
        port_issues = [i for i in result["issues"] if "18080" in i]
        assert port_issues, "ポート到達不可は issue に含まれるべき"

    def test_port_fail_no_api_auth_issue(self):
        result = _patched_run(is_live=True, port_ok=False)
        auth_issues = [i for i in result["issues"] if "API_AUTH" in i]
        assert not auth_issues, "ポート到達不可時は API_AUTH issue を出さない"

    # ── return dict キー確認 ───────────────────────────────────────────────────

    def test_return_key_api_auth_ok_exists(self):
        result = _patched_run(is_live=False, auth_return=(True, "API token OK"))
        assert "api_auth_ok" in result, "return dict に api_auth_ok キーが必要"

    def test_return_key_api_auth_msg_exists(self):
        result = _patched_run(is_live=False, auth_return=(True, "API token OK"))
        assert "api_auth_msg" in result, "return dict に api_auth_msg キーが必要"

    def test_all_expected_keys_present(self):
        result = _patched_run(is_live=False, auth_return=(True, "API token OK"))
        for key in (
            "ok", "issues", "warnings", "summary", "state",
            "stale_data_detected", "snapshot_date", "expected_date",
            "api_auth_ok", "api_auth_msg", "timestamp",
        ):
            assert key in result, f"return dict に {key} キーが必要"

    # ── 6/8 インシデント再現 ───────────────────────────────────────────────────

    def test_6_8_incident_port_ok_401_live_blocked(self):
        # 6/8: port=到達可, fetch_token → 401 → LIVE発注停止
        # startup_check が [7] で検知できていれば SignalBridge まで到達しない
        result = _patched_run(
            is_live=True,
            port_ok=True,
            auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログインまたはAPIパスワード不一致)"),
        )
        assert result["ok"] is False, "6/8再現: port=OK + 401 → LIVE ok=False"
        assert result["api_auth_ok"] is False
        auth_issues = [i for i in result["issues"] if "API_AUTH" in i]
        assert auth_issues, "6/8再現: [API_AUTH] issue が生成されるべき"

    def test_6_8_incident_dry_would_continue(self):
        """
        Broker-as-Sole-SSOT (2026-07-18): 旧実装はDRY+401でも継続していたが、
        signal_bridgeフォールバック自体が廃止されたため、DRYもLIVEと同様broker
        snapshot取得失敗でok=Falseとなる（fail-closed統一）。[7]認証チェック自体の
        DRY warning扱いは変わらない。
        """
        result = _patched_run(
            is_live=False,
            port_ok=True,
            auth_return=(False, "API認証失敗 (HTTP 401: kabu Station未ログインまたはAPIパスワード不一致)"),
        )
        assert result["ok"] is False, "DRY+401: broker snapshot取得も失敗するためok=False"
        auth_warns = [w for w in result["warnings"] if "API_AUTH" in w]
        assert auth_warns


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckApiTokenWithRetry — リトライラッパーのユニットテスト
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckApiTokenWithRetry(unittest.TestCase):
    """
    _check_api_token_with_retry をテストする。
    - _check_api_token をモックして返り値を制御
    - _retry_sleep をモックして実際のスリープを省略
    """

    def _run_retry(
        self,
        side_effects: list,
        retries: int = 6,
        retry_seconds: int = 30,
    ) -> tuple[bool, str]:
        """
        side_effects: _check_api_token が順番に返す (ok, msg) のリスト。
        _retry_sleep をモックしてスリープを省略。
        """
        import src.startup_check as sc
        from src.startup_check import _check_api_token_with_retry

        call_iter = iter(side_effects)

        def _mock_token(_out=None):
            try:
                return next(call_iter)
            except StopIteration:
                return (False, "exhausted")

        with (
            patch.object(sc, "_check_api_token",  side_effect=_mock_token),
            patch.object(sc, "_retry_sleep",       MagicMock()) as mock_sleep,
        ):
            result = _check_api_token_with_retry(
                retries=retries, retry_seconds=retry_seconds
            )
            self._mock_sleep = mock_sleep
        return result

    # ── 成功ケース ────────────────────────────────────────────────────────────

    def test_success_on_first_attempt_returns_original_msg(self):
        ok, msg = self._run_retry([(True, "API token OK")])
        assert ok is True
        assert msg == "API token OK"
        assert "retry_success" not in msg, "1回目成功はサフィックスなし"

    def test_retry_success_on_second_attempt(self):
        side_effects = [
            (False, "API認証失敗 (HTTP 401: ...)"),
            (True,  "API token OK"),
        ]
        ok, msg = self._run_retry(side_effects)
        assert ok is True
        assert "retry_success" in msg
        assert "attempt=2/6" in msg

    def test_retry_success_on_last_attempt(self):
        side_effects = (
            [(False, "API認証失敗 (HTTP 401: ...)")] * 5
            + [(True, "API token OK")]
        )
        ok, msg = self._run_retry(side_effects)
        assert ok is True
        assert "retry_success" in msg
        assert "attempt=6/6" in msg

    def test_retry_exhausted_returns_false(self):
        side_effects = [(False, "API認証失敗 (HTTP 401: ...)")] * 6
        ok, msg = self._run_retry(side_effects)
        assert ok is False
        assert "retry_exhausted" in msg
        assert "6/6" in msg

    def test_retry_exhausted_preserves_last_error_msg(self):
        side_effects = [(False, "API認証失敗 (HTTP 401: kabu Station未ログイン)")] * 6
        ok, msg = self._run_retry(side_effects)
        assert "401" in msg or "kabu Station" in msg

    # ── スリープ呼び出し確認 ─────────────────────────────────────────────────

    def test_no_sleep_after_success_on_first_attempt(self):
        self._run_retry([(True, "API token OK")])
        assert self._mock_sleep.call_count == 0, "1回目成功はスリープしない"

    def test_no_sleep_after_success_on_second_attempt(self):
        side_effects = [(False, "fail"), (True, "API token OK")]
        self._run_retry(side_effects)
        # 1回失敗 → 1回スリープ → 成功
        assert self._mock_sleep.call_count == 1

    def test_sleep_called_between_failures(self):
        side_effects = [(False, "fail")] * 6
        self._run_retry(side_effects, retries=6, retry_seconds=30)
        # 6回失敗: attempt 0..4 でスリープ (5回)、attempt 5 ではスリープしない
        assert self._mock_sleep.call_count == 5

    def test_sleep_uses_retry_seconds_arg(self):
        side_effects = [(False, "fail"), (True, "OK")]
        self._run_retry(side_effects, retries=6, retry_seconds=30)
        self._mock_sleep.assert_called_with(30)

    def test_sleep_not_called_after_last_failure(self):
        """最後の失敗後はスリープしない（次のリトライがないため）。"""
        side_effects = [(False, "fail")] * 3
        self._run_retry(side_effects, retries=3, retry_seconds=30)
        # retries=3: attempt 0→sleep, 1→sleep, 2→no sleep
        assert self._mock_sleep.call_count == 2

    # ── メッセージフォーマット ─────────────────────────────────────────────────

    def test_retry_success_msg_format(self):
        side_effects = [(False, "fail"), (False, "fail"), (True, "API token OK")]
        ok, msg = self._run_retry(side_effects)
        assert ok is True
        assert "retry_success attempt=3/6" in msg

    def test_retry_exhausted_msg_format(self):
        side_effects = [(False, "401 error")] * 6
        ok, msg = self._run_retry(side_effects)
        assert "retry_exhausted 6/6" in msg

    # ── run_startup_check 統合: リトライ後に成功 → LIVE継続 ──────────────────

    def test_run_startup_check_live_continues_after_retry_success(self):
        """
        LIVE run: _check_api_token が1回目401、2回目OK
        → run_startup_check は ok=True でなければならない。

        Broker-as-Sole-SSOT (2026-07-18): [7]認証チェック（_check_api_token,
        リトライ付き）とは別に、_compute_startup_equity()もbroker snapshot取得の
        ためkabuステーションへ接続する（fetch_broker_snapshot() 経由）。この
        テストは[7]の認証リトライだけでなく、equity計算用のKabuClient接続も
        モックする必要がある（そうしないと実際にlocalhost:18080へ接続を試みて
        失敗し、ok=Falseになる）。
        """
        import src.startup_check as sc

        dummy_state = {
            "cb_state":       "NORMAL",
            "equity_peak":    3_000_000.0,
            "available_cash": 2_500_000.0,
            "last_equity":    2_800_000.0,
        }

        call_count = {"n": 0}

        def _flaky_token(_out=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return (False, "API認証失敗 (HTTP 401: ...)")
            return (True, "API token OK")

        mock_equity_client = MagicMock()
        mock_equity_client.get_wallet_cash.return_value = {"StockAccountWallet": 3_000_000.0}
        mock_equity_client.get_positions.return_value = []

        with (
            patch.object(sc, "_check_api_port",        return_value=(True, "OK")),
            patch.object(sc, "_check_api_token",        side_effect=_flaky_token),
            patch.object(sc, "_retry_sleep",            MagicMock()),
            patch.object(sc, "_check_portfolio_state",  return_value=(True, [], [], dummy_state)),
            patch.object(sc, "_check_lock_file",        return_value=(True, "OK")),
            patch.object(sc, "_check_snapshot_date",
                         return_value=(False, "snapshot OK", "2026-06-06", "2026-06-05")),
            patch("src.startup_check.datetime") as mock_dt,
            patch("src.kabusapi.client.KabuClient", return_value=mock_equity_client),
        ):
            mock_dt.now.return_value.strftime.return_value = "2026-06-09 08:43:00 JST"
            mock_dt.now.return_value.date.return_value = __import__("datetime").date(2026, 6, 9)
            result = sc.run_startup_check(is_live=True)

        assert result["ok"] is True, "1回401 + 2回目OK → LIVE継続"
        assert result["api_auth_ok"] is True
        assert "retry_success" in result["api_auth_msg"]

    def test_run_startup_check_live_blocks_after_all_retries_fail(self):
        """
        LIVE run: _check_api_token が常に401
        → run_startup_check は ok=False でなければならない（FAIL_CLOSED維持）。
        """
        import src.startup_check as sc

        dummy_state = {
            "cb_state":       "NORMAL",
            "equity_peak":    3_000_000.0,
            "available_cash": 2_500_000.0,
            "last_equity":    2_800_000.0,
        }

        with (
            patch.object(sc, "_check_api_port",        return_value=(True, "OK")),
            patch.object(sc, "_check_api_token",        return_value=(False, "API認証失敗 (HTTP 401: ...)")),
            patch.object(sc, "_retry_sleep",            MagicMock()),
            patch.object(sc, "_check_portfolio_state",  return_value=(True, [], [], dummy_state)),
            patch.object(sc, "_check_lock_file",        return_value=(True, "OK")),
            patch.object(sc, "_check_snapshot_date",
                         return_value=(False, "snapshot OK", "2026-06-06", "2026-06-05")),
            patch("src.startup_check.datetime") as mock_dt,
        ):
            mock_dt.now.return_value.strftime.return_value = "2026-06-09 08:43:00 JST"
            mock_dt.now.return_value.date.return_value = __import__("datetime").date(2026, 6, 9)
            result = sc.run_startup_check(is_live=True)

        assert result["ok"] is False, "全リトライ失敗 → LIVE停止 (FAIL_CLOSED維持)"
        assert result["api_auth_ok"] is False
        assert "retry_exhausted" in result["api_auth_msg"]


if __name__ == "__main__":
    unittest.main()
