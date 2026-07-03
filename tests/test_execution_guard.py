import unittest
from src.live.execution_guard import check_execution_preconditions


class TestExecutionGuard(unittest.TestCase):

    def test_blocked_universe_too_narrow(self):
        result = check_execution_preconditions(
            active_syms=["A", "B"],  # 3未満
            signals=[{"symbol": "A"}],
            current_positions={},
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "universe_too_narrow")

    def test_no_signal(self):
        result = check_execution_preconditions(
            active_syms=["A", "B", "C", "D"],
            signals=[],
            current_positions={},
        )
        self.assertEqual(result["status"], "NO_SIGNAL")

    def test_daily_limit(self):
        result = check_execution_preconditions(
            active_syms=["A"] * 5,
            signals=[{"symbol": "A"}],
            current_positions={},
            daily_order_count=10,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "daily_order_limit_exceeded")

    def test_ok(self):
        result = check_execution_preconditions(
            active_syms=["A", "B", "C", "D"],
            signals=[{"symbol": "A"}],
            current_positions={},
            daily_order_count=0,
        )
        self.assertEqual(result["status"], "OK")

    def test_exactly_min_syms_ok(self):
        """ちょうど MIN_ACTIVE_SYMS (=3) なら OK"""
        result = check_execution_preconditions(
            active_syms=["A", "B", "C"],
            signals=[{"symbol": "A"}],
            current_positions={},
        )
        self.assertEqual(result["status"], "OK")

    def test_daily_limit_not_yet_exceeded(self):
        """MAX_ORDERS_PER_DAY - 1 なら通過する"""
        result = check_execution_preconditions(
            active_syms=["A", "B", "C"],
            signals=[{"symbol": "A"}],
            current_positions={},
            daily_order_count=9,
        )
        self.assertEqual(result["status"], "OK")


if __name__ == "__main__":
    unittest.main()
