"""
addon — Winner add-on execution layer

Exports the public interface for:
  - WinnerConfirmationEngine (pure confirmation logic)
  - ExtensionFilter + PersistenceQualityScoring (exhaustion suppression)
  - AddOnExecutionPolicy (bounded order generation + state persistence)
  - State helpers
"""
from .winner_confirmation import (
    check_winner,
    WinnerConfirmation,
    MIN_PROFIT_THRESHOLD,
    MIN_HOLD_BEFORE_ADDON,
    MAX_RSR_RANK_FOR_ADDON,
    MIN_RSR_STRENGTH,
    MIN_TREND_QUALITY,
    MAX_DETERIORATION,
    MAX_ATR_RATIO,
    REGIME_BEAR_BLOCK,
)
from .extension_filter import (
    check_extension,
    ExtensionInput,
    ExtensionResult,
    MAX_VELOCITY_RATE,
    MAX_RSR_ACCELERATION,
    MAX_CANDLE_RANGE_RATIO,
    MAX_VOLUME_RATIO,
    ISOLATED_RSR_THRESHOLD,
    PORTFOLIO_DIVERGENCE_THRESHOLD,
)
from .addon_policy import (
    AddOnExecutionPolicy,
    AddOnOrder,
    AddOnResult,
    UNIT_SHARES,
    MAX_ADDON_DEPTH,
    ADDON_COOLDOWN_DAYS,
    MAX_POSITION_WEIGHT_ADDON,
    MIN_PORTFOLIO_PNL_FOR_ADDON,
    MAX_ADDONS_PER_RUN,
)
from .addon_state import (
    AddonSymbolState,
    AddonDecisionRecord,
    load_addon_state,
    save_addon_state,
    append_decision_record,
    purge_exited_symbols,
)

__all__ = [
    "check_winner",
    "WinnerConfirmation",
    "MIN_PROFIT_THRESHOLD",
    "MIN_HOLD_BEFORE_ADDON",
    "MAX_RSR_RANK_FOR_ADDON",
    "MIN_RSR_STRENGTH",
    "MIN_TREND_QUALITY",
    "MAX_DETERIORATION",
    "MAX_ATR_RATIO",
    "REGIME_BEAR_BLOCK",
    "check_extension",
    "ExtensionInput",
    "ExtensionResult",
    "MAX_VELOCITY_RATE",
    "MAX_RSR_ACCELERATION",
    "MAX_CANDLE_RANGE_RATIO",
    "MAX_VOLUME_RATIO",
    "ISOLATED_RSR_THRESHOLD",
    "PORTFOLIO_DIVERGENCE_THRESHOLD",
    "AddOnExecutionPolicy",
    "AddOnOrder",
    "AddOnResult",
    "UNIT_SHARES",
    "MAX_ADDON_DEPTH",
    "ADDON_COOLDOWN_DAYS",
    "MAX_POSITION_WEIGHT_ADDON",
    "MIN_PORTFOLIO_PNL_FOR_ADDON",
    "MAX_ADDONS_PER_RUN",
    "AddonSymbolState",
    "AddonDecisionRecord",
    "load_addon_state",
    "save_addon_state",
    "append_decision_record",
    "purge_exited_symbols",
]
