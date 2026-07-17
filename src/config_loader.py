from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from src.paths import STRATEGY_CONFIG_FILE


@dataclass(frozen=True)
class FujikoConfig:
    min_sepa: int
    min_rsr: float        # entry threshold (絶対変更禁止)
    rsr_exit: float       # exit threshold (entry と分離; fallback=min_rsr)
    mom_period: int
    turtle_entry: int
    turtle_exit: int
    use_turtle_entry: bool


@dataclass(frozen=True)
class MeanReversionConfig:
    rsi_period: int
    rsi_entry: float
    rsi_exit: float
    ma_long: int
    stop_loss_pct: float
    max_hold_days: int


@dataclass(frozen=True)
class PortfolioConfig:
    capital: int
    max_positions: int
    min_sectors: int
    max_single_weight: float
    max_dd_limit: float
    vol_target: float
    use_idm: bool


@dataclass(frozen=True)
class RiskConfig:
    min_hold_days: int
    max_hold_days: int
    emergency_exit_pct: float
    rebalance_interval: int
    reentry_cooldown_days: int = 5   # SELL後の同銘柄再エントリー禁止（営業日）。全SELL理由に適用。


@dataclass(frozen=True)
class SectorConcentrationConfig:
    enabled: bool = True
    max_names_per_sector: int = 1
    max_weight_per_sector: float = 0.12


@dataclass(frozen=True)
class BearUniverseFilterConfig:
    enabled: bool = True
    lookback_days: int = 60
    ma200_below_days: int = 40
    excluded_sectors: tuple = (
        "機械", "鉄鋼", "銀行業", "保険業", "輸送用機器", "海運業", "化学"
    )


@dataclass(frozen=True)
class RiskControlsConfig:
    shock_exit_mode: str    # "full_exit" | "partial_50" | "composite"
    regime_sizing: str      # "none" | "regime_2" | "regime_4"
    bear_scale: float       # TOPIX MA200下のポジションサイズ係数
    dynamic_cap: bool       # True=ボラ連動cap, False=固定cap（default OFF）
    symbol_cap: float       # 固定 symbol cap（dynamic_cap=False 時）
    sector_cap: float       # 固定 sector cap（dynamic_cap=False 時）
    cluster_cap: float      # cluster cap 上限
    bear_sector_cap: float  # Bear 時の sector cap
    bear_cluster_cap: float # Bear 時の cluster cap
    # gross exposure 縦方向制御（2026-04-07 追加）
    gross_exposure_enabled: bool = True
    gross_cap_normal: float = 1.0
    gross_cap_drawdown_5pct: float = 0.6
    gross_cap_drawdown_8pct: float = 0.4
    # bear universe filter（2026-04-07 本番反映）
    bear_universe_filter: BearUniverseFilterConfig = field(default_factory=BearUniverseFilterConfig)
    # セクター集中ゲート（2026-04-09 追加）
    sector_concentration: SectorConcentrationConfig = field(default_factory=SectorConcentrationConfig)


@dataclass(frozen=True)
class EntryFreezeConfig:
    """資産保全モード（2026-07-17 Study100/101帰結・Entry Freeze Mode）。
    enabled=True で新規BUYを全面停止（SELL/exit/signal generationは無停止）。
    環境変数 ENTRY_FREEZE_ENABLED（"1"/"true"→True, "0"/"false"→False）が
    yaml設定より優先される（rollback含む緊急上書き用）。
    """
    enabled: bool = False
    reason: str = "Research Freeze"


def resolve_entry_freeze(cfg_enabled: bool, cfg_reason: str) -> tuple[bool, str]:
    """env var ENTRY_FREEZE_ENABLED があればyaml設定より優先して返す（緊急上書き用）。"""
    raw = os.environ.get("ENTRY_FREEZE_ENABLED")
    if raw is None:
        return cfg_enabled, cfg_reason
    enabled = raw.strip().lower() in ("1", "true", "yes", "on")
    if enabled and not cfg_enabled:
        return True, "Research Freeze (env override)"
    return enabled, cfg_reason


@dataclass(frozen=True)
class LiveExecutionConfig:
    shadow_mode: bool = True
    shadow_mode_reason: str = "Initial live validation: signal→universe→risk→execution pipeline check"


@dataclass(frozen=True)
class CapitalScalingConfig:
    effective_capital_growth_limit_daily: float = 0.015  # 1.5%/day max growth
    cash_buffer_ratio: float = 0.10
    volatility_scalar_enabled: bool = True
    volatility_scalar_min: float = 0.50
    liquidity_scalar_enabled: bool = True
    liquidity_scalar_min: float = 0.50
    execution_scalar_enabled: bool = True
    execution_scalar_min: float = 0.70
    max_participation_rate: float = 0.05
    min_expected_fill_ratio: float = 0.80
    max_lot_cost_ratio: float = 0.30
    shadow_capital_tiers: tuple = (1, 2, 5)
    # freeze thresholds
    freeze_slippage_bps_max: float = 50.0
    freeze_reject_rate_max: float = 0.30
    freeze_fill_ratio_min: float = 0.70
    freeze_liquidity_stress_max: float = 0.70


@dataclass(frozen=True)
class AdaptiveGrowthConfig:
    # Aggression EMA parameters
    aggression_expand_alpha: float = 0.10   # slow expansion
    aggression_contract_alpha: float = 0.40  # fast contraction
    aggression_min: float = 0.30
    aggression_max: float = 1.40
    half_kelly_cap: float = 0.50
    # Edge persistence windows (days)
    edge_window_short: int = 5
    edge_window_med: int = 21
    edge_window_long: int = 63
    # Deployment state thresholds
    enter_aggressive_min_periods: int = 3   # consecutive days required
    enter_opportunistic_min_periods: int = 2
    exit_aggressive_threshold: float = 0.75
    enter_defensive_threshold: float = 0.45
    # Alpha-after-impact gate
    min_net_alpha_threshold_bps: float = 5.0
    # Exploration budget
    exploration_fraction: float = 0.05     # 5% of effective_capital
    # Reflexivity detection
    reflexivity_window: int = 10
    # Multi-horizon EMA rates
    multihorizon_alpha_execution: float = 0.40
    multihorizon_alpha_signal: float = 0.15
    multihorizon_alpha_regime: float = 0.06
    multihorizon_alpha_strategic: float = 0.02


@dataclass(frozen=True)
class StrategyConfig:
    fujiko: FujikoConfig
    mean_reversion: MeanReversionConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    risk_controls: RiskControlsConfig
    live_execution: LiveExecutionConfig = field(default_factory=LiveExecutionConfig)
    capital_scaling: CapitalScalingConfig = field(default_factory=CapitalScalingConfig)
    adaptive_growth: AdaptiveGrowthConfig = field(default_factory=AdaptiveGrowthConfig)
    entry_freeze: EntryFreezeConfig = field(default_factory=EntryFreezeConfig)


def _load_yaml() -> dict:
    if not STRATEGY_CONFIG_FILE.exists():
        raise RuntimeError(f"戦略設定ファイルが見つかりません: {STRATEGY_CONFIG_FILE}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML が見つかりません。`pip install pyyaml` を実行してください。") from exc

    try:
        data = yaml.safe_load(STRATEGY_CONFIG_FILE.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"strategy.yaml の読み込みに失敗しました: {STRATEGY_CONFIG_FILE}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"strategy.yaml の形式が不正です: {STRATEGY_CONFIG_FILE}")
    return data


def _require(section: dict, path: str):
    if path not in section:
        raise RuntimeError(f"strategy.yaml の {path} が未設定です。")
    return section[path]


def _parse_sector_concentration(raw: dict) -> SectorConcentrationConfig:
    sc = raw.get("sector_concentration", {}) if isinstance(raw, dict) else {}
    if not isinstance(sc, dict):
        return SectorConcentrationConfig()
    return SectorConcentrationConfig(
        enabled=bool(sc.get("enabled", True)),
        max_names_per_sector=int(sc.get("max_names_per_sector", 1)),
        max_weight_per_sector=float(sc.get("max_weight_per_sector", 0.12)),
    )


def _parse_bear_universe_filter(raw: dict) -> BearUniverseFilterConfig:
    buf = raw.get("bear_universe_filter", {}) if isinstance(raw, dict) else {}
    if not isinstance(buf, dict):
        return BearUniverseFilterConfig()
    excluded = buf.get("excluded_sectors", list(BearUniverseFilterConfig.excluded_sectors))
    return BearUniverseFilterConfig(
        enabled=bool(buf.get("enabled", True)),
        lookback_days=int(buf.get("lookback_days", 60)),
        ma200_below_days=int(buf.get("ma200_below_days", 40)),
        excluded_sectors=tuple(excluded) if excluded else BearUniverseFilterConfig.excluded_sectors,
    )


def _parse_risk_controls(rc: dict, data: dict | None = None) -> RiskControlsConfig:
    return RiskControlsConfig(
        shock_exit_mode=str(rc.get("shock_exit_mode", "full_exit")),
        regime_sizing=str(rc.get("regime_sizing", "none")),
        bear_scale=float(rc.get("bear_scale", 1.0)),
        dynamic_cap=bool(rc.get("dynamic_cap", False)),
        symbol_cap=float(rc.get("symbol_cap", 0.08)),
        sector_cap=float(rc.get("sector_cap", 0.25)),
        cluster_cap=float(rc.get("cluster_cap", 0.35)),
        bear_sector_cap=float(rc.get("bear_sector_cap", 0.18)),
        bear_cluster_cap=float(rc.get("bear_cluster_cap", 0.25)),
        gross_exposure_enabled=bool(rc.get("gross_exposure_enabled", True)),
        gross_cap_normal=float(rc.get("gross_cap_normal", 1.0)),
        gross_cap_drawdown_5pct=float(rc.get("gross_cap_drawdown_5pct", 0.6)),
        gross_cap_drawdown_8pct=float(rc.get("gross_cap_drawdown_8pct", 0.4)),
        bear_universe_filter=_parse_bear_universe_filter(data if data is not None else rc),
        sector_concentration=_parse_sector_concentration(rc),
    )


@lru_cache(maxsize=1)
def load_strategy_config() -> StrategyConfig:
    data = _load_yaml()

    fujiko = data.get("fujiko")
    mean_reversion = data.get("mean_reversion")
    portfolio = data.get("portfolio")
    risk = data.get("risk")

    if not isinstance(fujiko, dict):
        raise RuntimeError("strategy.yaml の fujiko セクションが不正です。")
    if not isinstance(mean_reversion, dict):
        raise RuntimeError("strategy.yaml の mean_reversion セクションが不正です。")
    if not isinstance(portfolio, dict):
        raise RuntimeError("strategy.yaml の portfolio セクションが不正です。")
    if not isinstance(risk, dict):
        raise RuntimeError("strategy.yaml の risk セクションが不正です。")

    risk_controls_raw = data.get("risk_controls", {})

    ef_raw = data.get("entry_freeze", {})
    if not isinstance(ef_raw, dict):
        ef_raw = {}
    _ef_enabled, _ef_reason = resolve_entry_freeze(
        bool(ef_raw.get("enabled", False)), str(ef_raw.get("reason", "Research Freeze")),
    )
    entry_freeze = EntryFreezeConfig(enabled=_ef_enabled, reason=_ef_reason)

    le_raw = data.get("live_execution", {})
    if not isinstance(le_raw, dict):
        le_raw = {}
    live_execution = LiveExecutionConfig(
        shadow_mode=bool(le_raw.get("shadow_mode", True)),
        shadow_mode_reason=str(le_raw.get(
            "shadow_mode_reason",
            "Initial live validation: signal→universe→risk→execution pipeline check",
        )),
    )

    cs_raw = data.get("capital_scaling", {})
    if not isinstance(cs_raw, dict):
        cs_raw = {}
    tiers_raw = cs_raw.get("shadow_capital_tiers", [1, 2, 5])
    capital_scaling = CapitalScalingConfig(
        effective_capital_growth_limit_daily=float(cs_raw.get("effective_capital_growth_limit_daily", 0.015)),
        cash_buffer_ratio=float(cs_raw.get("cash_buffer_ratio", 0.10)),
        volatility_scalar_enabled=bool(cs_raw.get("volatility_scalar_enabled", True)),
        volatility_scalar_min=float(cs_raw.get("volatility_scalar_min", 0.50)),
        liquidity_scalar_enabled=bool(cs_raw.get("liquidity_scalar_enabled", True)),
        liquidity_scalar_min=float(cs_raw.get("liquidity_scalar_min", 0.50)),
        execution_scalar_enabled=bool(cs_raw.get("execution_scalar_enabled", True)),
        execution_scalar_min=float(cs_raw.get("execution_scalar_min", 0.70)),
        max_participation_rate=float(cs_raw.get("max_participation_rate", 0.05)),
        min_expected_fill_ratio=float(cs_raw.get("min_expected_fill_ratio", 0.80)),
        max_lot_cost_ratio=float(cs_raw.get("max_lot_cost_ratio", 0.30)),
        shadow_capital_tiers=tuple(int(x) for x in tiers_raw),
        freeze_slippage_bps_max=float(cs_raw.get("freeze_slippage_bps_max", 50.0)),
        freeze_reject_rate_max=float(cs_raw.get("freeze_reject_rate_max", 0.30)),
        freeze_fill_ratio_min=float(cs_raw.get("freeze_fill_ratio_min", 0.70)),
        freeze_liquidity_stress_max=float(cs_raw.get("freeze_liquidity_stress_max", 0.70)),
    )

    ag_raw = data.get("adaptive_growth", {})
    if not isinstance(ag_raw, dict):
        ag_raw = {}
    adaptive_growth = AdaptiveGrowthConfig(
        aggression_expand_alpha=float(ag_raw.get("aggression_expand_alpha", 0.10)),
        aggression_contract_alpha=float(ag_raw.get("aggression_contract_alpha", 0.40)),
        aggression_min=float(ag_raw.get("aggression_min", 0.30)),
        aggression_max=float(ag_raw.get("aggression_max", 1.40)),
        half_kelly_cap=float(ag_raw.get("half_kelly_cap", 0.50)),
        edge_window_short=int(ag_raw.get("edge_window_short", 5)),
        edge_window_med=int(ag_raw.get("edge_window_med", 21)),
        edge_window_long=int(ag_raw.get("edge_window_long", 63)),
        enter_aggressive_min_periods=int(ag_raw.get("enter_aggressive_min_periods", 3)),
        enter_opportunistic_min_periods=int(ag_raw.get("enter_opportunistic_min_periods", 2)),
        exit_aggressive_threshold=float(ag_raw.get("exit_aggressive_threshold", 0.75)),
        enter_defensive_threshold=float(ag_raw.get("enter_defensive_threshold", 0.45)),
        min_net_alpha_threshold_bps=float(ag_raw.get("min_net_alpha_threshold_bps", 5.0)),
        exploration_fraction=float(ag_raw.get("exploration_fraction", 0.05)),
        reflexivity_window=int(ag_raw.get("reflexivity_window", 10)),
        multihorizon_alpha_execution=float(ag_raw.get("multihorizon_alpha_execution", 0.40)),
        multihorizon_alpha_signal=float(ag_raw.get("multihorizon_alpha_signal", 0.15)),
        multihorizon_alpha_regime=float(ag_raw.get("multihorizon_alpha_regime", 0.06)),
        multihorizon_alpha_strategic=float(ag_raw.get("multihorizon_alpha_strategic", 0.02)),
    )

    _min_rsr_val = float(_require(fujiko, "min_rsr"))
    return StrategyConfig(
        fujiko=FujikoConfig(
            min_sepa=int(_require(fujiko, "min_sepa")),
            min_rsr=_min_rsr_val,
            rsr_exit=float(fujiko.get("rsr_exit", _min_rsr_val)),  # fallback=min_rsr (fail-safe)
            mom_period=int(_require(fujiko, "mom_period")),
            turtle_entry=int(_require(fujiko, "turtle_entry")),
            turtle_exit=int(_require(fujiko, "turtle_exit")),
            use_turtle_entry=bool(_require(fujiko, "use_turtle_entry")),
        ),
        mean_reversion=MeanReversionConfig(
            rsi_period=int(_require(mean_reversion, "rsi_period")),
            rsi_entry=float(_require(mean_reversion, "rsi_entry")),
            rsi_exit=float(_require(mean_reversion, "rsi_exit")),
            ma_long=int(_require(mean_reversion, "ma_long")),
            stop_loss_pct=float(_require(mean_reversion, "stop_loss_pct")),
            max_hold_days=int(_require(mean_reversion, "max_hold_days")),
        ),
        portfolio=PortfolioConfig(
            capital=int(_require(portfolio, "capital")),
            max_positions=int(_require(portfolio, "max_positions")),
            min_sectors=int(_require(portfolio, "min_sectors")),
            max_single_weight=float(_require(portfolio, "max_single_weight")),
            max_dd_limit=float(_require(portfolio, "max_dd_limit")),
            vol_target=float(_require(portfolio, "vol_target")),
            use_idm=bool(_require(portfolio, "use_idm")),
        ),
        risk=RiskConfig(
            min_hold_days=int(_require(risk, "min_hold_days")),
            max_hold_days=int(_require(risk, "max_hold_days")),
            emergency_exit_pct=float(_require(risk, "emergency_exit_pct")),
            rebalance_interval=int(_require(risk, "rebalance_interval")),
            reentry_cooldown_days=int(risk.get("reentry_cooldown_days", 5)),
        ),
        risk_controls=_parse_risk_controls(risk_controls_raw, data=data),
        live_execution=live_execution,
        capital_scaling=capital_scaling,
        adaptive_growth=adaptive_growth,
        entry_freeze=entry_freeze,
    )
