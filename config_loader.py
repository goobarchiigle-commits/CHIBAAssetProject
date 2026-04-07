from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from src.paths import STRATEGY_CONFIG_FILE


@dataclass(frozen=True)
class FujikoConfig:
    min_sepa: int
    min_rsr: float
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


@dataclass(frozen=True)
class LiveExecutionConfig:
    shadow_mode: bool = True
    shadow_mode_reason: str = "Initial live validation: signal→universe→risk→execution pipeline check"


@dataclass(frozen=True)
class StrategyConfig:
    fujiko: FujikoConfig
    mean_reversion: MeanReversionConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    risk_controls: RiskControlsConfig
    live_execution: LiveExecutionConfig = field(default_factory=LiveExecutionConfig)


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


def _parse_risk_controls(rc: dict) -> RiskControlsConfig:
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
        bear_universe_filter=_parse_bear_universe_filter(rc),
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

    return StrategyConfig(
        fujiko=FujikoConfig(
            min_sepa=int(_require(fujiko, "min_sepa")),
            min_rsr=float(_require(fujiko, "min_rsr")),
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
        ),
        risk_controls=_parse_risk_controls(risk_controls_raw),
        live_execution=live_execution,
    )
