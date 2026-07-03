"""
src/governance
Repository governance enforcement layer.
"""
from src.governance.registry_guard import (
    RegistryGuard,
    StrategyNotRegisteredError,
    StrategyStatusError,
    DEFAULT_GUARD,
)
from src.governance.mode import (
    RuntimeMode,
    ModeViolationError,
    get_current_mode,
    assert_production_mode,
    assert_not_research_writing_live_state,
    assert_not_paper_mutating_capital,
)
from src.governance.state_guard import (
    get_state_path,
    assert_no_shared_capital,
)
from src.governance.config_snapshot import (
    create_snapshot,
    load_snapshot,
    list_snapshots,
)
from src.governance.determinism import (
    set_deterministic,
    require_seed,
    get_seed,
    reset_seed,
    seed_from_env,
    scan_for_datetime_now,
    SeedNotSetError,
)
from src.governance.clock import (
    ClockSource,
    ProductionClock,
    FixedClock,
    SequentialClock,
    get_clock,
    set_clock,
    reset_clock,
    now as clock_now,
)
from src.governance.experiment import (
    ExperimentManifest,
    ExperimentRegistry,
    ArtifactStore,
    ArtifactExistsError,
    ExperimentNotFoundError,
    make_experiment_id,
    build_manifest,
)

__all__ = [
    # registry guard
    "RegistryGuard", "StrategyNotRegisteredError", "StrategyStatusError", "DEFAULT_GUARD",
    # mode
    "RuntimeMode", "ModeViolationError", "get_current_mode",
    "assert_production_mode", "assert_not_research_writing_live_state",
    "assert_not_paper_mutating_capital",
    # state guard
    "get_state_path", "assert_no_shared_capital",
    # config snapshot
    "create_snapshot", "load_snapshot", "list_snapshots",
    # determinism
    "set_deterministic", "require_seed", "get_seed", "reset_seed",
    "seed_from_env", "scan_for_datetime_now", "SeedNotSetError",
    # clock
    "ClockSource", "ProductionClock", "FixedClock", "SequentialClock",
    "get_clock", "set_clock", "reset_clock", "clock_now",
    # experiment
    "ExperimentManifest", "ExperimentRegistry", "ArtifactStore",
    "ArtifactExistsError", "ExperimentNotFoundError",
    "make_experiment_id", "build_manifest",
]
