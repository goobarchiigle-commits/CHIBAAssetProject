"""
bootstrap_report.py — Bootstrap integrity report generation and atomic persistence

Output path: runtime/bootstrap/bootstrap_report_YYYYMMDD.json
Atomic write:  tmp-file → os.replace (NTFS-atomic on Windows)
Schema version: 1
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bootstrap.initialization_graph import RuntimeInitializationGraph
from src.bootstrap.root_cause_compressor import RootCauseCompressor

SCHEMA_VERSION: int = 1


@dataclass
class BootstrapReport:
    schema_version:       int
    date:                 str
    run_id:               str
    timestamp:            str
    initialization_order: List[str]
    dependency_graph:     Dict[str, Any]
    phases:               Dict[str, Any]
    failed_phases:        List[str]
    blocked_phases:       List[str]
    root_cause_chain:     List[str]
    root_cause_summary:   str
    fail_fast_decision:   str             # "ABORT" | "CONTINUE"
    fail_fast_reason:     str
    operator_summary:     Dict[str, Any]
    contract_violations:  Dict[str, List[str]] = field(default_factory=dict)
    coherence_violations: List[str]            = field(default_factory=list)


def build_bootstrap_report(
    graph:                 RuntimeInitializationGraph,
    run_id:                str,
    compressor:            Optional[RootCauseCompressor]      = None,
    contract_violations:   Optional[Dict[str, List[str]]]    = None,
    coherence_violations:  Optional[List[str]]               = None,
    operator_safe_action:  Optional[str]                     = None,
    operator_required_fix: Optional[str]                     = None,
    now_utc:               Optional[datetime]                = None,
) -> BootstrapReport:
    """Build a BootstrapReport from the current graph state. Deterministic."""
    if compressor is None:
        compressor = RootCauseCompressor()
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    date_str = now_utc.strftime("%Y%m%d")
    ts_str   = now_utc.isoformat()

    failed  = graph.get_failed_phases()
    blocked = graph.get_blocked_phases()
    chain   = compressor.compress_chain(graph)
    summary = compressor.compress(graph)
    op_sum  = compressor.operator_summary(
        graph,
        safe_action=operator_safe_action,
        required_fix=operator_required_fix,
    )

    coh_viol = coherence_violations or []
    has_crit_coherence = any(":CRITICAL]" in v for v in coh_viol)

    if graph.has_critical_failures():
        decision = "ABORT"
        reason   = (
            f"Critical phase failures: {failed}. "
            f"Downstream blocked: {blocked}. "
            "Live execution cannot proceed."
        )
    elif has_crit_coherence:
        decision = "ABORT"
        reason   = "Critical state coherence violations detected during bootstrap."
    else:
        decision = "CONTINUE"
        reason   = "All critical phases initialized. Bootstrap integrity confirmed."

    graph_dump = graph.dump()

    return BootstrapReport(
        schema_version=SCHEMA_VERSION,
        date=date_str,
        run_id=run_id,
        timestamp=ts_str,
        initialization_order=graph_dump["topology_order"],
        dependency_graph={
            name: {
                "dependencies": info["dependencies"],
                "is_critical":  info["is_critical"],
            }
            for name, info in graph_dump["phases"].items()
        },
        phases=graph_dump["phases"],
        failed_phases=failed,
        blocked_phases=blocked,
        root_cause_chain=chain,
        root_cause_summary=summary,
        fail_fast_decision=decision,
        fail_fast_reason=reason,
        operator_summary=op_sum,
        contract_violations=contract_violations or {},
        coherence_violations=coh_viol,
    )


def write_bootstrap_report(
    report:     BootstrapReport,
    output_dir: Path,
) -> Path:
    """
    Atomic write: runtime/bootstrap/bootstrap_report_YYYYMMDD.json
    Uses tmp → os.replace (atomic on NTFS / Windows).
    Returns the final path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename    = f"bootstrap_report_{report.date}.json"
    final_path  = output_dir / filename
    tmp_path    = output_dir / f"{filename}.tmp"

    data     = asdict(report)
    json_str = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str)

    tmp_path.write_text(json_str, encoding="utf-8")
    os.replace(str(tmp_path), str(final_path))   # atomic on Windows NTFS

    return final_path


def run_bootstrap_validation(
    graph:             RuntimeInitializationGraph,
    run_id:            str,
    state:             Optional[Dict[str, Any]]  = None,
    broker_positions:  Optional[Dict[str, int]]  = None,
    runtime_positions: Optional[Dict[str, int]]  = None,
    output_dir:        Optional[Path]            = None,
    now_utc:           Optional[datetime]        = None,
) -> BootstrapReport:
    """
    Full bootstrap validation pipeline:
      1. Per-phase contract verification
      2. Cross-phase coherence validation
      3. Report construction
      4. Atomic write (when output_dir provided)

    Returns BootstrapReport. Does NOT raise — use report.fail_fast_decision.
    """
    from src.bootstrap.contract_verifier import InitializationContractVerifier
    from src.bootstrap.state_coherence_validator import StateCoherenceValidator

    s          = state or {}
    verifier   = InitializationContractVerifier()
    coherence  = StateCoherenceValidator()
    compressor = RootCauseCompressor()

    contract_viol  = verifier.verify_all(s)
    coherence_viol = coherence.validate_all(
        graph=graph,
        state=s,
        broker_positions=broker_positions,
        runtime_positions=runtime_positions,
    )

    report = build_bootstrap_report(
        graph=graph,
        run_id=run_id,
        compressor=compressor,
        contract_violations=contract_viol,
        coherence_violations=coherence_viol,
        now_utc=now_utc,
    )

    if output_dir is not None:
        write_bootstrap_report(report, output_dir)

    return report
