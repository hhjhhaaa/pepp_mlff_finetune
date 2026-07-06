from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StaticMetrics:
    energy_mae_per_atom: float | None = None
    energy_rmse_per_atom: float | None = None
    force_mae: float | None = None
    force_rmse: float | None = None


def evaluate_static(*args, **kwargs) -> StaticMetrics:
    """Reserve static energy/force error evaluation for MACE calculators."""
    raise NotImplementedError("Static validation is reserved for the next implementation step.")


def apply_validation_gates(metrics: dict, gates: dict) -> dict:
    """Apply configured validation gates to already computed metrics."""
    failures = []
    for gate, limit in gates.items():
        if gate.startswith("max_"):
            metric = gate.removeprefix("max_")
            if metric in metrics and metrics[metric] is not None and metrics[metric] > limit:
                failures.append(f"{metric}={metrics[metric]} exceeds {limit}")
    if gates.get("require_no_nan") and metrics.get("nan_count", 0):
        failures.append("NaN values were observed")
    return {"ok": not failures, "failures": failures}
