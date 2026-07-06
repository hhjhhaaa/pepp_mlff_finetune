from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SplitProfile:
    name: str
    method: str
    group_by: list[str]
    stratify_by: list[str]
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1


DEFAULT_SPLIT_PROFILES = {
    "interpolation": SplitProfile(
        name="interpolation",
        method="group_stratified",
        group_by=["parent_frame_id"],
        stratify_by=["density_g_cm3", "composition", "sample_type"],
    ),
    "extrapolation_density": SplitProfile(
        name="extrapolation_density",
        method="leave_one_group_out",
        group_by=["density_g_cm3"],
        stratify_by=[],
    ),
    "extrapolation_composition": SplitProfile(
        name="extrapolation_composition",
        method="leave_one_group_out",
        group_by=["composition"],
        stratify_by=[],
    ),
}


def _tuple_key(row: dict, keys: Iterable[str]) -> tuple:
    return tuple(row.get(key) for key in keys)


def split_records(records: list[dict], profile: SplitProfile) -> dict[str, list[dict]]:
    """Split records without frame-level leakage.

    `interpolation` performs a deterministic grouped split with a light
    stratification sort. Extrapolation profiles create one split per held-out
    group value and are represented as `test_<value>` keys alongside training
    records for that fold.
    """
    if profile.method == "leave_one_group_out":
        key = profile.group_by[0]
        result: dict[str, list[dict]] = {}
        values = sorted({record.get(key) for record in records})
        for value in values:
            label = str(value).replace("/", "_")
            result[f"train_without_{label}"] = [r for r in records if r.get(key) != value]
            result[f"test_{label}"] = [r for r in records if r.get(key) == value]
        return result

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[_tuple_key(record, profile.group_by)].append(record)

    def sort_key(item):
        _, group_records = item
        first = group_records[0]
        return _tuple_key(first, profile.stratify_by) + _tuple_key(first, profile.group_by)

    ordered_groups = [rows for _, rows in sorted(groups.items(), key=sort_key)]
    n_groups = len(ordered_groups)
    n_train = int(round(n_groups * profile.train_fraction))
    n_val = int(round(n_groups * profile.val_fraction))
    train_groups = ordered_groups[:n_train]
    val_groups = ordered_groups[n_train : n_train + n_val]
    test_groups = ordered_groups[n_train + n_val :]

    return {
        "train": [record for group in train_groups for record in group],
        "val": [record for group in val_groups for record in group],
        "test": [record for group in test_groups for record in group],
    }
