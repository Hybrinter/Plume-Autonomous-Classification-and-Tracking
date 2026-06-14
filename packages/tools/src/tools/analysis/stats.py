"""Per-signal summary statistics over a captured run.

Reduces a CaptureResult to one tidy stats row per emitted column. Numeric columns get the usual
descriptive set (count of non-NaN samples, NaN count, mean/std/min/max, first/last, sum, unique
count, and a transition count); categorical columns get count, unique count, the modal label, a
transition count, and first/last. The result is a single DataFrame so the report can render one
"summary stats" table per group. Stats are computed deterministically from the recorder output;
nothing here drives the SIL.

Contains:
  - summarize: reduce a CaptureResult to a tidy per-signal stats DataFrame.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# third-party
import pandas as pd

# internal
from tools.analysis.datapoints import REGISTRY, SignalKind, accumulable_names
from tools.analysis.recorder import CaptureResult

_CUMULATIVE_SUFFIX = ".cumulative"


def _fmt_scalar(value: float) -> str:
    """Format a numeric first/last sample as a compact string (uniform-typed for Parquet)."""
    if value != value:  # NaN
        return ""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


STATS_COLUMNS: tuple[str, ...] = (
    "signal",
    "group",
    "kind",
    "unit",
    "n",
    "n_nan",
    "mean",
    "std",
    "min",
    "max",
    "first",
    "last",
    "total",
    "n_unique",
    "n_transitions",
    "mode",
)


def _column_meta() -> dict[str, tuple[str, str, SignalKind]]:
    """Return name -> (group, unit, kind) for every emitted column, including cumulative ones."""
    meta: dict[str, tuple[str, str, SignalKind]] = {
        signal.name: (signal.group, signal.unit, signal.kind) for signal in REGISTRY
    }
    for name in accumulable_names():
        group, _unit, _kind = meta[name]
        meta[f"{name}{_CUMULATIVE_SUFFIX}"] = (group, "count", SignalKind.NUMERIC)
    return meta


def _numeric_stats(series: pd.Series) -> dict[str, object]:
    """Compute the descriptive stats for one numeric column (NaN entries are failed extractions)."""
    valid = series.dropna()
    n = int(valid.shape[0])
    n_nan = int(series.isna().sum())
    transitions = int((valid.to_numpy()[1:] != valid.to_numpy()[:-1]).sum()) if n > 1 else 0
    first = _fmt_scalar(float(valid.iloc[0])) if n else ""
    last = _fmt_scalar(float(valid.iloc[-1])) if n else ""
    return {
        "n": n,
        "n_nan": n_nan,
        "mean": float(valid.mean()) if n else float("nan"),
        "std": float(valid.std(ddof=0)) if n else float("nan"),
        "min": float(valid.min()) if n else float("nan"),
        "max": float(valid.max()) if n else float("nan"),
        "first": first,
        "last": last,
        "total": float(valid.sum()) if n else float("nan"),
        "n_unique": int(valid.nunique()),
        "n_transitions": transitions,
        "mode": "",
    }


def _categorical_stats(series: pd.Series) -> dict[str, object]:
    """Compute the descriptive stats for one categorical (label) column."""
    labels = series.astype("string")
    present = labels[labels.str.len() > 0]
    n = int(present.shape[0])
    values = labels.to_numpy()
    transitions = int((values[1:] != values[:-1]).sum()) if labels.shape[0] > 1 else 0
    modes = present.mode()
    mode = str(modes.iloc[0]) if not modes.empty else ""
    return {
        "n": n,
        "n_nan": int(labels.shape[0] - n),
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "first": str(values[0]) if labels.shape[0] else "",
        "last": str(values[-1]) if labels.shape[0] else "",
        "total": float("nan"),
        "n_unique": int(present.nunique()),
        "n_transitions": transitions,
        "mode": mode,
    }


def summarize(capture: CaptureResult) -> pd.DataFrame:
    """Reduce a captured run to one summary-stats row per emitted column.

    Args:
        capture: The CaptureResult whose per-group wide frames to summarize.

    Returns:
        A tidy DataFrame with one row per signal (registry order, grouped by app/bus), carrying
        the columns in STATS_COLUMNS. Numeric and categorical signals share the frame; the stats
        that do not apply to a kind are left as NaN ("") sentinels.

    Notes:
        Computed entirely from the recorder's wide frames, so it inherits the run's determinism.
    """
    meta = _column_meta()
    rows: list[dict[str, object]] = []
    for group, frame in capture.wide.items():
        for column in frame.columns:
            if column == "t":
                continue
            _group, unit, kind = meta[column]
            series = frame[column]
            stats = (
                _numeric_stats(series) if kind is SignalKind.NUMERIC else _categorical_stats(series)
            )
            rows.append(
                {"signal": column, "group": group, "kind": kind.value, "unit": unit, **stats}
            )
    return pd.DataFrame(rows, columns=list(STATS_COLUMNS))
