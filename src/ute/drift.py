"""Population Stability Index per feature: a cheap check before scoring a new batch.

Rule of thumb used by credit-risk teams: PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 act.
Criteo features are heavily discrete (many repeated values), so bin edges are the unique
quantiles of the reference sample, not a fixed number of equal-width bins.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def bin_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    qs = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(qs)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = bin_edges(reference, bins)
    ref = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1)
    cur = np.bincount(np.searchsorted(edges, current, side="right"), minlength=len(edges) + 1)
    p = ref / ref.sum() + EPS
    q = cur / cur.sum() + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def status(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "watch"
    return "drift"


def psi_table(reference: np.ndarray, current: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for j, name in enumerate(names):
        v = psi(reference[:, j], current[:, j])
        rows.append({"feature": name, "psi": v, "status": status(v)})
    return rows
