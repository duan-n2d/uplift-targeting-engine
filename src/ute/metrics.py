"""Uplift evaluation: Qini and uplift curves, normalised areas, uplift@k, bootstrap intervals.

Definitions follow scikit-uplift (Diemert et al. use the same ones for this dataset), so the
numbers are comparable with published results:

* Rank customers by predicted uplift, highest first. Tied scores form one block.
* Qini curve      Q(n) = Y_t(n) - Y_c(n) * N_t(n) / N_c(n)
* Uplift curve    U(n) = (Y_t(n)/N_t(n) - Y_c(n)/N_c(n)) * (N_t(n) + N_c(n))
  where Y_* is the number of converters and N_* the number of people in the top n.
* Qini coefficient / AUUC = (area(model) - area(random)) / (area(perfect) - area(random)),
  random being the straight line to the curve's end point. 1 is a perfect ranking, 0 random.
* uplift@k = conversion rate of treated minus control inside the top k share of the ranking.

Confidence intervals use the Poisson bootstrap (each row gets a Poisson(1) weight): the ranking
is computed once and every replicate only reweights the cumulative sums, so 200 replicates on
millions of rows take seconds instead of minutes. It approximates the ordinary bootstrap well
for large n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b != 0)
    return out


@dataclass
class _Ranked:
    """Rows sorted by a score with the end index of every tie block."""

    y: np.ndarray
    t: np.ndarray
    ends: np.ndarray
    order: np.ndarray

    @classmethod
    def build(cls, y: np.ndarray, t: np.ndarray, score: np.ndarray) -> _Ranked:
        order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
        s = np.asarray(score)[order]
        ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
        return cls(
            y=np.asarray(y, np.float64)[order],
            t=np.asarray(t, np.float64)[order],
            ends=ends,
            order=order,
        )

    def cums(self, w: np.ndarray | None = None):
        wt = self.t if w is None else self.t * w[self.order]
        wc = (1.0 - self.t) if w is None else (1.0 - self.t) * w[self.order]
        n_t = np.cumsum(wt)[self.ends]
        n_c = np.cumsum(wc)[self.ends]
        y_t = np.cumsum(self.y * wt)[self.ends]
        y_c = np.cumsum(self.y * wc)[self.ends]
        return n_t, n_c, y_t, y_c


def _qini(n_t, n_c, y_t, y_c):
    x = np.r_[0.0, n_t + n_c]
    q = np.r_[0.0, y_t - _divide(y_c * n_t, n_c)]
    return x, q


def _uplift(n_t, n_c, y_t, y_c):
    x = np.r_[0.0, n_t + n_c]
    u = np.r_[0.0, (_divide(y_t, n_t) - _divide(y_c, n_c)) * (n_t + n_c)]
    return x, u


def _norm_area(x, curve, x_p, curve_p) -> float:
    base = x_p[-1] * curve_p[-1] / 2.0
    perfect = np.trapezoid(curve_p, x_p) - base
    actual = np.trapezoid(curve, x) - base
    return float(actual / perfect) if perfect != 0 else float("nan")


def perfect_qini_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Treated converters first, control converters last (negative effects allowed)."""
    return y * t - y * (1 - t)


def perfect_uplift_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    control_conv = np.sum((y == 1) & (t == 0))
    treated_nonconv = np.sum((y == 0) & (t == 1))
    summand = y if control_conv > treated_nonconv else t
    return 2 * (y == t) + summand


def qini_curve(y, t, score):
    return _qini(*_Ranked.build(y, t, score).cums())


def uplift_curve(y, t, score):
    return _uplift(*_Ranked.build(y, t, score).cums())


def qini_coefficient(y, t, score) -> float:
    x, q = qini_curve(y, t, score)
    xp, qp = qini_curve(y, t, perfect_qini_score(y, t))
    return _norm_area(x, q, xp, qp)


def auuc(y, t, score) -> float:
    x, u = uplift_curve(y, t, score)
    xp, up = uplift_curve(y, t, perfect_uplift_score(y, t))
    return _norm_area(x, u, xp, up)


def _uplift_at_k(r: _Ranked, k: float, w: np.ndarray | None = None) -> float:
    n = int(len(r.y) * k)
    y, t = r.y[:n], r.t[:n]
    wk = np.ones(n) if w is None else w[r.order[:n]]
    treated = (t * wk).sum()
    control = ((1 - t) * wk).sum()
    if treated == 0 or control == 0:
        return float("nan")
    return float((y * t * wk).sum() / treated - (y * (1 - t) * wk).sum() / control)


def uplift_at_k(y, t, score, k: float) -> float:
    return _uplift_at_k(_Ranked.build(y, t, score), k)


def curve_on_grid(y, t, score, kind: str = "qini", points: int = 101) -> np.ndarray:
    """Curve sampled at evenly spaced population shares, scaled per row targeted.

    Returned values are incremental conversions *per customer in the evaluation population*,
    so they can be rescaled to any customer base in the simulator.
    """
    x, c = (qini_curve if kind == "qini" else uplift_curve)(y, t, score)
    n = x[-1]
    grid = np.linspace(0.0, 1.0, points)
    return np.interp(grid * n, x, c) / n


def evaluate(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    budgets=(0.1, 0.2, 0.3),
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Point estimates and 95% Poisson-bootstrap intervals for every headline metric."""
    y = np.asarray(y)
    t = np.asarray(t)
    model = _Ranked.build(y, t, score)
    pq = _Ranked.build(y, t, perfect_qini_score(y, t))
    pu = _Ranked.build(y, t, perfect_uplift_score(y, t))

    def one(w):
        mc, qc, uc = model.cums(w), pq.cums(w), pu.cums(w)
        out = {
            "qini": _norm_area(*_qini(*mc), *_qini(*qc)),
            "auuc": _norm_area(*_uplift(*mc), *_uplift(*uc)),
        }
        for k in budgets:
            out[f"uplift@{int(k * 100)}"] = _uplift_at_k(model, k, w)
        return out

    point = one(None)
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(n_boot):
        w = rng.poisson(1.0, size=len(y)).astype(np.float64)
        for k, v in one(w).items():
            reps[k].append(v)
    result = {}
    for k, v in point.items():
        arr = np.asarray(reps[k], dtype=np.float64)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
        result[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return result


def ate(y: np.ndarray, t: np.ndarray) -> float:
    """Average treatment effect (difference in conversion rates). Equals uplift@k for random
    targeting in expectation, for every k."""
    t = np.asarray(t).astype(bool)
    return float(np.mean(y[t]) - np.mean(y[~t]))


def decile_table(y: np.ndarray, t: np.ndarray, score: np.ndarray, bins: int = 10) -> list[dict]:
    """Observed conversion rates by predicted-uplift decile (decile 1 = highest scores)."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), start=1):
        yt, tt = y[idx], t[idx].astype(bool)
        cr_t = float(yt[tt].mean()) if tt.any() else float("nan")
        cr_c = float(yt[~tt].mean()) if (~tt).any() else float("nan")
        rows.append(
            {
                "decile": i,
                "n": int(len(idx)),
                "n_treated": int(tt.sum()),
                "n_control": int((~tt).sum()),
                "cr_treated": cr_t,
                "cr_control": cr_c,
                "uplift": cr_t - cr_c,
                "mean_score": float(np.mean(np.asarray(score)[idx])),
            }
        )
    return rows
