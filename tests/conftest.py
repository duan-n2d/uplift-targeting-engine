"""A small synthetic RCT whose true individual uplift is known, shaped like the Criteo data:
12 features, 85% treated, binary outcome, and a segment where the treatment hurts."""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from ute import config
from ute.data import frame_from_polars

os.environ.setdefault("UTE_NO_MLFLOW", "1")  # tests never write to the local tracking store

FAST = {"n_estimators": 200, "learning_rate": 0.1, "min_child_samples": 50, "num_leaves": 15}


def make_rct(n: int = 100_000, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    t = (rng.random(n) < 0.85).astype(np.int8)
    base = 1 / (1 + np.exp(-(-2.0 + 1.5 * X[:, 0])))  # who would buy anyway
    tau = np.where(X[:, 1] > 0.5, 0.20, np.where(X[:, 1] < -1.0, -0.08, 0.02))  # true uplift
    tau = np.where(X[:, 0] > 1.0, 0.0, tau)  # "sure things": buy anyway, voucher changes nothing
    p = np.clip(base + t * tau, 0, 1)
    y = (rng.random(n) < p).astype(np.int8)
    visit = np.maximum(y, (rng.random(n) < 0.1).astype(np.int8))
    exposure = (t * (rng.random(n) < 0.6)).astype(np.int8)
    cols = {f: X[:, i] for i, f in enumerate(config.FEATURES)}
    df = pl.DataFrame(
        {**cols, "treatment": t, "conversion": y, "visit": visit, "exposure": exposure}
    )
    return df, tau


@pytest.fixture(scope="session")
def rct():
    df, tau = make_rct()
    return frame_from_polars(df), tau
