"""Uplift learners built on LightGBM.

Every learner exposes ``fit(X, t, y, valid=None)`` and ``predict_uplift(X)``. ``valid`` is an
optional ``(X, t, y)`` tuple used only for early stopping; the test split is never touched here.

The experiment is a randomised controlled trial, so the propensity P(t=1 | x) is a constant
(the treatment share) rather than a fitted model.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import lightgbm as lgb
import numpy as np


def _default_threads() -> int:
    """Half the cores, at most 8. OpenMP threads spin at barriers, so asking for every core on a
    machine that is also doing other work can make training 10x slower, not faster."""
    if env := os.environ.get("UTE_N_JOBS"):
        return int(env)
    return max(1, min(8, (os.cpu_count() or 2) // 2))


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "n_jobs": _default_threads(),
    "verbose": -1,
}
EARLY_STOPPING = 50


def _params(overrides: dict | None, seed: int) -> dict:
    p = {**DEFAULT_PARAMS, "random_state": seed}
    if overrides:
        p.update(overrides)
    return p


def _early_stopping_kwargs(model, valid) -> dict:
    """LightGBM 4.7 renamed ``eval_set`` to ``eval_X``/``eval_y``; support both."""
    if valid is None:
        return {}
    kw: dict = {"callbacks": [lgb.early_stopping(EARLY_STOPPING, verbose=False)]}
    if "eval_X" in inspect.signature(model.fit).parameters:
        kw.update(eval_X=(valid[0],), eval_y=(valid[1],))
    else:
        kw["eval_set"] = [valid]
    return kw


def _fit_classifier(X, y, valid, params):
    model = lgb.LGBMClassifier(**params)
    if valid is not None and len(np.unique(valid[1])) < 2:
        valid = None  # no positives in this validation slice: cannot early-stop on log loss
    model.fit(X, y, **_early_stopping_kwargs(model, valid))
    return model


def _fit_regressor(X, y, valid, params):
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, **_early_stopping_kwargs(model, valid))
    return model


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)


class UpliftLearner:
    name = "base"

    def __init__(self, params: dict | None = None, seed: int = 42):
        self.params = params
        self.seed = seed

    def fit(self, X, t, y, valid=None):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_uplift(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def n_trees(self) -> dict[str, int]:
        return {
            k: int(getattr(m, "best_iteration_", 0) or m.n_estimators)
            for k, m in vars(self).items()
            if isinstance(m, lgb.LGBMModel)
        }


class RandomTargeting(UpliftLearner):
    """Baseline: whoever the budget reaches first. Expected uplift@k equals the ATE."""

    name = "random"

    def fit(self, X, t, y, valid=None):
        return self

    def predict_uplift(self, X):
        return np.random.default_rng(self.seed).random(len(X))


class ResponseModel(UpliftLearner):
    """Baseline most CRM teams use: P(convert | x, treated). Targets likely buyers, which
    includes people who would have converted anyway."""

    name = "response"

    def fit(self, X, t, y, valid=None):
        m = t == 1
        v = None if valid is None else (valid[0][valid[1] == 1], valid[2][valid[1] == 1])
        self.model_ = _fit_classifier(X[m], y[m], v, _params(self.params, self.seed))
        return self

    def predict_uplift(self, X):
        return _proba(self.model_, X)


class SLearner(UpliftLearner):
    """One model with treatment as a feature; uplift = f(x, 1) - f(x, 0)."""

    name = "s_learner"

    @staticmethod
    def _with_t(X, t_value):
        col = np.full((len(X), 1), t_value, dtype=X.dtype)
        return np.hstack([X, col])

    def fit(self, X, t, y, valid=None):
        Xt = np.hstack([X, t.reshape(-1, 1).astype(X.dtype)])
        v = None
        if valid is not None:
            v = (np.hstack([valid[0], valid[1].reshape(-1, 1).astype(X.dtype)]), valid[2])
        self.model_ = _fit_classifier(Xt, y, v, _params(self.params, self.seed))
        return self

    def predict_uplift(self, X):
        return _proba(self.model_, self._with_t(X, 1)) - _proba(self.model_, self._with_t(X, 0))


class TLearner(UpliftLearner):
    """Separate outcome models for treated and control; uplift = mu1(x) - mu0(x)."""

    name = "t_learner"

    def fit(self, X, t, y, valid=None):
        p = _params(self.params, self.seed)
        for arm, attr in ((1, "mu1_"), (0, "mu0_")):
            m = t == arm
            v = None
            if valid is not None:
                vm = valid[1] == arm
                v = (valid[0][vm], valid[2][vm])
            setattr(self, attr, _fit_classifier(X[m], y[m], v, p))
        return self

    def predict_uplift(self, X):
        return _proba(self.mu1_, X) - _proba(self.mu0_, X)


class XLearner(UpliftLearner):
    """Kuenzel et al. (2019). Stage 1 is a T-learner; stage 2 regresses imputed individual
    effects in each arm and blends them with the propensity:
    tau(x) = e * tau0(x) + (1 - e) * tau1(x). Helps when one arm is much larger (here 85/15).
    """

    name = "x_learner"

    def fit(self, X, t, y, valid=None):
        stage1 = TLearner(self.params, self.seed).fit(X, t, y, valid)
        self.mu1_, self.mu0_ = stage1.mu1_, stage1.mu0_
        self.e_ = float(np.mean(t))

        def imputed(Xa, ta, ya):
            d1 = ya[ta == 1] - _proba(self.mu0_, Xa[ta == 1])
            d0 = _proba(self.mu1_, Xa[ta == 0]) - ya[ta == 0]
            return d1, d0

        d1, d0 = imputed(X, t, y)
        v1 = v0 = None
        if valid is not None:
            vd1, vd0 = imputed(*valid)
            v1 = (valid[0][valid[1] == 1], vd1)
            v0 = (valid[0][valid[1] == 0], vd0)
        p = _params(self.params, self.seed)
        self.tau1_ = _fit_regressor(X[t == 1], d1, v1, p)
        self.tau0_ = _fit_regressor(X[t == 0], d0, v0, p)
        return self

    def predict_uplift(self, X):
        tau1 = self.tau1_.predict(X).astype(np.float64)
        tau0 = self.tau0_.predict(X).astype(np.float64)
        return self.e_ * tau0 + (1.0 - self.e_) * tau1


class TransformedOutcome(UpliftLearner):
    """Class-transformation for unequal arms (Athey and Imbens): regress
    y * (t - e) / (e * (1 - e)), whose expectation given x is the uplift."""

    name = "transformed_outcome"

    def fit(self, X, t, y, valid=None):
        self.e_ = float(np.mean(t))

        def target(ta, ya):
            return ya * (ta - self.e_) / (self.e_ * (1.0 - self.e_))

        v = None if valid is None else (valid[0], target(valid[1], valid[2]))
        self.model_ = _fit_regressor(X, target(t, y), v, _params(self.params, self.seed))
        return self

    def predict_uplift(self, X):
        return self.model_.predict(X).astype(np.float64)


LEARNERS = {
    cls.name: cls
    for cls in (RandomTargeting, ResponseModel, SLearner, TLearner, XLearner, TransformedOutcome)
}
UPLIFT_LEARNERS = ("s_learner", "t_learner", "x_learner", "transformed_outcome")
