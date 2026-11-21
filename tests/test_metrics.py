import numpy as np
import pytest

from ute import metrics


def _toy(n=20_000, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    tau = np.where(np.arange(n) < n // 2, 0.3, 0.0)
    y = (rng.random(n) < 0.1 + t * tau).astype(int)
    return y, t, tau


def test_perfect_ranking_scores_one():
    y, t, _ = _toy()
    assert metrics.qini_coefficient(y, t, metrics.perfect_qini_score(y, t)) == pytest.approx(1.0)
    assert metrics.auuc(y, t, metrics.perfect_uplift_score(y, t)) == pytest.approx(1.0)


def test_random_ranking_scores_near_zero():
    y, t, _ = _toy()
    s = np.random.default_rng(3).random(len(y))
    assert abs(metrics.qini_coefficient(y, t, s)) < 0.05


def test_true_uplift_beats_reversed():
    y, t, tau = _toy()
    good = metrics.qini_coefficient(y, t, tau + 1e-9 * np.arange(len(tau))[::-1])
    bad = metrics.qini_coefficient(y, t, -tau)
    assert good > 0.1 > bad


def test_ties_are_one_block():
    """A constant score must give the random line, whatever the row order."""
    y, t, _ = _toy()
    x, q = metrics.qini_curve(y, t, np.zeros(len(y)))
    assert len(x) == 2
    assert metrics.qini_coefficient(y, t, np.zeros(len(y))) == pytest.approx(0.0)


def test_uplift_at_k_by_hand():
    # top 50%: rows 0-3 -> treated conv 2/2, control conv 0/2 -> uplift 1.0
    y = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    t = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    s = np.arange(8)[::-1]
    assert metrics.uplift_at_k(y, t, s, 0.5) == pytest.approx(1.0)
    assert metrics.uplift_at_k(y, t, s, 1.0) == pytest.approx(3 / 4 - 1 / 4)


def test_bootstrap_interval_contains_point():
    y, t, tau = _toy()
    res = metrics.evaluate(y, t, tau, n_boot=50, seed=0)
    for m in res.values():
        assert m["ci_low"] <= m["value"] <= m["ci_high"]


def test_ate_matches_full_uplift_at_k():
    y, t, _ = _toy()
    s = np.random.default_rng(0).random(len(y))
    assert metrics.uplift_at_k(y, t, s, 1.0) == pytest.approx(metrics.ate(y, t))


def test_matches_scikit_uplift_if_installed():
    sk = pytest.importorskip("sklift.metrics")
    y, t, tau = _toy()
    s = tau + np.random.default_rng(5).normal(0, 0.1, len(tau))
    assert metrics.qini_coefficient(y, t, s) == pytest.approx(sk.qini_auc_score(y, s, t), rel=1e-6)
    assert metrics.auuc(y, t, s) == pytest.approx(sk.uplift_auc_score(y, s, t), rel=1e-6)
    assert metrics.uplift_at_k(y, t, s, 0.3) == pytest.approx(
        sk.uplift_at_k(y, s, t, strategy="overall", k=0.3), rel=1e-6
    )
