import numpy as np
import pytest

from tests.conftest import FAST
from ute import metrics
from ute.data import split
from ute.learners import LEARNERS, UPLIFT_LEARNERS


@pytest.fixture(scope="module")
def splits(rct):
    frame, tau = rct
    train, valid, test = split(frame)
    # recover true tau for the test rows by matching on the unique float feature f2
    lookup = dict(zip(frame.X[:, 2].tolist(), tau.tolist(), strict=True))
    tau_test = np.array([lookup[v] for v in test.X[:, 2].tolist()])
    return train, valid, test, tau_test


@pytest.mark.parametrize("name", UPLIFT_LEARNERS)
def test_learner_recovers_true_uplift(splits, name):
    train, valid, test, tau_test = splits
    y = "conversion"
    learner = LEARNERS[name](params=FAST, seed=0)
    learner.fit(train.X, train.t, train.y[y], valid=(valid.X, valid.t, valid.y[y]))
    s = learner.predict_uplift(test.X)
    assert np.corrcoef(s, tau_test)[0, 1] > 0.6
    # Normalised Qini is capped by outcome noise: even ranking by the *true* uplift scores
    # about 0.12 here, so compare with that oracle instead of with 1.
    tie_break = 1e-9 * np.random.default_rng(0).random(len(tau_test))
    oracle = metrics.qini_coefficient(test.y[y], test.t, tau_test + tie_break)
    assert metrics.qini_coefficient(test.y[y], test.t, s) > 0.6 * oracle


def test_uplift_beats_response_model(splits):
    """The response model chases people who buy anyway (driven by f0), not the uplift (f1)."""
    train, valid, test, _ = splits
    y = "conversion"
    scores = {}
    for name in ("response", "t_learner"):
        m = LEARNERS[name](params=FAST, seed=0)
        m.fit(train.X, train.t, train.y[y], valid=(valid.X, valid.t, valid.y[y]))
        scores[name] = metrics.qini_coefficient(test.y[y], test.t, m.predict_uplift(test.X))
    assert scores["t_learner"] > scores["response"] + 0.05


def test_random_is_near_zero(splits):
    _, _, test, _ = splits
    s = LEARNERS["random"](seed=0).fit(None, None, None).predict_uplift(test.X)
    assert abs(metrics.qini_coefficient(test.y["conversion"], test.t, s)) < 0.1


def test_split_is_stratified(rct):
    frame, _ = rct
    train, valid, test = split(frame)
    assert len(train) + len(valid) + len(test) == len(frame)
    for part in (train, valid, test):
        assert part.t.mean() == pytest.approx(frame.t.mean(), abs=0.002)
        assert part.y["conversion"].mean() == pytest.approx(frame.y["conversion"].mean(), abs=0.002)
