import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import FAST
from ute import api, drift, report, train


@pytest.fixture(scope="module")
def run_dir(rct, tmp_path_factory):
    frame, _ = rct
    root = tmp_path_factory.mktemp("run")
    summary = train.run(
        frame=frame,
        outcomes=("conversion",),
        learners=("random", "response", "t_learner", "x_learner"),
        n_boot=20,
        params=FAST,
        artifacts_dir=root / "artifacts",
        models_dir=root / "models",
    )
    return root, summary


def test_pipeline_writes_artifacts(run_dir):
    root, summary = run_dir
    for f in ("metrics.json", "curves.csv", "deciles.csv", "psi.csv"):
        assert (root / "artifacts" / f).exists()
    assert (root / "models" / "production.joblib").exists()
    assert summary["production_model"] in ("t_learner", "x_learner")
    m = json.loads((root / "artifacts" / "metrics.json").read_text())
    assert (
        m["leakage"]["leaky"]["response_model_auc_treated"]
        > (m["leakage"]["clean"]["response_model_auc_treated"])
    )


def test_report_renders(run_dir, tmp_path):
    root, summary = run_dir
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\ny\n")
    report.main(root / "artifacts" / "metrics.json", tmp_path / "docs", readme)
    text = readme.read_text()
    assert "old" not in text and "Production model" in text
    assert "Leakage study" in (tmp_path / "docs" / "results.md").read_text()


def test_api_scores_and_targets(run_dir, monkeypatch):
    root, _ = run_dir
    monkeypatch.setenv("UTE_MODEL", str(root / "models" / "production.joblib"))
    api.load_bundle.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is True
    rng = np.random.default_rng(0)
    customers = [
        {"id": str(i), **{f"f{j}": float(v) for j, v in enumerate(rng.normal(size=12))}}
        for i in range(200)
    ]
    r = client.post("/score", json={"customers": customers, "budget": 0.2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 200
    assert 10 < body["targeted"] < 90  # roughly 20% of a similar population
    # a customer's decision does not depend on the rest of the batch
    one = client.post("/score", json={"customers": customers[:1], "budget": 0.2}).json()
    assert one["results"][0]["target"] == body["results"][0]["target"]


def test_api_rejects_bad_budget(run_dir, monkeypatch):
    root, _ = run_dir
    monkeypatch.setenv("UTE_MODEL", str(root / "models" / "production.joblib"))
    client = TestClient(api.app)
    c = {f"f{j}": 0.0 for j in range(12)}
    assert client.post("/score", json={"customers": [c], "budget": 1.5}).status_code == 422


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50_000)
    assert drift.psi(a, rng.normal(size=50_000)) < 0.01
    assert drift.psi(a, rng.normal(loc=0.5, size=50_000)) > 0.1
