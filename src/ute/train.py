"""End-to-end experiment: split, fit every learner, evaluate on test, pick a production model on
valid, run the leakage study and the drift check, write small artifacts, log to MLflow."""

from __future__ import annotations

import json
import os
import platform
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ute import config, drift, metrics
from ute.data import Frame, load, split
from ute.learners import LEARNERS, UPLIFT_LEARNERS, ResponseModel


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _mlflow():
    if os.environ.get("UTE_NO_MLFLOW"):
        return None
    try:
        import mlflow
    except ImportError:  # tracking is optional
        return None

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
    )
    mlflow.set_experiment("uplift-targeting")
    return mlflow


def fit_and_score(name, train: Frame, valid: Frame, test: Frame, outcome, X_fn=None, params=None):
    X_fn = X_fn or (lambda f: f.X)
    learner = LEARNERS[name](params=params, seed=config.SEED)
    start = time.perf_counter()
    learner.fit(
        X_fn(train), train.t, train.y[outcome], valid=(X_fn(valid), valid.t, valid.y[outcome])
    )
    seconds = time.perf_counter() - start
    return learner, learner.predict_uplift(X_fn(valid)), learner.predict_uplift(X_fn(test)), seconds


def run(
    sample_frac: float | None = None,
    outcomes=config.OUTCOMES,
    learners=tuple(LEARNERS),
    n_boot: int = 200,
    params: dict | None = None,
    frame: Frame | None = None,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    models_dir: Path = config.MODELS_DIR,
    leakage: bool = True,
    secondary_fit_frac: float = 0.2,
) -> dict:
    """``secondary_fit_frac``: share of train/valid used to fit learners for outcomes other than
    conversion. Visit has 16x more positives, so early stopping runs far longer; it is the
    secondary outcome, so its learners are fitted on a random sample and still evaluated on the
    full test split."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if frame is None:
        _log(f"loading parquet (sample_frac={sample_frac})")
        frame = load(sample_frac=sample_frac)
    n_rows = len(frame)
    train, valid, test = split(frame)
    del frame
    _log(f"split rows train={len(train):,} valid={len(valid):,} test={len(test):,}")

    summary: dict = {
        "meta": {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows": n_rows,
            "sample_frac": sample_frac,
            "split": {"train": len(train), "valid": len(valid), "test": len(test)},
            "treatment_share": float(np.mean(test.t)),
            "n_boot": n_boot,
            "seed": config.SEED,
            "python": platform.python_version(),
        },
        "outcomes": {},
    }
    curves, deciles = [], []
    production = None

    mlflow = _mlflow()
    parent = mlflow.start_run(run_name="pipeline") if mlflow else nullcontext()
    with parent:
        if mlflow:
            mlflow.log_params({"rows": n_rows, "sample_frac": sample_frac, "n_boot": n_boot})

        for outcome in outcomes:
            y_test = test.y[outcome]
            info = {
                "base_rate_treated": float(y_test[test.t == 1].mean()),
                "base_rate_control": float(y_test[test.t == 0].mean()),
                "ate": metrics.ate(y_test, test.t),
                "models": {},
            }
            fitted = {}
            tr_o, va_o = train, valid
            if outcome != "conversion" and secondary_fit_frac < 1.0:
                rng = np.random.default_rng(config.SEED)
                tr_o = train.take(np.flatnonzero(rng.random(len(train)) < secondary_fit_frac))
                va_o = valid.take(np.flatnonzero(rng.random(len(valid)) < secondary_fit_frac))
            info["fit_rows"] = {"train": len(tr_o), "valid": len(va_o)}
            for name in learners:
                _log(f"{outcome}: fitting {name} on {len(tr_o):,} rows")
                learner, s_valid, s_test, secs = fit_and_score(
                    name, tr_o, va_o, test, outcome, params=params
                )
                res = metrics.evaluate(
                    y_test, test.t, s_test, budgets=config.BUDGETS, n_boot=n_boot, seed=config.SEED
                )
                res["valid_qini"] = metrics.qini_coefficient(va_o.y[outcome], va_o.t, s_valid)
                res["fit_seconds"] = round(secs, 1)
                res["n_trees"] = learner.n_trees()
                info["models"][name] = res
                fitted[name] = (learner, s_valid, s_test)
                for kind in ("qini", "uplift"):
                    for share, v in zip(
                        np.linspace(0, 1, 101),
                        metrics.curve_on_grid(y_test, test.t, s_test, kind),
                        strict=True,
                    ):
                        curves.append(
                            {
                                "outcome": outcome,
                                "model": name,
                                "kind": kind,
                                "share": round(float(share), 2),
                                "value": float(v),
                            }
                        )
                _log(
                    f"{outcome}: {name} qini={res['qini']['value']:.4f} "
                    f"uplift@20={res['uplift@20']['value']:.5f} ({secs:.0f}s)"
                )
                if mlflow:
                    with mlflow.start_run(run_name=f"{outcome}/{name}", nested=True):
                        mlflow.log_params({"outcome": outcome, "learner": name})
                        mlflow.log_metrics(
                            {
                                **{
                                    k.replace("@", "_at_"): v["value"]
                                    for k, v in res.items()
                                    if isinstance(v, dict) and "value" in v
                                },
                                "valid_qini": res["valid_qini"],
                                "fit_seconds": secs,
                            }
                        )

            # Production model: best *validation* Qini among the uplift learners, primary outcome.
            if outcome == "conversion":
                candidates = [n for n in learners if n in UPLIFT_LEARNERS]
                if candidates:
                    best = max(candidates, key=lambda n: info["models"][n]["valid_qini"])
                    info["production_model"] = best
                    learner, s_valid, s_test = fitted[best]
                    production = (best, learner, s_valid)
                    for row in metrics.decile_table(y_test, test.t, s_test):
                        deciles.append({"outcome": outcome, "model": best, **row})
                    if "visit" in test.y:
                        for row in metrics.decile_table(test.y["visit"], test.t, s_test):
                            deciles.append({"outcome": "visit (same scores)", "model": best, **row})
            summary["outcomes"][outcome] = info

        if leakage and "conversion" in outcomes:
            summary["leakage"] = leakage_study(train, valid, test, params, n_boot)

        summary["drift"] = drift_check(train, test)

        if production is not None:
            name, learner, s_valid = production
            bundle = {
                "name": name,
                "learner": learner,
                "features": config.FEATURES,
                "outcome": "conversion",
                # thresholds that reproduce a budget on the validation population
                "score_quantiles": {
                    str(q): float(v)
                    for q, v in zip(
                        np.round(np.linspace(0, 1, 101), 2),
                        np.quantile(s_valid, np.linspace(0, 1, 101)),
                        strict=True,
                    )
                },
                "created_at": summary["meta"]["created_at"],
                "train_rows": len(train),
            }
            joblib.dump(bundle, models_dir / "production.joblib", compress=3)
            summary["production_model"] = name

        summary["meta"]["total_seconds"] = round(time.perf_counter() - t0, 1)
        (artifacts_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(curves).to_csv(artifacts_dir / "curves.csv", index=False)
        pd.DataFrame(deciles).to_csv(artifacts_dir / "deciles.csv", index=False)
        pd.DataFrame(summary["drift"]["features"]).to_csv(artifacts_dir / "psi.csv", index=False)
        if mlflow:
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="artifacts")
    _log(f"done in {summary['meta']['total_seconds']}s")
    return summary


def leakage_study(train: Frame, valid: Frame, test: Frame, params, n_boot: int) -> dict:
    """Same learners with post-treatment columns added. Not a model to ship: at decision time
    nobody knows yet whether the customer will be exposed to or click the campaign."""

    def with_post(f: Frame) -> np.ndarray:
        extra = np.column_stack([f.extra[c] for c in config.POST_TREATMENT]).astype(np.float32)
        return np.hstack([f.X, extra])

    out: dict = {"post_treatment_columns": config.POST_TREATMENT, "outcome": "conversion"}
    y = test.y["conversion"]
    treated = test.t == 1
    for label, X_fn in (("clean", None), ("leaky", with_post)):
        _log(f"leakage study: {label}")
        _, _, s_t, _ = fit_and_score("t_learner", train, valid, test, "conversion", X_fn, params)
        resp = ResponseModel(params=params, seed=config.SEED)
        Xf = X_fn or (lambda f: f.X)
        resp.fit(
            Xf(train),
            train.t,
            train.y["conversion"],
            valid=(Xf(valid), valid.t, valid.y["conversion"]),
        )
        auc = roc_auc_score(y[treated], resp.predict_uplift(Xf(test))[treated])
        ev = metrics.evaluate(y, test.t, s_t, budgets=config.BUDGETS, n_boot=n_boot)
        out[label] = {"t_learner": ev, "response_model_auc_treated": float(auc)}
    return out


def drift_check(train: Frame, test: Frame, n: int = 1_000_000) -> dict:
    """PSI of the test split against training, plus a simulated shifted batch to show what an
    alert looks like. The shift is synthetic and labelled as such."""
    rng = np.random.default_rng(config.SEED)
    ref = train.X[rng.choice(len(train), min(n, len(train)), replace=False)]
    cur = test.X[rng.choice(len(test), min(n, len(test)), replace=False)]
    # Simulated audience-mix change (e.g. a campaign pulled from a different channel): customers
    # with f0 above its median are three times as likely to be in the batch. Shifting values
    # instead would be unrealistic here, because most features sit on a few repeated values.
    above = cur[:, 0] > np.median(ref[:, 0])
    keep = rng.random(len(cur)) < np.where(above, 0.75, 0.25)
    shifted = cur[keep]
    rows = drift.psi_table(ref, cur, config.FEATURES)
    sim = drift.psi_table(ref, shifted, config.FEATURES)
    for r, s in zip(rows, sim, strict=True):
        r["psi_simulated_shift"] = s["psi"]
        r["status_simulated_shift"] = s["status"]
    return {
        "reference": "train sample",
        "current": "test sample",
        "simulated_shift": "audience mix: customers with f0 above its median 3x as likely",
        "features": rows,
    }
