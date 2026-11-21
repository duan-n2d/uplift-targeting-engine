"""Batch scoring API: features in, uplift score and a targeting decision out.

The decision uses a *population* threshold (the score quantile measured on the validation
split), so a customer's decision does not depend on who else is in the request.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ute import config

MAX_ROWS = 50_000


class Customer(BaseModel):
    id: str | None = None
    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float
    f11: float


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1)
    budget: float = Field(0.2, gt=0.0, le=1.0, description="share of the base to target")

    @field_validator("customers")
    @classmethod
    def _limit(cls, v):
        if len(v) > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} customers per request")
        return v


class Scored(BaseModel):
    id: str | None
    uplift: float
    target: bool


class ScoreResponse(BaseModel):
    model: str
    budget: float
    threshold: float
    targeted: int
    results: list[Scored]


def model_path() -> Path:
    return Path(os.environ.get("UTE_MODEL", config.MODELS_DIR / "production.joblib"))


@lru_cache(maxsize=1)
def load_bundle(path: str) -> dict:
    return joblib.load(path)


def threshold_for(bundle: dict, budget: float) -> float:
    """Score above which a customer falls in the top ``budget`` share of the validation set."""
    q = round(1.0 - budget, 2)
    return bundle["score_quantiles"][str(q)]


app = FastAPI(title="Uplift Targeting Engine", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model_path().exists()}


@app.get("/model")
def model_info() -> dict:
    b = _bundle()
    return {
        "name": b["name"],
        "outcome": b["outcome"],
        "features": b["features"],
        "created_at": b["created_at"],
        "train_rows": b["train_rows"],
    }


def _bundle() -> dict:
    p = model_path()
    if not p.exists():
        raise HTTPException(503, f"model not found at {p}; run `ute train` first")
    return load_bundle(str(p))


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    b = _bundle()
    X = np.array([[getattr(c, f) for f in b["features"]] for c in req.customers], np.float32)
    s = b["learner"].predict_uplift(X)
    thr = threshold_for(b, req.budget)
    target = s >= thr
    return ScoreResponse(
        model=b["name"],
        budget=req.budget,
        threshold=thr,
        targeted=int(target.sum()),
        results=[
            Scored(id=c.id, uplift=float(v), target=bool(d))
            for c, v, d in zip(req.customers, s, target, strict=True)
        ],
    )
