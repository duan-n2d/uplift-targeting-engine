"""Download, convert and split the Criteo Uplift dataset."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ute import config

SCHEMA = {
    **{f: pl.Float32 for f in config.FEATURES},
    "treatment": pl.Int8,
    "conversion": pl.Int8,
    "visit": pl.Int8,
    "exposure": pl.Int8,
}


def download(dest: Path = config.RAW_CSV, url: str = config.DATA_URL) -> Path:
    """Fetch the gzipped CSV (about 300 MB) unless it is already on disk."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
    tmp.replace(dest)
    return dest


def to_parquet(src: Path = config.RAW_CSV, dest: Path = config.PARQUET) -> Path:
    """Convert the CSV to a compact parquet file (float32 features, int8 flags)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(src, schema_overrides=SCHEMA)
    df.write_parquet(dest, compression="zstd")
    return dest


@dataclass(frozen=True)
class Frame:
    """Numpy view of one split. ``extra`` holds post-treatment columns for the leakage study."""

    X: np.ndarray
    t: np.ndarray
    y: dict[str, np.ndarray]
    extra: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.t)

    def take(self, idx: np.ndarray) -> Frame:
        return Frame(
            X=self.X[idx],
            t=self.t[idx],
            y={k: v[idx] for k, v in self.y.items()},
            extra={k: v[idx] for k, v in self.extra.items()},
        )


def load(path: Path = config.PARQUET, sample_frac: float | None = None, seed: int = config.SEED):
    df = pl.read_parquet(path)
    if sample_frac is not None and sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=seed)
    return frame_from_polars(df)


def frame_from_polars(df: pl.DataFrame) -> Frame:
    return Frame(
        X=df.select(config.FEATURES).to_numpy().astype(np.float32, copy=False),
        t=df["treatment"].to_numpy().astype(np.int8, copy=False),
        y={o: df[o].to_numpy().astype(np.int8, copy=False) for o in config.OUTCOMES},
        extra={c: df[c].to_numpy().astype(np.int8, copy=False) for c in config.POST_TREATMENT},
    )


def split(
    frame: Frame, fractions=config.SPLIT, seed: int = config.SEED
) -> tuple[Frame, Frame, Frame]:
    """Train/valid/test split stratified by treatment x conversion.

    Conversion is rare (about 0.3%), so stratifying keeps the number of treated and control
    converters proportional in every split; otherwise the control-group converters in a split
    can swing enough to move the metrics.
    """
    train_f, valid_f, _ = fractions
    strata = frame.t.astype(np.int16) * 2 + frame.y["conversion"]
    idx = np.arange(len(frame))
    idx_train, idx_rest = train_test_split(
        idx, train_size=train_f, stratify=strata, random_state=seed
    )
    rel_valid = valid_f / (1.0 - train_f)
    idx_valid, idx_test = train_test_split(
        idx_rest, train_size=rel_valid, stratify=strata[idx_rest], random_state=seed + 1
    )
    return frame.take(idx_train), frame.take(idx_valid), frame.take(idx_test)
