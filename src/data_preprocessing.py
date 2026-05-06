from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    TARGET,
)
from .utils import ensure_directories


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def load_gtd(data_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    usecols = list(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]))
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_path, usecols=usecols, nrows=nrows)
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path, usecols=usecols, nrows=nrows, low_memory=False)
    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path, columns=usecols)
    raise ValueError(f"Unsupported data format: {data_path.suffix}")


def clean_gtd(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = df.copy()
    present_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    present_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    df = df[df[TARGET].isin([0, 1])].copy()
    for col in present_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in present_categorical:
        df[col] = df[col].astype("string").fillna("Unknown")

    y = df[TARGET].astype(int)
    X = df[present_numeric + present_categorical]
    return X, y, present_numeric, present_categorical


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: int = 2014,
    valid_end: int = 2017,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    years = X["iyear"]
    train_mask = years <= train_end
    valid_mask = (years > train_end) & (years <= valid_end)
    test_mask = years > valid_end

    if not train_mask.any() or not valid_mask.any() or not test_mask.any():
        raise ValueError(
            "Temporal split produced an empty partition. Check the dataset year range."
        )

    return (
        X.loc[train_mask].copy(),
        X.loc[valid_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[train_mask].copy(),
        y.loc[valid_mask].copy(),
        y.loc[test_mask].copy(),
    )


def load_and_split_dataset(
    data_path: str | Path,
    nrows: int | None = None,
    train_end: int = 2014,
    valid_end: int = 2017,
) -> dict[str, Any]:
    raw = load_gtd(data_path, nrows=nrows)
    X, y, numeric_features, categorical_features = clean_gtd(raw)
    splits = temporal_split(X, y, train_end=train_end, valid_end=valid_end)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return {
        "splits": splits,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def save_processed_splits(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    ensure_directories(output_dir)
    for name, frame in {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
    }.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    for name, series in {"y_train": y_train, "y_valid": y_valid, "y_test": y_test}.items():
        series.to_frame(TARGET).to_csv(output_dir / f"{name}.csv", index=False)
    joblib.dump({"columns": list(X_train.columns)}, output_dir / "metadata.joblib")

