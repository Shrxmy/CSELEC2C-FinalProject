from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


TARGET = "success"
ID_COLUMN = "eventid"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "iyear", "imonth", "iday", "extended", "latitude", "longitude",
    "specificity", "vicinity", "doubtterr", "multiple", "suicide",
    "guncertain1", "individual", "claimed", "INT_LOG", "INT_IDEO",
    "INT_MISC", "INT_ANY",
]

CATEGORICAL_FEATURES = [
    "region_txt", "country_txt", "attacktype1_txt", "targtype1_txt",
    "targsubtype1_txt", "natlty1_txt", "weaptype1_txt", "weapsubtype1_txt",
]

LEAKAGE_COLUMNS_EXCLUDED = [
    "nkill", "nkillus", "nkillter", "nwound", "nwoundus", "nwoundte",
    "property", "propextent", "propextent_txt", "propvalue", "ransom",
    "ransomamt", "ransompaid", "hostkidoutcome", "hostkidoutcome_txt", "nreleased",
]


@dataclass
class ProjectPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    main_file: Path
    supplement_2021_file: Path
    cache_file: Path


def _resolve_required_file(raw_dir: Path, stems: list[str], label: str) -> Path:
    """Resolve a mandatory raw-data file from accepted stems and tabular extensions."""
    extensions = [".xlsx", ".xls", ".csv", ".parquet"]
    for stem in stems:
        for ext in extensions:
            candidate = raw_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    accepted = [f"{stem}{ext}" for stem in stems for ext in extensions]
    raise FileNotFoundError(
        f"Missing mandatory {label}. Place one of these files in {raw_dir}: {accepted}"
    )


def get_project_paths(root: str | Path | None = None) -> ProjectPaths:
    root = Path.cwd() if root is None else Path(root)
    if root.name.lower() == "notebooks":
        root = root.parent
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    models_dir = root / "models"
    figures_dir = root / "results" / "figures"
    tables_dir = root / "results" / "tables"
    for d in [raw_dir, processed_dir, models_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    main_file = _resolve_required_file(
        raw_dir,
        stems=["globalterrorismdb_0522dist", "globalterrorismdb", "gtd"],
        label="main GTD dataset",
    )
    supplement_2021_file = _resolve_required_file(
        raw_dir,
        stems=["globalterrorismdb_2021Jan-June_1222dist", "gtd_2021Jan-June", "gtd_2021"],
        label="2021 GTD supplement",
    )

    return ProjectPaths(
        root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        main_file=main_file,
        supplement_2021_file=supplement_2021_file,
        cache_file=processed_dir / "gtd_combined_cleaned.parquet",
    )


def requested_columns() -> list[str]:
    return list(dict.fromkeys([ID_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]))


def read_table(path: Path, usecols: list[str], nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required GTD file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        header = pd.read_excel(path, nrows=0).columns.tolist()
        available = [c for c in usecols if c in header]
        return pd.read_excel(path, usecols=available, nrows=nrows)
    if suffix == ".csv":
        header = pd.read_csv(path, nrows=0).columns.tolist()
        available = [c for c in usecols if c in header]
        return pd.read_csv(path, usecols=available, nrows=nrows, low_memory=False)
    if suffix == ".parquet":
        header = pd.read_parquet(path, columns=None).head(0).columns.tolist()
        frame = pd.read_parquet(path, columns=[c for c in usecols if c in header])
        return frame.head(nrows) if nrows else frame
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_gtd(paths: ProjectPaths, use_cache: bool = True, sample_nrows: int | None = None) -> pd.DataFrame:
    """Load mandatory main + mandatory 2021 supplement, clean, deduplicate, and cache."""
    if use_cache and sample_nrows is None and paths.cache_file.exists():
        return pd.read_parquet(paths.cache_file)

    usecols = requested_columns()
    main = read_table(paths.main_file, usecols, nrows=sample_nrows)
    supplement = read_table(paths.supplement_2021_file, usecols, nrows=None if sample_nrows is None else sample_nrows)
    main["source_file"] = paths.main_file.name
    supplement["source_file"] = paths.supplement_2021_file.name

    raw = pd.concat([main, supplement], ignore_index=True)
    required = [ID_COLUMN, "iyear", TARGET]
    missing_required = [c for c in required if c not in raw.columns]
    if missing_required:
        raise ValueError(f"GTD data missing required columns: {missing_required}")

    raw = raw.drop_duplicates(subset=ID_COLUMN, keep="first").reset_index(drop=True)
    raw = raw[raw[TARGET].isin([0, 1])].copy()

    for col in [c for c in NUMERIC_FEATURES if c in raw.columns]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in [c for c in CATEGORICAL_FEATURES if c in raw.columns]:
        raw[col] = raw[col].astype("string").fillna("Unknown")

    if sample_nrows is None:
        raw.to_parquet(paths.cache_file, index=False)
    return raw


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[numeric + categorical].copy()
    y = df[TARGET].astype(int).copy()
    return X, y, numeric, categorical


def temporal_split(X: pd.DataFrame, y: pd.Series, train_end: int = 2014, valid_end: int = 2017):
    train_mask = X["iyear"] <= train_end
    valid_mask = (X["iyear"] > train_end) & (X["iyear"] <= valid_end)
    test_mask = X["iyear"] > valid_end
    if min(train_mask.sum(), valid_mask.sum(), test_mask.sum()) == 0:
        raise ValueError("Temporal split produced an empty partition.")
    return (
        X.loc[train_mask].copy(), X.loc[valid_mask].copy(), X.loc[test_mask].copy(),
        y.loc[train_mask].copy(), y.loc[valid_mask].copy(), y.loc[test_mask].copy(),
    )


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", make_one_hot_encoder())]), categorical),
    ], remainder="drop", sparse_threshold=0.3)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=(128, 64), lr=1e-3, weight_decay=1e-4,
                 batch_size=512, max_epochs=30, patience=5, use_gpu=True,
                 random_state=42, verbose=True):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.verbose = verbose

    def _device(self):
        return torch.device("cuda" if self.use_gpu and torch and torch.cuda.is_available() else "cpu")

    def _build_network(self, input_dim):
        layers, prev = [], input_dim
        for h in self.hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.20)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _to_tensor(self, X_batch):
        if sparse.issparse(X_batch):
            X_batch = X_batch.toarray()
        return torch.as_tensor(X_batch, dtype=torch.float32, device=self.device_)

    def fit(self, X, y):
        if torch is None:
            raise ImportError("PyTorch is required for TorchMLPClassifier")
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.array([0, 1])
        self.device_ = self._device()
        self.device_name_ = str(self.device_)
        y_array = np.asarray(y).astype(np.float32)
        self.network_ = self._build_network(X.shape[1]).to(self.device_)

        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        valid_size = max(int(0.10 * len(idx)), 1)
        valid_idx, train_idx = idx[:valid_size], idx[valid_size:]
        pos = max(float((y_array[train_idx] == 1).sum()), 1.0)
        neg = max(float((y_array[train_idx] == 0).sum()), 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=self.device_))
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_state, best_loss, stale = None, float("inf"), 0

        for epoch in range(1, self.max_epochs + 1):
            self.network_.train()
            rng.shuffle(train_idx)
            losses = []
            for start in range(0, len(train_idx), self.batch_size):
                batch = train_idx[start:start + self.batch_size]
                xb = self._to_tensor(X[batch])
                yb = torch.as_tensor(y_array[batch], dtype=torch.float32, device=self.device_).view(-1, 1)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network_(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_loss = self._validation_loss(X, y_array, valid_idx, criterion)
            if self.verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"Torch NN epoch {epoch:02d} | train_loss={np.mean(losses):.4f} | valid_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.network_.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state:
            self.network_.load_state_dict(best_state)
            self.network_.to(self.device_)
        return self

    def _validation_loss(self, X, y_array, valid_idx, criterion):
        self.network_.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(valid_idx), self.batch_size):
                batch = valid_idx[start:start + self.batch_size]
                xb = self._to_tensor(X[batch])
                yb = torch.as_tensor(y_array[batch], dtype=torch.float32, device=self.device_).view(-1, 1)
                losses.append(float(criterion(self.network_(xb), yb).detach().cpu()))
        return float(np.mean(losses)) if losses else float("inf")

    def predict_proba(self, X):
        self.network_.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                logits = self.network_(self._to_tensor(X[start:start + self.batch_size]))
                probs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        probs = np.asarray(probs)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save_torch_state(self, path: Path):
        torch.save({
            "state_dict": self.network_.state_dict(),
            "hidden_layers": self.hidden_layers,
            "device": self.device_name_,
        }, path)


def build_models(numeric: list[str], categorical: list[str], y_train: pd.Series, use_gpu=True, random_state=RANDOM_STATE):
    pre = lambda: make_preprocessor(numeric, categorical)
    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", pre()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, solver="saga", random_state=random_state)),
        ])
    }
    if XGBClassifier is not None:
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        device = "cuda" if use_gpu and torch is not None and torch.cuda.is_available() else "cpu"
        models["xgboost"] = Pipeline([
            ("preprocessor", pre()),
            ("model", XGBClassifier(
                n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=neg / pos, random_state=random_state, n_jobs=-1,
                tree_method="hist", device=device,
            )),
        ])
    models["feedforward_torch_nn"] = Pipeline([
        ("preprocessor", pre()),
        ("model", TorchMLPClassifier(use_gpu=use_gpu, random_state=random_state, verbose=True)),
    ])
    return models


def train_models(models: dict, X_train, y_train, paths: ProjectPaths):
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        if name == "feedforward_torch_nn":
            joblib.dump(model.named_steps["preprocessor"], paths.models_dir / "feedforward_torch_preprocessor.joblib")
            model.named_steps["model"].save_torch_state(paths.models_dir / "feedforward_torch_nn.pt")
        else:
            joblib.dump(model, paths.models_dir / f"{name}.joblib")
    return models


def predict_proba_positive(model, X):
    return model.predict_proba(X)[:, 1]


def metrics_at_threshold(y_true, proba, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def robust_metrics(model_name, split_name, y_true, proba, threshold=0.5) -> dict:
    row = {
        "model": model_name,
        "split": split_name,
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision_success": average_precision_score(y_true, proba),
        "average_precision_failure": average_precision_score(1 - np.asarray(y_true), 1 - np.asarray(proba)),
        "brier_score": brier_score_loss(y_true, proba),
    }
    row.update(metrics_at_threshold(y_true, proba, threshold))
    row["mcc"] = matthews_corrcoef(np.asarray(y_true), (np.asarray(proba) >= threshold).astype(int))
    return row


def evaluate_models(models: dict, X_valid, y_valid, X_test, y_test) -> pd.DataFrame:
    rows = []
    for split, Xs, ys in [("validation", X_valid, y_valid), ("test", X_test, y_test)]:
        for name, model in models.items():
            proba = predict_proba_positive(model, Xs)
            rows.append(robust_metrics(name, split, ys, proba, 0.5))
    return pd.DataFrame(rows)


def threshold_simulation(models: dict, X_test, y_test, thresholds=(0.30, 0.40, 0.50, 0.60, 0.70)) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        proba = predict_proba_positive(model, X_test)
        for t in thresholds:
            row = {"model": name, "split": "test"}
            row.update(metrics_at_threshold(y_test, proba, t))
            rows.append(row)
    return pd.DataFrame(rows)


def select_validation_thresholds(models: dict, X_valid, y_valid, X_test, y_test, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.20, 0.81, 0.01), 2) if grid is None else grid
    rows = []
    for name, model in models.items():
        valid_proba = predict_proba_positive(model, X_valid)
        candidates = []
        for t in grid:
            r = {"model": name, "split": "validation"}
            r.update(metrics_at_threshold(y_valid, valid_proba, float(t)))
            candidates.append(r)
        best = pd.DataFrame(candidates).sort_values(["balanced_accuracy", "f1"], ascending=False).iloc[0]
        rows.append(best.to_dict())
        test_proba = predict_proba_positive(model, X_test)
        r = {"model": name, "split": "test_selected_validation_threshold"}
        r.update(metrics_at_threshold(y_test, test_proba, float(best["threshold"])))
        rows.append(r)
    return pd.DataFrame(rows)


def subgroup_error_analysis(model_name, model, X, y, subgroup_columns: Iterable[str], threshold=0.5, min_group_size=50) -> pd.DataFrame:
    rows = []
    proba = predict_proba_positive(model, X)
    pred = (proba >= threshold).astype(int)
    y_arr = np.asarray(y)
    for col in subgroup_columns:
        if col not in X.columns:
            continue
        frame = pd.DataFrame({"group": X[col].astype("string").fillna("Unknown"), "y": y_arr, "pred": pred})
        for group, g in frame.groupby("group", dropna=False):
            if len(g) < min_group_size:
                continue
            fp = int(((g.pred == 1) & (g.y == 0)).sum())
            fn = int(((g.pred == 0) & (g.y == 1)).sum())
            tp = int(((g.pred == 1) & (g.y == 1)).sum())
            tn = int(((g.pred == 0) & (g.y == 0)).sum())
            rows.append({
                "model": model_name, "subgroup_column": col, "group": group, "n": len(g),
                "base_success_rate": float(g.y.mean()),
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })
    return pd.DataFrame(rows)


def plot_roc_pr_curves(models, X_test, y_test, figures_dir: Path):
    from sklearn.metrics import precision_recall_curve, roc_curve
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        proba = predict_proba_positive(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, proba):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves_test.png", dpi=180)
    plt.show()

    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        proba = predict_proba_positive(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        plt.plot(recall, precision, label=f"{name} AP={average_precision_score(y_test, proba):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curves_test.png", dpi=180)
    plt.show()


def save_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df.to_csv(path, index=False)
    return df
