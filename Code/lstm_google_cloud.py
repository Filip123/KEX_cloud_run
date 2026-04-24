import argparse
import json
import os
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential


warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


LOOKBACK = 20
MIN_TRAIN_MONTHS = 48
TEST_MONTHS = 3
VAL_MONTHS = 12

THESIS_LEARNING_RATE = 1e-4
THESIS_BATCH_SIZE = 64
THESIS_EPOCHS = 20
THESIS_PATIENCE = 5
THESIS_USE_CLASS_WEIGHT = False

SMALL_LSTM_CFG = {"lstm_units": 8, "dense_units": 4}
LARGE_LSTM_CFG = {"lstm_units": 64, "dense_units": 32}

THESIS_DROPOUT_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
THESIS_WEIGHT_DECAY_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
STRONG_DROPOUT_GRID = [0.60, 0.70, 0.80]
STRONG_WEIGHT_DECAY_GRID = [5e-2, 1e-1, 2e-1]


MODEL_SPECS = {
    "Small Unreg": {
        "runner": "unreg",
        "kwargs": SMALL_LSTM_CFG,
    },
    "Large Unreg": {
        "runner": "unreg",
        "kwargs": LARGE_LSTM_CFG,
    },
    "Large Dropout": {
        "runner": "tuned",
        "regularization": "dropout",
        "param_grid": THESIS_DROPOUT_GRID,
        "kwargs": LARGE_LSTM_CFG,
    },
    "Large Weight Decay": {
        "runner": "tuned",
        "regularization": "weight_decay",
        "param_grid": THESIS_WEIGHT_DECAY_GRID,
        "kwargs": LARGE_LSTM_CFG,
    },
    "Large Strong Dropout": {
        "runner": "tuned",
        "regularization": "dropout",
        "param_grid": STRONG_DROPOUT_GRID,
        "kwargs": LARGE_LSTM_CFG,
    },
    "Large Strong Weight Decay": {
        "runner": "tuned",
        "regularization": "weight_decay",
        "param_grid": STRONG_WEIGHT_DECAY_GRID,
        "kwargs": LARGE_LSTM_CFG,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clean LSTM seed-study workflow in a cloud-friendly script and "
            "save resumable artifacts under a chosen output folder."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/complete_dataset.csv"),
        help="Path to the complete dataset CSV.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/lstm_seed_study"),
        help="Directory where pickle/CSV/manifest artifacts will be stored.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="gcloud",
        help="Suffix used for the saved artifact file names.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated seed list, for example '3,4,5,6,7'.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Model selection. Use 'all', 'unreg', 'regularized', or a comma-separated "
            "list of exact model names."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, ignore existing saved combinations and rerun them.",
    )
    parser.add_argument(
        "--disable-class-weight",
        action="store_true",
        help="Disable class weighting even if you later change the default in code.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=THESIS_EPOCHS,
        help="Training epochs for each fold/model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=THESIS_BATCH_SIZE,
        help="Batch size.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=THESIS_PATIENCE,
        help="Early stopping patience.",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    df["month_id"] = df.index.year * 100 + df.index.month
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"target", "regime", "month_id"}
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq, y_seq, end_positions = [], [], []
    for end_idx in range(lookback - 1, len(X)):
        start_idx = end_idx - lookback + 1
        X_seq.append(X[start_idx:end_idx + 1])
        y_seq.append(y[end_idx])
        end_positions.append(end_idx)

    if len(X_seq) == 0:
        return (
            np.empty((0, lookback, X.shape[1]), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    return np.array(X_seq), np.array(y_seq), np.array(end_positions)


def fit_scaler_and_build_sequences(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
) -> dict[str, np.ndarray]:
    scaler = StandardScaler()

    X_train_raw = train_df[feature_cols].values
    y_train = train_df["target"].values.astype(int)

    X_test_raw = test_df[feature_cols].values
    y_test = test_df["target"].values.astype(int)

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_tr_seq, y_tr_seq, tr_end_pos = make_sequences(X_train_scaled, y_train, lookback)

    history_len = min(lookback - 1, len(X_train_scaled))
    X_test_aug = np.vstack([X_train_scaled[-history_len:], X_test_scaled])
    y_test_aug = np.concatenate([y_train[-history_len:], y_test])

    X_te_seq, y_te_seq_full, te_end_pos_full = make_sequences(X_test_aug, y_test_aug, lookback)

    test_mask = te_end_pos_full >= history_len
    X_te_seq = X_te_seq[test_mask]
    y_te_seq = y_te_seq_full[test_mask]
    te_end_pos = te_end_pos_full[test_mask] - history_len

    test_dates = test_df.index.values[te_end_pos]
    train_dates = train_df.index.values[tr_end_pos]

    return {
        "scaler": scaler,
        "X_tr_seq": X_tr_seq,
        "y_tr_seq": y_tr_seq,
        "train_dates": train_dates,
        "X_te_seq": X_te_seq,
        "y_te_seq": y_te_seq,
        "test_dates": test_dates,
    }


def build_lstm_model_thesis(
    n_steps: int,
    n_features: int,
    regularization: str | None = None,
    regularization_strength: float = 0.0,
    lstm_units: int = 16,
    dense_units: int = 8,
    learning_rate: float = THESIS_LEARNING_RATE,
) -> tf.keras.Model:
    if regularization not in {None, "dropout", "weight_decay"}:
        raise ValueError('regularization must be None, "dropout", or "weight_decay"')

    if regularization == "dropout":
        effective_dropout = regularization_strength
        weight_decay = 0.0
    elif regularization == "weight_decay":
        effective_dropout = 0.0
        weight_decay = regularization_strength
    else:
        effective_dropout = 0.0
        weight_decay = 0.0

    if weight_decay > 0:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(lstm_units, dropout=effective_dropout),
        Dense(dense_units, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def fit_lstm_model_thesis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    regularization: str | None = None,
    regularization_strength: float = 0.0,
    lstm_units: int = 16,
    dense_units: int = 8,
    learning_rate: float = THESIS_LEARNING_RATE,
    epochs: int = THESIS_EPOCHS,
    batch_size: int = THESIS_BATCH_SIZE,
    patience: int = THESIS_PATIENCE,
    use_class_weight: bool = THESIS_USE_CLASS_WEIGHT,
    verbose: int = 0,
) -> tf.keras.Model:
    model = build_lstm_model_thesis(
        n_steps=X_train.shape[1],
        n_features=X_train.shape[2],
        regularization=regularization,
        regularization_strength=regularization_strength,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
    )

    callbacks = []
    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=False,
    )

    if use_class_weight:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_train,
        )
        fit_kwargs["class_weight"] = {0: float(weights[0]), 1: float(weights[1])}

    if X_val is not None and len(X_val) > 0:
        callbacks.append(EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ))
        fit_kwargs["validation_data"] = (X_val, y_val)
        fit_kwargs["callbacks"] = callbacks
    else:
        callbacks.append(EarlyStopping(
            monitor="loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ))
        fit_kwargs["callbacks"] = callbacks

    model.fit(**fit_kwargs)
    return model


def build_prediction_split_df(
    dates: np.ndarray,
    label: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    best_param: float | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(dates)})
    out["window"] = label
    out["split"] = split_name
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["y_prob"] = y_prob
    if threshold is not None:
        out["threshold"] = threshold
    if best_param is not None:
        out["best_param"] = best_param
    return out


def walk_forward_thesis_unreg(
    df: pd.DataFrame,
    feature_cols: list[str],
    lstm_units: int,
    dense_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    use_class_weight: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_months = sorted(df["month_id"].unique())
    month_label = {m: f"{m//100}-{m%100:02d}" for m in unique_months}
    results, preds = [], []

    for i in range(MIN_TRAIN_MONTHS, len(unique_months) - TEST_MONTHS + 1, TEST_MONTHS):
        train_months = set(unique_months[:i])
        test_months_ = set(unique_months[i:i + TEST_MONTHS])

        train = df[df["month_id"].isin(train_months)].copy()
        test = df[df["month_id"].isin(test_months_)].copy()
        if len(test) == 0:
            continue

        split_idx = max(VAL_MONTHS + 1, i - VAL_MONTHS)
        inner_tr = df[df["month_id"].isin(set(unique_months[:split_idx]))].copy()
        inner_val = df[df["month_id"].isin(set(unique_months[split_idx:i]))].copy()

        inner_seq = fit_scaler_and_build_sequences(inner_tr, inner_val, feature_cols, LOOKBACK)
        Xi_tr, yi_tr = inner_seq["X_tr_seq"], inner_seq["y_tr_seq"]
        Xi_val, yi_val = inner_seq["X_te_seq"], inner_seq["y_te_seq"]

        full_seq = fit_scaler_and_build_sequences(train, test, feature_cols, LOOKBACK)
        X_tr, y_tr = full_seq["X_tr_seq"], full_seq["y_tr_seq"]
        X_te, y_te = full_seq["X_te_seq"], full_seq["y_te_seq"]

        if min(len(Xi_val), len(X_tr), len(X_te)) == 0:
            continue

        threshold = 0.5
        tf.keras.backend.clear_session()
        model = fit_lstm_model_thesis(
            X_tr,
            y_tr,
            regularization=None,
            regularization_strength=0.0,
            lstm_units=lstm_units,
            dense_units=dense_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            use_class_weight=use_class_weight,
            verbose=0,
        )

        pp_tr = model.predict(X_tr, verbose=0).ravel()
        pp_te = model.predict(X_te, verbose=0).ravel()
        yp_tr = (pp_tr >= threshold).astype(int)
        yp_te = (pp_te >= threshold).astype(int)

        label = f"{month_label[unique_months[i]]}–{month_label[unique_months[i + TEST_MONTHS - 1]]}"
        always_up_acc = float(np.mean(y_te == 1))

        results.append({
            "window": label,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "threshold": threshold,
            "always_up_acc": always_up_acc,
            "train_acc": accuracy_score(y_tr, yp_tr),
            "test_acc": accuracy_score(y_te, yp_te),
            "train_f1": f1_score(y_tr, yp_tr, zero_division=0),
            "test_f1": f1_score(y_te, yp_te, zero_division=0),
            "train_auc": safe_auc(y_tr, pp_tr),
            "test_auc": safe_auc(y_te, pp_te),
            "train_mcc": matthews_corrcoef(y_tr, yp_tr),
            "test_mcc": matthews_corrcoef(y_te, yp_te),
        })

        preds.append(pd.concat([
            build_prediction_split_df(full_seq["train_dates"], label, "train", y_tr, yp_tr, pp_tr, threshold=threshold),
            build_prediction_split_df(full_seq["test_dates"], label, "test", y_te, yp_te, pp_te, threshold=threshold),
        ], ignore_index=True))

    return pd.DataFrame(results), pd.concat(preds, ignore_index=True)


def walk_forward_thesis_tuned(
    df: pd.DataFrame,
    feature_cols: list[str],
    regularization: str,
    param_grid: list[float],
    lstm_units: int,
    dense_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    use_class_weight: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_months = sorted(df["month_id"].unique())
    month_label = {m: f"{m//100}-{m%100:02d}" for m in unique_months}
    results, preds = [], []

    for i in range(MIN_TRAIN_MONTHS, len(unique_months) - TEST_MONTHS + 1, TEST_MONTHS):
        train_months = set(unique_months[:i])
        test_months_ = set(unique_months[i:i + TEST_MONTHS])

        train = df[df["month_id"].isin(train_months)].copy()
        test = df[df["month_id"].isin(test_months_)].copy()
        if len(test) == 0:
            continue

        split_idx = max(VAL_MONTHS + 1, i - VAL_MONTHS)
        inner_tr = df[df["month_id"].isin(set(unique_months[:split_idx]))].copy()
        inner_val = df[df["month_id"].isin(set(unique_months[split_idx:i]))].copy()

        inner_seq = fit_scaler_and_build_sequences(inner_tr, inner_val, feature_cols, LOOKBACK)
        Xi_tr, yi_tr = inner_seq["X_tr_seq"], inner_seq["y_tr_seq"]
        Xi_val, yi_val = inner_seq["X_te_seq"], inner_seq["y_te_seq"]

        if len(Xi_tr) == 0 or len(Xi_val) == 0:
            continue

        best_param, best_auc = param_grid[0], -np.inf
        for param_value in param_grid:
            tf.keras.backend.clear_session()
            model = fit_lstm_model_thesis(
                Xi_tr,
                yi_tr,
                X_val=Xi_val,
                y_val=yi_val,
                regularization=regularization,
                regularization_strength=param_value,
                lstm_units=lstm_units,
                dense_units=dense_units,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                use_class_weight=use_class_weight,
                verbose=0,
            )
            val_prob = model.predict(Xi_val, verbose=0).ravel()
            val_auc = safe_auc(yi_val, val_prob)
            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_param = param_value

        full_seq = fit_scaler_and_build_sequences(train, test, feature_cols, LOOKBACK)
        X_tr, y_tr = full_seq["X_tr_seq"], full_seq["y_tr_seq"]
        X_te, y_te = full_seq["X_te_seq"], full_seq["y_te_seq"]

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        tf.keras.backend.clear_session()
        model = fit_lstm_model_thesis(
            X_tr,
            y_tr,
            regularization=regularization,
            regularization_strength=best_param,
            lstm_units=lstm_units,
            dense_units=dense_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            use_class_weight=use_class_weight,
            verbose=0,
        )

        pp_tr = model.predict(X_tr, verbose=0).ravel()
        pp_te = model.predict(X_te, verbose=0).ravel()
        yp_tr = (pp_tr >= 0.5).astype(int)
        yp_te = (pp_te >= 0.5).astype(int)

        label = f"{month_label[unique_months[i]]}–{month_label[unique_months[i + TEST_MONTHS - 1]]}"
        always_up_acc = float(np.mean(y_te == 1))

        results.append({
            "window": label,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "best_param": best_param,
            "always_up_acc": always_up_acc,
            "train_acc": accuracy_score(y_tr, yp_tr),
            "test_acc": accuracy_score(y_te, yp_te),
            "train_f1": f1_score(y_tr, yp_tr, zero_division=0),
            "test_f1": f1_score(y_te, yp_te, zero_division=0),
            "train_auc": safe_auc(y_tr, pp_tr),
            "test_auc": safe_auc(y_te, pp_te),
            "train_mcc": matthews_corrcoef(y_tr, yp_tr),
            "test_mcc": matthews_corrcoef(y_te, yp_te),
        })

        preds.append(pd.concat([
            build_prediction_split_df(full_seq["train_dates"], label, "train", y_tr, yp_tr, pp_tr, best_param=best_param),
            build_prediction_split_df(full_seq["test_dates"], label, "test", y_te, yp_te, pp_te, best_param=best_param),
        ], ignore_index=True))

    return pd.DataFrame(results), pd.concat(preds, ignore_index=True)


def load_pickle_dict(path: Path) -> dict:
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    return {}


def build_seed_detail_df(seed_run_results: dict) -> pd.DataFrame:
    rows = []
    for (model_name, seed_label), res in seed_run_results.items():
        rows.append({
            "Model": str(model_name).strip(),
            "Seed": str(seed_label).strip(),
            "Train ACC": res["train_acc"].mean(),
            "Test ACC": res["test_acc"].mean(),
            "Gap ACC": (res["train_acc"] - res["test_acc"]).mean(),
            "Always-Up ACC": res["always_up_acc"].mean(),
            "Delta vs Up": (res["test_acc"] - res["always_up_acc"]).mean(),
            "Train AUC": res["train_auc"].mean(),
            "Test AUC": res["test_auc"].mean(),
            "Gap AUC": (res["train_auc"] - res["test_auc"]).mean(),
            "Train MCC": res["train_mcc"].mean(),
            "Test MCC": res["test_mcc"].mean(),
            "Gap MCC": (res["train_mcc"] - res["test_mcc"]).mean(),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Model"] = out["Model"].astype(str).str.strip()
    out["Seed"] = out["Seed"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["Model", "Seed"], keep="last")
    out = out.sort_values(["Model", "Seed"], kind="stable").reset_index(drop=True)
    return out


def artifact_paths(artifact_dir: Path, run_tag: str) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_pkl": artifact_dir / f"seed_run_results_{run_tag}.pkl",
        "preds_pkl": artifact_dir / f"seed_pred_results_{run_tag}.pkl",
        "detail_csv": artifact_dir / f"seed_detail_results_{run_tag}.csv",
        "manifest_json": artifact_dir / f"seed_manifest_{run_tag}.json",
    }


def persist_seed_artifacts(paths: dict[str, Path], seed_run_results: dict, seed_pred_results: dict, manifest: dict) -> pd.DataFrame:
    with paths["results_pkl"].open("wb") as fh:
        pickle.dump(seed_run_results, fh)

    with paths["preds_pkl"].open("wb") as fh:
        pickle.dump(seed_pred_results, fh)

    detail_df = build_seed_detail_df(seed_run_results)
    if not detail_df.empty:
        detail_df.to_csv(paths["detail_csv"], index=False)

    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, default=str))
    return detail_df


def resolve_models(raw_models: str) -> list[str]:
    lowered = raw_models.strip().lower()
    if lowered == "all":
        return list(MODEL_SPECS.keys())
    if lowered == "unreg":
        return ["Small Unreg", "Large Unreg"]
    if lowered == "regularized":
        return [name for name in MODEL_SPECS if "Unreg" not in name]

    models = [part.strip() for part in raw_models.split(",") if part.strip()]
    invalid = [name for name in models if name not in MODEL_SPECS]
    if invalid:
        raise ValueError(f"Unknown model names: {invalid}")
    return models


def run_model_for_seed(
    model_name: str,
    seed: int,
    df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int,
    batch_size: int,
    patience: int,
    use_class_weight: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_all_seeds(seed)
    spec = MODEL_SPECS[model_name]

    if spec["runner"] == "unreg":
        return walk_forward_thesis_unreg(
            df,
            feature_cols,
            **spec["kwargs"],
            learning_rate=THESIS_LEARNING_RATE,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            use_class_weight=use_class_weight,
        )

    return walk_forward_thesis_tuned(
        df,
        feature_cols,
        regularization=spec["regularization"],
        param_grid=spec["param_grid"],
        **spec["kwargs"],
        learning_rate=THESIS_LEARNING_RATE,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        use_class_weight=use_class_weight,
    )


def main() -> None:
    args = parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    models_to_run = resolve_models(args.models)
    use_class_weight = False if args.disable_class_weight else THESIS_USE_CLASS_WEIGHT

    data = load_data(args.data_path)
    feature_cols = get_feature_columns(data)[:20]
    paths = artifact_paths(args.artifact_dir, args.run_tag)

    seed_run_results = {} if args.overwrite else load_pickle_dict(paths["results_pkl"])
    seed_pred_results = {} if args.overwrite else load_pickle_dict(paths["preds_pkl"])

    manifest = {
        "run_tag": args.run_tag,
        "data_path": str(args.data_path),
        "artifact_dir": str(args.artifact_dir),
        "seeds": seeds,
        "models": models_to_run,
        "lookback": LOOKBACK,
        "min_train_months": MIN_TRAIN_MONTHS,
        "test_months": TEST_MONTHS,
        "val_months": VAL_MONTHS,
        "learning_rate": THESIS_LEARNING_RATE,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "use_class_weight": use_class_weight,
        "started_at": pd.Timestamp.utcnow().isoformat(),
        "completed": [],
    }

    print("TensorFlow version:", tf.__version__)
    print("Data shape:", data.shape)
    print("Date range:", data.index.min().date(), "to", data.index.max().date())
    print("Using", len(feature_cols), "features")
    print("Artifact files:")
    for key, path in paths.items():
        print(" ", key, "->", path)

    for seed in seeds:
        print(f"\n===== Running seed {seed} =====")
        for model_name in models_to_run:
            combo_key = (model_name, seed)
            if not args.overwrite and combo_key in seed_run_results and combo_key in seed_pred_results:
                print(f"Skipping {model_name}, seed={seed} because artifacts already exist")
                continue

            print(f"Running {model_name}, seed={seed}")
            start = time.time()
            res, preds = run_model_for_seed(
                model_name,
                seed,
                data,
                feature_cols,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                use_class_weight=use_class_weight,
            )
            elapsed_min = (time.time() - start) / 60

            seed_run_results[combo_key] = res.copy()
            seed_pred_results[combo_key] = preds.copy()

            manifest["completed"].append({
                "model": model_name,
                "seed": seed,
                "elapsed_minutes": round(elapsed_min, 2),
                "mean_test_acc": float(res["test_acc"].mean()),
                "mean_test_auc": float(res["test_auc"].mean()),
                "n_windows": int(len(res)),
            })

            detail_df = persist_seed_artifacts(paths, seed_run_results, seed_pred_results, manifest)
            print(
                f"  done in {elapsed_min:.1f} min | "
                f"test_acc={res['test_acc'].mean():.4f} | "
                f"test_auc={res['test_auc'].mean():.4f} | "
                f"detail_rows={len(detail_df)}"
            )

    manifest["finished_at"] = pd.Timestamp.utcnow().isoformat()
    persist_seed_artifacts(paths, seed_run_results, seed_pred_results, manifest)

    print("\nFinished. Saved artifacts:")
    for path in paths.values():
        print(" ", path)


if __name__ == "__main__":
    main()
