from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42


def _try_xgb():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier, "xgboost", ""
    except Exception as exc:
        return None, "random_forest_fallback", f"xgboost unavailable: {type(exc).__name__}: {exc}"


def make_model(n_classes: int, config: dict):
    xgb_cls, model_name, note = _try_xgb()
    if xgb_cls is None:
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
        return model, model_name, note
    model = xgb_cls(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=int(config.get("xgboost_n_estimators", 150)),
        max_depth=int(config.get("xgboost_max_depth", 3)),
        learning_rate=float(config.get("xgboost_learning_rate", 0.05)),
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return model, model_name, note


def filter_classes(metadata: pd.DataFrame, target_col: str, min_count: int) -> tuple[np.ndarray, pd.Series]:
    labels = metadata[target_col].fillna("<missing>").astype(str)
    counts = labels.value_counts()
    keep_labels = counts[counts >= min_count].index
    keep_mask = labels.isin(keep_labels).to_numpy()
    return keep_mask, counts


def _metric_row(
    representation: str,
    cv_scheme: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    note: str,
) -> dict:
    return {
        "representation": representation,
        "cv_scheme": cv_scheme,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "n_samples": int(len(y_true)),
        "note": note,
    }


def _cross_val_predictions(
    x: np.ndarray,
    y: np.ndarray,
    splits,
    n_classes: int,
    model_config: dict,
) -> tuple[np.ndarray, str, str]:
    pred = np.full_like(y, fill_value=-1)
    model_name = ""
    notes = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(np.unique(y[train_idx])) < n_classes:
            notes.append(f"fold {fold} skipped: training fold lacks at least one class")
            continue
        model, model_name, note = make_model(n_classes, model_config)
        if note:
            notes.append(note)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    valid = pred >= 0
    if not valid.all():
        notes.append(f"{int((~valid).sum())} samples not predicted because folds were skipped")
    return pred, model_name, "; ".join(sorted(set(notes)))


def run_cv_for_representation(
    representation: str,
    frame: pd.DataFrame,
    labels_text: pd.Series,
    metadata: pd.DataFrame,
    model_config: dict,
    out_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    le = LabelEncoder()
    y = le.fit_transform(labels_text)
    x = frame.to_numpy(dtype=float)
    n_classes = len(le.classes_)
    metrics_rows: list[dict] = []
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []

    skf = StratifiedKFold(
        n_splits=int(model_config.get("n_splits", 5)),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    pred, model_name, note = _cross_val_predictions(
        x, y, skf.split(x, y), n_classes, model_config
    )
    valid = pred >= 0
    metrics_rows.append(_metric_row(representation, "stratified_kfold", model_name, y[valid], pred[valid], note))
    per_class_rows.extend(_per_class_rows(representation, "stratified_kfold", le.classes_, y[valid], pred[valid]))
    confusion_rows.extend(_confusion_rows(representation, "stratified_kfold", le.classes_, y[valid], pred[valid]))

    group_col = choose_group_column(metadata, model_config)
    if group_col:
        groups = metadata[group_col].fillna("<missing>").astype(str).to_numpy()
        if len(np.unique(groups)) >= int(model_config.get("n_splits", 5)):
            gkf = GroupKFold(n_splits=int(model_config.get("n_splits", 5)))
            pred_g, model_name_g, note_g = _cross_val_predictions(
                x, y, gkf.split(x, y, groups), n_classes, model_config
            )
            valid_g = pred_g >= 0
            note_g = (note_g + "; " if note_g else "") + f"group_column={group_col}"
            metrics_rows.append(
                _metric_row(representation, "group_kfold", model_name_g, y[valid_g], pred_g[valid_g], note_g)
            )
            per_class_rows.extend(
                _per_class_rows(representation, "group_kfold", le.classes_, y[valid_g], pred_g[valid_g])
            )
            confusion_rows.extend(
                _confusion_rows(representation, "group_kfold", le.classes_, y[valid_g], pred_g[valid_g])
            )

    return metrics_rows, per_class_rows, confusion_rows


def choose_group_column(metadata: pd.DataFrame, model_config: dict) -> str | None:
    if not model_config.get("run_group_cv_if_available", True):
        return None
    priority = model_config.get("group_column_priority", ["source", "paper", "citation", "deposit", "region"])
    lower_map = {col.lower(): col for col in metadata.columns}
    for wanted in priority:
        if wanted.lower() in lower_map:
            return lower_map[wanted.lower()]
    return None


def _per_class_rows(
    representation: str,
    cv_scheme: str,
    class_names: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return [
        {
            "representation": representation,
            "cv_scheme": cv_scheme,
            "class": class_name,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }
        for i, class_name in enumerate(class_names)
    ]


def _confusion_rows(
    representation: str,
    cv_scheme: str,
    class_names: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            rows.append(
                {
                    "representation": representation,
                    "cv_scheme": cv_scheme,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(matrix[i, j]),
                }
            )
    return rows


def run_classification(
    reps: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    target_col: str,
    min_count: int,
    model_config: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    keep_mask, class_counts_all = filter_classes(metadata, target_col, min_count)
    filtered_metadata = metadata.loc[keep_mask].reset_index(drop=True)
    labels_text = filtered_metadata[target_col].fillna("<missing>").astype(str)
    filtered_reps = {name: frame.loc[keep_mask].reset_index(drop=True) for name, frame in reps.items()}

    metrics_rows: list[dict] = []
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for name, frame in filtered_reps.items():
        print(f"[classification] {name}")
        m, p, c = run_cv_for_representation(
            name,
            frame,
            labels_text,
            filtered_metadata,
            model_config,
            out_dir,
        )
        metrics_rows.extend(m)
        per_class_rows.extend(p)
        confusion_rows.extend(c)

    notes = {
        "class_counts_all": class_counts_all.to_dict(),
        "class_counts_used": labels_text.value_counts().to_dict(),
        "n_samples_used": int(len(labels_text)),
        "min_count": int(min_count),
        "group_column": choose_group_column(filtered_metadata, model_config),
    }
    return pd.DataFrame(metrics_rows), pd.DataFrame(per_class_rows), pd.DataFrame(confusion_rows), notes


def feature_importance(
    reps: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    target_col: str,
    min_count: int,
    model_config: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    keep_mask, _ = filter_classes(metadata, target_col, min_count)
    filtered_metadata = metadata.loc[keep_mask].reset_index(drop=True)
    labels_text = filtered_metadata[target_col].fillna("<missing>").astype(str)
    le = LabelEncoder()
    y = le.fit_transform(labels_text)
    n_classes = len(le.classes_)
    permutation_rows = []
    gain_rows = []
    shap_rows = []
    notes: list[str] = []

    for name, frame in reps.items():
        print(f"[importance] {name}")
        x = frame.loc[keep_mask].reset_index(drop=True).to_numpy(dtype=float)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        model, model_name, note = make_model(n_classes, model_config)
        if note:
            notes.append(f"{name}: {note}")
        model.fit(x_train, y_train)

        try:
            perm = permutation_importance(
                model,
                x_test,
                y_test,
                scoring="f1_macro",
                n_repeats=int(model_config.get("permutation_repeats", 5)),
                random_state=RANDOM_STATE,
                n_jobs=2,
            )
            for feature, mean, std in zip(frame.columns, perm.importances_mean, perm.importances_std):
                permutation_rows.append(
                    {
                        "representation": name,
                        "model": model_name,
                        "feature": feature,
                        "importance_mean": float(mean),
                        "importance_std": float(std),
                    }
                )
        except Exception as exc:
            notes.append(f"{name}: permutation importance skipped: {type(exc).__name__}: {exc}")

        if model_name == "xgboost":
            try:
                booster = model.get_booster()
                score = booster.get_score(importance_type="gain")
                for feature_index, feature in enumerate(frame.columns):
                    gain_rows.append(
                        {
                            "representation": name,
                            "feature": feature,
                            "gain": float(score.get(f"f{feature_index}", 0.0)),
                        }
                    )
            except Exception as exc:
                notes.append(f"{name}: xgb gain skipped: {type(exc).__name__}: {exc}")

        shap_rows.extend(_optional_shap_rows(name, model, model_name, frame.columns, x_test, notes))

    return (
        pd.DataFrame(permutation_rows),
        pd.DataFrame(gain_rows),
        pd.DataFrame(shap_rows),
        notes,
    )


def _optional_shap_rows(
    representation: str,
    model,
    model_name: str,
    features: pd.Index,
    x_test: np.ndarray,
    notes: list[str],
) -> list[dict]:
    if model_name != "xgboost":
        notes.append(f"{representation}: SHAP skipped because model is {model_name}")
        return []
    try:
        import shap  # type: ignore
    except Exception as exc:
        notes.append(f"{representation}: SHAP skipped: {type(exc).__name__}: {exc}")
        return []
    try:
        sample = x_test[: min(1000, len(x_test))]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            arr = np.stack(values, axis=0)
            mean_abs = np.abs(arr).mean(axis=(0, 1))
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(arr).mean(axis=0)
        return [
            {
                "representation": representation,
                "feature": feature,
                "mean_abs_shap": float(mean_abs[i]) if i < len(mean_abs) and math.isfinite(float(mean_abs[i])) else 0.0,
            }
            for i, feature in enumerate(features)
        ]
    except Exception as exc:
        notes.append(f"{representation}: SHAP skipped: {type(exc).__name__}: {exc}")
        return []
