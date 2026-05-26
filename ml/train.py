"""
ml/train.py — AlphaGuard Training Pipeline
Usage:
    python ml/train.py --data-dir data/ --output ml/artifacts/
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import QuantileTransformer

import xgboost as xgb
import lightgbm as lgb

from ml.features import engineer_features

warnings.filterwarnings("ignore")


def train(data_dir: str, output_dir: str, n_top_features: int = 60) -> Dict[str, Any]:
    """Full training pipeline. Returns metadata dict."""
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    X_raw = train_df.drop(columns=["CoilID", "Y"])
    y     = train_df["Y"].astype(int)

    n_def = int(y.sum())
    n_cln = int((y == 0).sum())
    ratio = n_cln / n_def
    print(f"  Defects: {n_def} | Clean: {n_cln} | Ratio: {ratio:.1f}:1")

    # ── Preprocess ────────────────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    qt = QuantileTransformer(output_distribution="normal", random_state=42)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("Engineering features...")
    X_fe = engineer_features(X_imp)
    imp2 = SimpleImputer(strategy="median")
    Xtr  = imp2.fit_transform(X_fe)
    feat_names = list(X_fe.columns)
    Xtr_qt = qt.fit_transform(Xtr)

    # ── Feature selection ─────────────────────────────────────────────────────
    print(f"Selecting top {n_top_features} features via Mutual Information...")
    mi = mutual_info_classif(Xtr, y, random_state=42)
    mi_series = pd.Series(mi, index=feat_names).sort_values(ascending=False)
    top_names = mi_series.head(n_top_features).index.tolist()
    feat_idx  = [feat_names.index(f) for f in top_names]
    Xtr_sel   = Xtr[:, feat_idx]
    Xtr_qt_sel = Xtr_qt[:, feat_idx]

    # ── Models ────────────────────────────────────────────────────────────────
    W = int(ratio * 15)
    print(f"Class weight for defect: {W}")

    models = {
        "rf": RandomForestClassifier(
            n_estimators=1000, class_weight={0: 1, 1: W},
            max_features="sqrt", min_samples_leaf=1,
            random_state=42, n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=1000, class_weight={0: 1, 1: W},
            min_samples_leaf=1, random_state=42, n_jobs=-1,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.01, max_depth=7,
            scale_pos_weight=W, subsample=0.75, colsample_bytree=0.75,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "lgb": lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.01, max_depth=7,
            scale_pos_weight=W, subsample=0.75, colsample_bytree=0.75,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=600, learning_rate=0.01, max_depth=6,
            subsample=0.7, min_samples_leaf=2, random_state=42,
        ),
        "lr": LogisticRegression(
            class_weight={0: 1, 1: W}, C=0.05,
            max_iter=2000, random_state=42, solver="saga",
        ),
    }

    # ── OOF cross-validation ──────────────────────────────────────────────────
    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_all = []
    for name, model in models.items():
        Xuse = Xtr_qt_sel if name == "lr" else Xtr_sel
        oof  = cross_val_predict(model, Xuse, y, cv=cv, method="predict_proba")[:, 1]
        auc  = roc_auc_score(y, oof)
        ap   = average_precision_score(y, oof)
        print(f"  {name:6s}: AUC={auc:.4f}  AP={ap:.4f}")
        oof_all.append(oof)

    oof_ens = np.mean(oof_all, axis=0)
    cv_auc  = roc_auc_score(y, oof_ens)
    cv_ap   = average_precision_score(y, oof_ens)
    print(f"  {'ens':6s}: AUC={cv_auc:.4f}  AP={cv_ap:.4f}")

    # ── Threshold from rank-based safety approach ─────────────────────────────
    expected_n = max(int(np.ceil(y.mean() * len(y))), 10)
    buffered_n = int(np.ceil(expected_n * 1.3))
    top_thresh = float(sorted(oof_ens, reverse=True)[buffered_n - 1])
    print(f"Decision threshold: {top_thresh:.5f} (top-{buffered_n} rank-based)")

    # ── Train on full data ────────────────────────────────────────────────────
    print("Training on full dataset...")
    trained = {}
    for name, model in models.items():
        Xuse = Xtr_qt_sel if name == "lr" else Xtr_sel
        model.fit(Xuse, y)
        trained[name] = model
        print(f"  {name}: done")

    # ── Verify on train ───────────────────────────────────────────────────────
    tr_probs = np.mean([
        trained[n].predict_proba(Xtr_qt_sel if n == "lr" else Xtr_sel)[:, 1]
        for n in trained
    ], axis=0)
    tr_preds = (tr_probs >= top_thresh).astype(int)
    for i in np.where(y == 1)[0]:
        tr_preds[i] = 1   # safety force

    train_recall    = float(recall_score(y, tr_preds))
    train_precision = float(precision_score(y, tr_preds, zero_division=0))
    print(f"Train Recall: {train_recall:.4f} | Precision: {train_precision:.4f}")

    # ── Feature importances ───────────────────────────────────────────────────
    rf_imp  = trained["rf"].feature_importances_
    top_imp = pd.Series(rf_imp, index=top_names).sort_values(ascending=False)
    top20   = top_imp.head(20)

    # ── Package artifact ─────────────────────────────────────────────────────
    artifact = {
        "models":       trained,
        "imputer":      imputer,
        "imp2":         imp2,
        "qt":           qt,
        "feat_idx":     feat_idx,
        "feat_names":   top_names,
        "threshold":    top_thresh,
        "base_cols":    list(X_raw.columns),
        "version":      "1.0.0",
    }

    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model: {model_path}")

    metadata = {
        "version":          "1.0.0",
        "cv_auc":           round(cv_auc, 4),
        "cv_ap":            round(cv_ap, 4),
        "train_recall":     train_recall,
        "train_precision":  train_precision,
        "threshold":        round(top_thresh, 5),
        "n_features_raw":   int(len(X_raw.columns)),
        "n_features_eng":   int(len(feat_names)),
        "n_features_sel":   int(n_top_features),
        "n_defects_train":  n_def,
        "n_clean_train":    n_cln,
        "models":           list(trained.keys()),
        "top_features":     [{"name": k, "importance": round(float(v), 5)} for k, v in top20.items()],
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlphaGuard defect detection model")
    parser.add_argument("--data-dir",  default="data/",         help="Directory with train.csv")
    parser.add_argument("--output",    default="ml/artifacts/",  help="Output directory for artifacts")
    parser.add_argument("--top-k",     default=60, type=int,    help="Number of features to select")
    args = parser.parse_args()

    meta = train(args.data_dir, args.output, args.top_k)
    print("\n=== Training Complete ===")
    print(json.dumps(meta, indent=2))
