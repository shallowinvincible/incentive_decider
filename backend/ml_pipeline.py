"""
ML Pipeline for Dynamic Incentive Optimization
Uses XGBoost directly on encoded features for acceptance prediction.
XGBoost handles non-linear interactions natively — no need for polynomial expansion.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)


class IncentivePipeline:
    """End-to-end ML pipeline for rider acceptance prediction."""

    def __init__(self, data_dir=None, models_dir=None):
        base = os.path.dirname(__file__)
        self.data_dir = data_dir or os.path.join(base, "data")
        self.models_dir = models_dir or os.path.join(base, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.scaler = None
        self.categorical_columns = ["weather", "city", "zone", "day_of_week", "traffic_level"]
        self.numeric_columns = []
        self.feature_columns = []
        self.model = None
        self.metrics = {}
        self.feature_importances = {}
        self._encoded_col_template = None

    def load_data(self):
        """Load historical orders CSV."""
        path = os.path.join(self.data_dir, "historical_orders.csv")
        self.df = pd.read_csv(path)
        print(f"[✓] Loaded {len(self.df)} records from {path}")
        return self.df

    def clean_data(self, df=None):
        """Data cleaning — dedup, imputation."""
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()

        initial = len(df)
        df = df.drop_duplicates()
        print(f"  → Removed {initial - len(df)} duplicates")

        # Identify numeric columns
        exclude = ["order_id", "order_accepted"]
        all_cols = [c for c in df.columns if c not in exclude]
        num_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in self.categorical_columns if c in df.columns]

        # Impute
        for c in num_cols:
            if df[c].isnull().sum() > 0:
                df[c] = df[c].fillna(df[c].median())

        for c in cat_cols:
            if df[c].isnull().sum() > 0:
                df[c] = df[c].fillna("Unknown")

        self.numeric_columns = num_cols
        return df

    def encode_and_prepare(self, df, fit=True):
        """One-hot encode categoricals + scale numerics. Returns feature matrix X."""
        cat_cols = [c for c in self.categorical_columns if c in df.columns]

        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Define feature columns (exclude metadata and target)
        exclude = ["order_id", "order_accepted"]
        if fit:
            self.feature_columns = [c for c in df_encoded.columns if c not in exclude]
            self._encoded_col_template = self.feature_columns.copy()
        else:
            # Align columns with training
            for col in self._encoded_col_template:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            self.feature_columns = self._encoded_col_template

        X = df_encoded[self.feature_columns].copy().astype(float)

        # Scale numeric features
        scale_cols = [c for c in self.numeric_columns if c in X.columns]
        if fit:
            self.scaler = StandardScaler()
            X[scale_cols] = self.scaler.fit_transform(X[scale_cols])
        else:
            X[scale_cols] = self.scaler.transform(X[scale_cols])

        return X

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier for acceptance prediction."""
        print("\n[4/5] Training XGBoost Acceptance Model...")

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=10,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
        }

        print(f"  → Accuracy:  {self.metrics['accuracy']}")
        print(f"  → ROC-AUC:   {self.metrics['roc_auc']}")
        print(f"  → Precision:  {self.metrics['precision']}")
        print(f"  → Recall:     {self.metrics['recall']}")
        print(f"  → F1 Score:   {self.metrics['f1_score']}")

        # Feature importances (top 20)
        importances = self.model.feature_importances_
        feature_names = self.feature_columns
        sorted_idx = np.argsort(importances)[::-1][:20]
        self.feature_importances = {
            feature_names[i]: round(float(importances[i]), 6)
            for i in sorted_idx
        }

        return self.model

    def run_full_pipeline(self):
        """Execute the complete training pipeline."""
        print("=" * 60)
        print("  ML PIPELINE — Training (XGBoost)")
        print("=" * 60)

        # Step 1
        print("\n[1/5] Loading data...")
        df = self.load_data()

        # Step 2
        print("\n[2/5] Cleaning data...")
        df = self.clean_data(df)

        # Step 3
        print("\n[3/5] Encoding features & scaling...")
        y = df["order_accepted"]
        X = self.encode_and_prepare(df, fit=True)

        print(f"  → Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  → Train: {len(X_train)}, Test: {len(X_test)}")

        # Train
        self.train_model(X_train, y_train, X_test, y_test)

        # Save
        print("\n[5/5] Saving model artifacts...")
        self.save_artifacts()

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
        return self.metrics

    def save_artifacts(self):
        """Save all model artifacts to disk."""
        artifacts = {
            "scaler": self.scaler,
            "model": self.model,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "_encoded_col_template": self._encoded_col_template,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
        }

        path = os.path.join(self.models_dir, "pipeline_artifacts.pkl")
        joblib.dump(artifacts, path)
        print(f"  → Artifacts saved to {path}")

        metrics_path = os.path.join(self.models_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  → Metrics saved to {metrics_path}")

        fi_path = os.path.join(self.models_dir, "feature_importances.json")
        with open(fi_path, "w") as f:
            json.dump(self.feature_importances, f, indent=2)
        print(f"  → Feature importances saved to {fi_path}")

    def load_artifacts(self):
        """Load model artifacts from disk."""
        path = os.path.join(self.models_dir, "pipeline_artifacts.pkl")
        artifacts = joblib.load(path)

        self.scaler = artifacts["scaler"]
        self.model = artifacts["model"]
        self.feature_columns = artifacts["feature_columns"]
        self.numeric_columns = artifacts["numeric_columns"]
        self.categorical_columns = artifacts["categorical_columns"]
        self._encoded_col_template = artifacts["_encoded_col_template"]
        self.metrics = artifacts["metrics"]
        self.feature_importances = artifacts["feature_importances"]

        print(f"[✓] Artifacts loaded from {path}")
        return True

    def predict_acceptance_probability(self, order_features, incentive):
        """
        Predict acceptance probability for a single order at a given incentive.
        order_features: dict of raw order features
        incentive: float, incentive amount
        """
        row = pd.DataFrame([order_features])
        row["incentive_given"] = incentive

        # Clean
        for c in self.numeric_columns:
            if c in row.columns:
                row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0)

        for c in self.categorical_columns:
            if c in row.columns:
                row[c] = row[c].fillna("Unknown")

        # Encode (fit=False → aligns to training columns)
        X = self.encode_and_prepare(row, fit=False)

        # Predict
        prob = self.model.predict_proba(X)[:, 1][0]
        return float(prob)


if __name__ == "__main__":
    pipeline = IncentivePipeline()
    pipeline.run_full_pipeline()
