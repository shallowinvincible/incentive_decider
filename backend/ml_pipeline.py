"""
Refined ML Pipeline for Dynamic Incentive Optimization
Follows the specific flow:
1. Data Cleaning & Encoding
2. Standard Scaling
3. Polynomial Features (Interaction Only, Degree 2) -> Save Dataset
4. LASSO Selection (LassoCV)
5. Acceptance Classification Model (GradientBoostingClassifier)
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)


class IncentivePipeline:
    """End-to-end ML pipeline with Polynomial Features and LASSO selection."""

    def __init__(self, data_dir=None, models_dir=None):
        base = os.path.dirname(__file__)
        self.data_dir = data_dir or os.path.join(base, "data")
        self.models_dir = models_dir or os.path.join(base, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.scaler = None
        self.poly = None
        self.lasso = None
        self.categorical_columns = ["weather", "city", "zone", "day_of_week", "traffic_level"]
        self.numeric_columns = []
        self.raw_feature_columns = []  # Before poly
        self.selected_feature_indices = [] # Indices of features selected by LASSO
        self.selected_feature_names = []
        
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

    def encode_and_prepare_raw(self, df, fit=True):
        """One-hot encode categoricals. Returns raw encoded feature matrix."""
        cat_cols = [c for c in self.categorical_columns if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Define feature columns (exclude metadata and target)
        exclude = ["order_id", "order_accepted"]
        if fit:
            self.raw_feature_columns = [c for c in df_encoded.columns if c not in exclude]
            self._encoded_col_template = self.raw_feature_columns.copy()
        else:
            # Align columns with training
            for col in self._encoded_col_template:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            self.raw_feature_columns = self._encoded_col_template

        return df_encoded[self.raw_feature_columns].copy().astype(float)

    def run_full_pipeline(self):
        """Execute the complete training pipeline."""
        print("=" * 60)
        print("  REFINED ML PIPELINE — Polynomial + LASSO + XGBoost")
        print("=" * 60)

        # Step 1: Load & Clean
        print("\n[1/7] Loading and cleaning data...")
        df = self.load_data()
        df = self.clean_data(df)
        y = df["order_accepted"]

        # Step 2: Encode
        print("\n[2/7] Encoding categoricals...")
        X_raw = self.encode_and_prepare_raw(df, fit=True)
        
        # Step 3: Scale
        print("\n[3/7] Scaling numeric columns...")
        num_cols_to_scale = [c for c in self.numeric_columns if c in X_raw.columns]
        self.scaler = StandardScaler()
        X_scaled = X_raw.copy()
        X_scaled[num_cols_to_scale] = self.scaler.fit_transform(X_raw[num_cols_to_scale])

        # Step 4: Polynomial Features
        print("\n[4/7] Generating Polynomial Features (Degree 2, Interaction Only)...")
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly_arr = self.poly.fit_transform(X_scaled)
        poly_feature_names = self.poly.get_feature_names_out(self.raw_feature_columns)
        X_poly = pd.DataFrame(X_poly_arr, columns=poly_feature_names)
        
        # Save expanded dataset
        poly_data_path = os.path.join(self.data_dir, "processed_polynomial_features.csv")
        X_poly_to_save = X_poly.copy()
        X_poly_to_save["TARGET_order_accepted"] = y.values
        X_poly_to_save.to_csv(poly_data_path, index=False)
        print(f"  → Expanded dataset saved to {poly_data_path}")
        print(f"  → Polynomial features: {X_poly.shape[1]}")

        # Step 5: LASSO Feature Selection
        print("\n[5/7] Running LASSO Selection (LassoCV)...")
        # Use LassoCV to find informative features
        # We use a small max_iter to keep it fast, or increase if it doesn't converge
        self.lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        self.lasso.fit(X_poly, y)
        
        # Select features with non-zero coefficients
        coef = pd.Series(self.lasso.coef_, index=X_poly.columns)
        self.selected_feature_names = coef[coef != 0].index.tolist()
        
        # fallback if lasso drops everything (unlikely with CV, but good for stability)
        if len(self.selected_feature_names) == 0:
            print("  ! LASSO dropped all features, falling back to top 20 by magnitude")
            self.selected_feature_names = coef.abs().sort_values(ascending=False).head(20).index.tolist()

        print(f"  → Selected {len(self.selected_feature_names)} features out of {len(poly_feature_names)}")

        # Step 6: Train Final Acceptance Model (GradientBoostingClassifier)
        print("\n[6/7] Training Final Acceptance Model (GradientBoostingClassifier)...")
        X_selected = X_poly[self.selected_feature_names]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X_train, y_train)

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
            "n_features_selected": len(self.selected_feature_names),
        }

        print(f"  → Accuracy:  {self.metrics['accuracy']}")
        print(f"  → ROC-AUC:   {self.metrics['roc_auc']}")
        
        # Feature importances
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:20]
        self.feature_importances = {
            self.selected_feature_names[i]: round(float(importances[i]), 6)
            for i in sorted_idx
        }

        # Step 7: Save Artifacts
        print("\n[7/7] Saving model artifacts...")
        self.save_artifacts()

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
        return self.metrics

    def save_artifacts(self):
        """Save all model artifacts to disk."""
        artifacts = {
            "scaler": self.scaler,
            "poly": self.poly,
            "selected_feature_names": self.selected_feature_names,
            "model": self.model,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "raw_feature_columns": self.raw_feature_columns,
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
            
        fi_path = os.path.join(self.models_dir, "feature_importances.json")
        with open(fi_path, "w") as f:
            json.dump(self.feature_importances, f, indent=2)

    def load_artifacts(self):
        """Load model artifacts from disk."""
        path = os.path.join(self.models_dir, "pipeline_artifacts.pkl")
        if not os.path.exists(path):
            return False
        
        artifacts = joblib.load(path)
        self.scaler = artifacts["scaler"]
        self.poly = artifacts["poly"]
        self.selected_feature_names = artifacts["selected_feature_names"]
        self.model = artifacts["model"]
        self.numeric_columns = artifacts["numeric_columns"]
        self.categorical_columns = artifacts["categorical_columns"]
        self.raw_feature_columns = artifacts["raw_feature_columns"]
        self._encoded_col_template = artifacts["_encoded_col_template"]
        self.metrics = artifacts["metrics"]
        self.feature_importances = artifacts.get("feature_importances", {})
        
        print(f"[✓] Artifacts loaded from {path}")
        return True

    def predict_acceptance_probability(self, order_features, incentive):
        """
        Predict acceptance probability for a single order at a given incentive.
        """
        row = pd.DataFrame([order_features])
        row["incentive_given"] = incentive

        # Encode raw
        X_raw = self.encode_and_prepare_raw(row, fit=False)
        
        # Scale
        num_cols = [c for c in self.numeric_columns if c in X_raw.columns]
        X_scaled = X_raw.copy()
        X_scaled[num_cols] = self.scaler.transform(X_raw[num_cols])
        
        # Poly
        X_poly_arr = self.poly.transform(X_scaled)
        poly_names = self.poly.get_feature_names_out(self.raw_feature_columns)
        X_poly = pd.DataFrame(X_poly_arr, columns=poly_names)
        
        # Select
        X_selected = X_poly[self.selected_feature_names]

        # Predict
        prob = self.model.predict_proba(X_selected)[:, 1][0]
        return float(prob)


if __name__ == "__main__":
    pipeline = IncentivePipeline()
    pipeline.run_full_pipeline()
