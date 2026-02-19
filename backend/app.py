"""
Flask API for Dynamic Incentive Optimization System
Serves predictions, model info, and dashboard statistics.
"""

import os
import sys
import json
import traceback
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from ml_pipeline import IncentivePipeline
from optimizer import IncentiveOptimizer
from generate_dataset import main as generate_data

app = Flask(__name__)
CORS(app)

# Global references
pipeline = None
optimizer = None
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def ensure_system_ready():
    """Generate data + train model if not already done."""
    global pipeline, optimizer

    # Use paths relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_data_dir = os.path.join(current_dir, "data")
    local_models_dir = os.path.join(current_dir, "models")

    # Check if model exists
    model_path = os.path.join(local_models_dir, "pipeline_artifacts.pkl")
    
    pipeline = IncentivePipeline(data_dir=local_data_dir, models_dir=local_models_dir)
    
    if os.path.exists(model_path):
        pipeline.load_artifacts()
    else:
        # Avoid auto-training in production/serverless environment
        print("[!] Warning: Model artifacts not found. Please train locally first.")

    config_path = os.path.join(local_data_dir, "platform_config.json")
    optimizer = IncentiveOptimizer(pipeline, config_path=config_path)


# ─── ROUTES ──────────────────────────────────────────────────────────────────


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None and pipeline.model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict optimal incentive for a single order."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = optimizer.optimize_single_order(data)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """Predict optimal incentives for a batch of orders (JSON array)."""
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({"error": "Expected JSON array of orders"}), 400

        df = pd.DataFrame(data)
        results = optimizer.optimize_batch(df)

        # Summary stats
        incentives = [r["recommended_incentive"] for r in results]
        profits = [r["expected_profit"] for r in results]
        acceptances = [r["predicted_acceptance_probability"] for r in results]

        summary = {
            "total_orders": len(results),
            "avg_incentive": round(float(np.mean(incentives)), 2),
            "total_incentive_cost": round(float(np.sum(incentives)), 2),
            "avg_profit": round(float(np.mean(profits)), 2),
            "total_profit": round(float(np.sum(profits)), 2),
            "avg_acceptance": round(float(np.mean(acceptances)), 4),
            "pct_above_threshold": round(float(np.mean([a >= 0.9 for a in acceptances])) * 100, 1),
        }

        return jsonify({"results": results, "summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model performance metrics and feature importances."""
    try:
        metrics = pipeline.metrics
        importances = pipeline.feature_importances

        # Load platform config
        config_path = os.path.join(DATA_DIR, "platform_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        return jsonify({
            "metrics": metrics,
            "feature_importances": importances,
            "config": config,
            "model_type": "GradientBoostingClassifier",
            "n_selected_features": len(pipeline.selected_feature_names),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard-stats", methods=["GET"])
def dashboard_stats():
    """Return aggregate statistics from the training data for the dashboard."""
    try:
        hist_path = os.path.join(DATA_DIR, "historical_orders.csv")
        df = pd.read_csv(hist_path)

        # Core KPIs
        total_orders = len(df)
        avg_incentive = round(float(df["incentive_given"].mean()), 2)
        avg_revenue = round(float(df["delivery_revenue"].mean()), 2)
        acceptance_rate = round(float(df["order_accepted"].mean()) * 100, 1)
        avg_distance = round(float(df["distance_km"].mean()), 1)
        avg_order_value = round(float(df["order_value"].mean()), 0)

        # Profit stats
        df["profit"] = df["delivery_revenue"] - df["incentive_given"]
        avg_profit = round(float(df["profit"].mean()), 2)
        total_profit = round(float(df["profit"].sum()), 2)

        # Distribution data
        incentive_dist = df["incentive_given"].describe().to_dict()
        incentive_dist = {k: round(float(v), 2) for k, v in incentive_dist.items()}

        # Incentive by weather
        weather_stats = df.groupby("weather").agg({
            "incentive_given": "mean",
            "order_accepted": "mean",
            "delivery_revenue": "mean",
            "distance_km": "mean"
        }).round(2).to_dict(orient="index")

        # Incentive by hour
        hour_stats = df.groupby("hour_of_day").agg({
            "incentive_given": "mean",
            "order_accepted": "mean",
        }).round(3).to_dict(orient="index")

        # Incentive by city
        city_stats = df.groupby("city").agg({
            "incentive_given": "mean",
            "order_accepted": "mean",
            "distance_km": "mean",
            "delivery_revenue": "mean"
        }).round(2).to_dict(orient="index")

        # Incentive by traffic
        traffic_stats = df.groupby("traffic_level").agg({
            "incentive_given": "mean",
            "order_accepted": "mean",
        }).round(3).to_dict(orient="index")

        # Distance bins
        df["distance_bin"] = pd.cut(df["distance_km"], bins=[0, 3, 6, 10, 15, 35],
                                     labels=["0-3km", "3-6km", "6-10km", "10-15km", "15+km"])
        distance_stats = df.groupby("distance_bin", observed=True).agg({
            "incentive_given": "mean",
            "order_accepted": "mean",
        }).round(3).to_dict(orient="index")

        # Incentive histogram bins
        bins = list(range(0, 220, 20))
        hist_counts, _ = np.histogram(df["incentive_given"], bins=bins)
        incentive_histogram = [
            {"range": f"{bins[i]}-{bins[i+1]}", "count": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ]

        return jsonify({
            "kpis": {
                "total_orders": total_orders,
                "avg_incentive": avg_incentive,
                "avg_revenue": avg_revenue,
                "acceptance_rate": acceptance_rate,
                "avg_distance": avg_distance,
                "avg_order_value": avg_order_value,
                "avg_profit": avg_profit,
                "total_profit": total_profit
            },
            "incentive_distribution": incentive_dist,
            "incentive_histogram": incentive_histogram,
            "weather_stats": weather_stats,
            "hour_stats": hour_stats,
            "city_stats": city_stats,
            "traffic_stats": traffic_stats,
            "distance_stats": distance_stats
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-orders", methods=["GET"])
def sample_orders():
    """Return sample new orders for the demo."""
    try:
        new_path = os.path.join(DATA_DIR, "new_orders.csv")
        df = pd.read_csv(new_path)
        n = min(int(request.args.get("n", 10)), 50)
        sample = df.sample(n=n, random_state=None).to_dict(orient="records")
        return jsonify({"orders": sample, "total_available": len(df)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Retrain the model (regenerate data + retrain pipeline)."""
    global pipeline, optimizer
    try:
        generate_data()
        pipeline = IncentivePipeline(data_dir=DATA_DIR, models_dir=MODELS_DIR)
        metrics = pipeline.run_full_pipeline()

        config_path = os.path.join(DATA_DIR, "platform_config.json")
        optimizer = IncentiveOptimizer(pipeline, config_path=config_path)

        return jsonify({"status": "retrained", "metrics": metrics})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── MAIN ────────────────────────────────────────────────────────────────────

# Initialize system on module load (required for serverless)
ensure_system_ready()

if __name__ == "__main__":
    print("\n🚀 Starting Incentive Optimization API Server...\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
