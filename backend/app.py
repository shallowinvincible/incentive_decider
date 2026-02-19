"""
Flask API for Dynamic Incentive Optimization System
Serves predictions, model info, and dashboard statistics.
"""

import os
import sys
import json
import traceback
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

        results = optimizer.optimize_batch(data)

        # Summary stats
        incentives = [r["recommended_incentive"] for r in results]
        profits = [r["expected_profit"] for r in results]
        acceptances = [r["predicted_acceptance_probability"] for r in results]

        summary = {
            "total_orders": len(results),
            "avg_incentive": round(float(np.mean(incentives)), 2) if incentives else 0,
            "total_incentive_cost": round(float(np.sum(incentives)), 2),
            "avg_profit": round(float(np.mean(profits)), 2) if profits else 0,
            "total_profit": round(float(np.sum(profits)), 2),
            "avg_acceptance": round(float(np.mean(acceptances)), 4) if acceptances else 0,
            "pct_above_threshold": round(float(np.mean([a >= 0.9 for a in acceptances])) * 100, 1) if acceptances else 0,
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
        # Try to load pre-calculated stats if they exist
        stats_json_path = os.path.join(DATA_DIR, "dashboard_stats.json")
        if os.path.exists(stats_json_path):
            with open(stats_json_path, "r") as f:
                return jsonify(json.load(f))

        # Fallback to loading CSV with native python (much slower but avoids pandas)
        import csv
        hist_path = os.path.join(DATA_DIR, "historical_orders.csv")
        
        data = []
        with open(hist_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Convert core numeric fields
                    row["incentive_given"] = float(row.get("incentive_given", 0))
                    row["delivery_revenue"] = float(row.get("delivery_revenue", 0))
                    row["order_accepted"] = int(float(row.get("order_accepted", 0)))
                    row["distance_km"] = float(row.get("distance_km", 0))
                    row["order_value"] = float(row.get("order_value", 0))
                    data.append(row)
                except: continue

        total_orders = len(data)
        if total_orders == 0: return jsonify({"error": "No data available"}), 404
        
        avg_incentive = round(sum(r["incentive_given"] for r in data) / total_orders, 2)
        avg_revenue = round(sum(r["delivery_revenue"] for r in data) / total_orders, 2)
        acceptance_rate = round((sum(r["order_accepted"] for r in data) / total_orders) * 100, 1)
        avg_distance = round(sum(r["distance_km"] for r in data) / total_orders, 1)
        
        total_profit = sum(r["delivery_revenue"] - r["incentive_given"] for r in data)
        avg_profit = round(total_profit / total_orders, 2)

        return jsonify({
            "kpis": {
                "total_orders": total_orders,
                "avg_incentive": avg_incentive,
                "avg_revenue": avg_revenue,
                "acceptance_rate": acceptance_rate,
                "avg_distance": avg_distance,
                "avg_profit": avg_profit,
                "total_profit": round(total_profit, 2)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample-orders", methods=["GET"])
def sample_orders():
    """Return sample new orders for the demo."""
    try:
        import csv, random
        new_path = os.path.join(DATA_DIR, "new_orders.csv")
        
        with open(new_path, "r") as f:
            reader = csv.DictReader(f)
            all_orders = list(reader)
            
        n = min(int(request.args.get("n", 10)), 50)
        sample = random.sample(all_orders, min(len(all_orders), n))
        
        # Convert numeric strings to floats for consistency in UI
        for row in sample:
            for k, v in row.items():
                try: row[k] = float(v)
                except: pass
                
        return jsonify({"orders": sample, "total_available": len(all_orders)})
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
    app.run(host="0.0.0.0", port=5000, debug=True)
