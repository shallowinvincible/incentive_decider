"""
Incentive Optimization Engine
Finds the optimal incentive for each order to maximize profit while maintaining acceptance threshold.
"""

import numpy as np
import json
import os


class IncentiveOptimizer:
    """Optimizes incentive amounts per order."""

    def __init__(self, pipeline, config_path=None):
        self.pipeline = pipeline
        base = os.path.dirname(__file__)
        config_path = config_path or os.path.join(base, "data", "platform_config.json")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.max_incentive = self.config["max_incentive"]
        self.min_incentive = self.config["min_incentive"]
        self.step = self.config["incentive_step"]
        self.threshold = self.config["required_acceptance_threshold"]

    def optimize_single_order(self, order_features):
        """
        Find the optimal incentive for a single order.

        Returns:
            dict with order_id, recommended_incentive, predicted_acceptance_probability, expected_profit
        """
        delivery_revenue = order_features.get("delivery_revenue", 100)
        order_id = order_features.get("order_id", 0)

        # Generate candidate incentives
        candidates = np.arange(self.min_incentive, self.max_incentive + self.step, self.step)

        best_incentive = None
        best_profit = -np.inf
        best_acceptance = 0
        all_results = []

        for incentive in candidates:
            try:
                prob = self.pipeline.predict_acceptance_probability(order_features, incentive)
            except Exception:
                prob = 0.5

            expected_profit = (delivery_revenue - incentive) * prob

            all_results.append({
                "incentive": float(incentive),
                "acceptance_probability": round(float(prob), 4),
                "expected_profit": round(float(expected_profit), 2)
            })

            # Only consider incentives that meet thresh
            if prob >= self.threshold:
                if expected_profit > best_profit:
                    best_profit = expected_profit
                    best_incentive = float(incentive)
                    best_acceptance = float(prob)

        # If no incentive meets threshold, pick the one with highest acceptance
        if best_incentive is None:
            # Fallback: pick highest acceptance
            best_result = max(all_results, key=lambda x: x["acceptance_probability"])
            best_incentive = best_result["incentive"]
            best_acceptance = best_result["acceptance_probability"]
            best_profit = best_result["expected_profit"]

        return {
            "order_id": order_id,
            "recommended_incentive": round(best_incentive, 0),
            "predicted_acceptance_probability": round(best_acceptance, 4),
            "expected_profit": round(best_profit, 2),
            "delivery_revenue": round(delivery_revenue, 2),
            "threshold_met": best_acceptance >= self.threshold,
            "incentive_curve": all_results  # For visualization
        }

    def optimize_batch(self, orders_list):
        """
        Optimize incentives for a batch of orders.

        Args:
            orders_list: List of dictionaries with order features

        Returns:
            list of result dicts
        """
        results = []
        total = len(orders_list)

        for idx, order_features in enumerate(orders_list):
            result = self.optimize_single_order(order_features)
            # Don't include curve in batch (too much data)
            result.pop("incentive_curve", None)
            results.append(result)

            if (idx + 1) % 100 == 0:
                print(f"  → Processed {idx + 1}/{total} orders")

        return results
