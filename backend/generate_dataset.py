"""
Synthetic Dataset Generator for Dynamic Incentive Optimization
Generates realistic delivery order data with correlated features and rider acceptance behavior.
"""

import pandas as pd
import numpy as np
import json
import os

SEED = 42
np.random.seed(SEED)

# --- Configuration ---
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Jaipur"]
ZONES = ["Central", "North", "South", "East", "West", "Suburban", "Airport", "Industrial"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Fog", "Storm"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Very High"]


def generate_orders(n_records=15000):
    """Generate synthetic delivery order records."""
    data = {}

    # Order IDs
    data["order_id"] = np.arange(10001, 10001 + n_records)

    # Location
    data["city"] = np.random.choice(CITIES, n_records, p=[0.2, 0.18, 0.18, 0.12, 0.1, 0.08, 0.08, 0.06])
    data["zone"] = np.random.choice(ZONES, n_records)

    # Distance & Time
    data["distance_km"] = np.round(np.random.lognormal(mean=1.5, sigma=0.6, size=n_records).clip(1, 35), 1)
    data["estimated_time_min"] = np.round(data["distance_km"] * np.random.uniform(2.5, 5.0, n_records) + np.random.normal(5, 2, n_records), 1).clip(5, 120)

    # Order Value & Fees
    data["order_value"] = np.round(np.random.lognormal(mean=5.5, sigma=0.5, size=n_records).clip(100, 3000), 0)
    data["base_delivery_fee"] = np.round(20 + data["distance_km"] * np.random.uniform(3, 6, n_records), 0)

    # Surge
    hour = np.random.choice(24, n_records)
    is_peak = ((hour >= 11) & (hour <= 14)) | ((hour >= 19) & (hour <= 22))
    data["surge_multiplier"] = np.where(is_peak, np.random.uniform(1.2, 2.5, n_records), np.random.uniform(1.0, 1.3, n_records)).round(2)

    # Weather
    data["weather"] = np.random.choice(WEATHER_CONDITIONS, n_records, p=[0.35, 0.25, 0.15, 0.1, 0.1, 0.05])
    weather_rain_map = {"Clear": 0, "Cloudy": 0, "Light Rain": 0.3, "Heavy Rain": 0.7, "Fog": 0.05, "Storm": 0.9}
    data["rain_intensity"] = np.array([weather_rain_map[w] + np.random.uniform(-0.05, 0.15) for w in data["weather"]]).clip(0, 1).round(2)

    data["temperature"] = np.round(np.random.normal(30, 6, n_records).clip(10, 48), 1)
    data["humidity"] = np.round(np.random.uniform(30, 95, n_records), 1)
    data["wind_speed"] = np.round(np.random.lognormal(mean=2, sigma=0.7, size=n_records).clip(0, 60), 1)

    # Traffic
    data["traffic_level"] = np.random.choice(TRAFFIC_LEVELS, n_records, p=[0.2, 0.35, 0.3, 0.15])

    # Time features
    data["hour_of_day"] = hour
    data["day_of_week"] = np.random.choice(DAYS_OF_WEEK, n_records)
    data["is_weekend"] = np.isin(data["day_of_week"], ["Saturday", "Sunday"]).astype(int)

    # Festival
    data["festival_flag"] = np.random.choice([0, 1], n_records, p=[0.92, 0.08])

    # Restaurant & Rider History
    data["restaurant_prep_time"] = np.round(np.random.lognormal(mean=2.5, sigma=0.4, size=n_records).clip(5, 45), 1)
    data["historical_rider_speed"] = np.round(np.random.normal(22, 5, n_records).clip(8, 40), 1)
    data["historical_acceptance_rate_zone"] = np.round(np.random.beta(8, 2, n_records), 3)
    data["historical_cancel_rate_zone"] = np.round(np.random.beta(1.5, 10, n_records), 3)

    # Delivery Revenue (platform earns from commission)
    commission_rate = 0.22
    data["delivery_revenue"] = np.round(data["order_value"] * commission_rate + data["base_delivery_fee"] * data["surge_multiplier"], 2)

    return pd.DataFrame(data), hour, is_peak


def compute_acceptance_and_incentive(df, hour, is_peak):
    """
    Compute realistic incentive_given and order_accepted based on order difficulty.
    Models complex conditional relationships with clear signal for ML learning.
    """
    n = len(df)

    # --- Difficulty Score (0-1, higher = harder to accept) ---
    dist_score = (df["distance_km"] - 1) / 34  # normalized
    rain_score = df["rain_intensity"]
    traffic_map = {"Low": 0.1, "Medium": 0.35, "High": 0.65, "Very High": 0.95}
    traffic_score = df["traffic_level"].map(traffic_map)
    night_score = ((hour >= 22) | (hour <= 5)).astype(float)
    festival_score = df["festival_flag"].astype(float)
    prep_score = (df["restaurant_prep_time"] - 5) / 40
    low_acceptance_zone = (1 - df["historical_acceptance_rate_zone"])
    high_cancel_zone = df["historical_cancel_rate_zone"]
    temp_extreme = np.where((df["temperature"] > 40) | (df["temperature"] < 15), 0.6, 0)
    wind_score = (df["wind_speed"]) / 60

    # Strong interaction effects (these make the model more interesting)
    rain_night = rain_score * night_score * 2.0
    rain_distance = rain_score * dist_score * 1.8
    traffic_distance = traffic_score * dist_score * 1.5
    storm_far = (df["weather"] == "Storm").astype(float) * dist_score * 2.5

    difficulty = (
        dist_score * 0.22 +
        rain_score * 0.15 +
        traffic_score * 0.13 +
        night_score * 0.06 +
        festival_score * 0.04 +
        prep_score * 0.04 +
        low_acceptance_zone * 0.06 +
        high_cancel_zone * 0.03 +
        temp_extreme * 0.02 +
        wind_score * 0.03 +
        rain_night * 0.06 +
        rain_distance * 0.06 +
        traffic_distance * 0.05 +
        storm_far * 0.05
    ).clip(0, 1)

    # Very small noise on difficulty
    difficulty = (difficulty + np.random.normal(0, 0.02, n)).clip(0, 1)

    # --- Incentive Given (based on difficulty + platform heuristic) ---
    base_incentive = difficulty * 180  # max ~180
    noise = np.random.normal(0, 10, n)
    # Platform sometimes deliberately experiments with over/underpaying
    platform_random = np.random.uniform(-15, 25, n)
    incentive_given = np.round((base_incentive + noise + platform_random).clip(0, 200), 0)

    # --- Acceptance Probability (STRONG signal) ---
    # The key insight: acceptance depends on how much incentive EXCEEDS what difficulty demands
    # This creates a clear, learnable relationship

    # What the rider "expects" based on difficulty
    expected_incentive = difficulty * 160 + 10  # riders expect 10-170 based on difficulty

    # Incentive adequacy ratio (how well we're paying vs expectations)
    adequacy = (incentive_given - expected_incentive) / 80  # >0 means overpaying, <0 underpaying

    # Base acceptance from adequacy (sigmoid-like)
    accept_prob = 1 / (1 + np.exp(-3.5 * adequacy))  # steeper sigmoid for clear boundary

    # Zone history shifts baseline slightly
    accept_prob = accept_prob + (df["historical_acceptance_rate_zone"] - 0.8) * 0.08

    # Small noise (but much less than before)
    accept_prob = (accept_prob + np.random.normal(0, 0.03, n)).clip(0.02, 0.98)

    # Binary acceptance
    order_accepted = (np.random.random(n) < accept_prob).astype(int)

    df["incentive_given"] = incentive_given
    df["order_accepted"] = order_accepted
    df["delivery_revenue"] = df["delivery_revenue"].round(2)

    return df


def generate_platform_config(output_dir):
    """Generate platform configuration JSON."""
    config = {
        "commission_rate": 0.22,
        "delay_penalty_per_minute": 2.5,
        "max_incentive": 200,
        "min_incentive": 0,
        "incentive_step": 5,
        "required_acceptance_threshold": 0.90,
        "currency": "INR",
        "platform_name": "SwiftDeliver",
        "version": "1.0"
    }
    path = os.path.join(output_dir, "platform_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[✓] Platform config saved to {path}")
    return config


def main():
    """Generate all datasets."""
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  SYNTHETIC DATA GENERATOR")
    print("  Dynamic Incentive Optimization System")
    print("=" * 60)

    # Generate historical data
    print("\n[1/3] Generating historical orders...")
    df, hour, is_peak = generate_orders(n_records=12000)
    df = compute_acceptance_and_incentive(df, hour, is_peak)
    hist_path = os.path.join(output_dir, "historical_orders.csv")
    df.to_csv(hist_path, index=False)
    print(f"  → {len(df)} records saved to {hist_path}")
    print(f"  → Acceptance rate: {df['order_accepted'].mean():.2%}")
    print(f"  → Avg incentive: ₹{df['incentive_given'].mean():.1f}")
    print(f"  → Avg delivery revenue: ₹{df['delivery_revenue'].mean():.1f}")

    # Generate new orders (no incentive, no acceptance)
    print("\n[2/3] Generating new orders for prediction...")
    new_df, _, _ = generate_orders(n_records=3000)
    new_path = os.path.join(output_dir, "new_orders.csv")
    new_df.to_csv(new_path, index=False)
    print(f"  → {len(new_df)} records saved to {new_path}")

    # Platform config
    print("\n[3/3] Generating platform config...")
    generate_platform_config(output_dir)

    print("\n" + "=" * 60)
    print("  ALL DATA GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
