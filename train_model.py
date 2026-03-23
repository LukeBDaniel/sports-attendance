"""
train_model.py  (shared)
------------------------
Trains the attendance prediction model for any team.

Usage:  python train_model.py <team_folder>
        e.g.  python train_model.py knicks
              python train_model.py rangers

Reads:  <team>/data/games.csv
Writes: <team>/model.pkl
        <team>/data/feature_importance.png
        <team>/data/actual_vs_predicted.png
"""

import os, sys, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import shap

if len(sys.argv) < 2:
    print("Usage: python train_model.py <team_folder>")
    print("       e.g. python train_model.py knicks")
    sys.exit(1)

TEAM = sys.argv[1].lower()
TEAM_DIR  = os.path.join(os.path.dirname(__file__), TEAM)
DATA_FILE = os.path.join(TEAM_DIR, "data", "games.csv")
MODEL_FILE = os.path.join(TEAM_DIR, "model.pkl")
DATA_DIR  = os.path.join(TEAM_DIR, "data")

FEATURES = [
    "season",
    "day_of_week",
    "month",
    "is_weekend",
    "is_evening",
    "temperature_f",
    "precipitation_mm",
    "home_win_rate_last5",
    "opp_win_rate_season",
    "is_rivalry",
    "venue_capacity",
    "season_progress",
]

TARGET = "attendance_pct"
TEST_FRACTION = 0.20
RANDOM_SEED = 42


def load_and_validate(path):
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=FEATURES + [TARGET])
    before = len(df)
    df = df[df[TARGET] <= 1.5].copy()
    print(f"Dropped {before - len(df)} games with fill rate > 1.5")
    df[TARGET] = df[TARGET].clip(upper=1.0)
    print(f"Loaded {len(df)} games across {df['season'].nunique()} seasons")
    print(f"Seasons: {sorted(df['season'].unique())}")
    return df


def evaluate(name, y_true, y_pred, capacities=None):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  MAE  (fill rate): {mae:.4f}  ({mae*100:.1f}% of capacity)")
    print(f"  RMSE (fill rate): {rmse:.4f}")
    print(f"  R²:               {r2:.3f}")
    if capacities is not None:
        mae_fans = mean_absolute_error(y_true * capacities, y_pred * capacities)
        print(f"  MAE  (fans):      {mae_fans:,.0f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    print(f"\n{'='*60}")
    print(f"  Training model for: {TEAM.upper()}")
    print(f"{'='*60}")

    df = load_and_validate(DATA_FILE)
    train_df, test_df = train_test_split(df, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

    # Baseline
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    baseline = evaluate("Baseline (Ridge)", y_test, ridge.predict(X_test), test_df["venue_capacity"])

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.1, n_iter_no_change=30, verbose=1,
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    metrics = evaluate("Gradient Boosting", y_test, gb_preds, test_df["venue_capacity"])

    # SHAP
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(gb)
    shap_vals = explainer(X_train)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_vals, max_display=len(FEATURES), ax=ax, show=False)
    ax.set_title(f"{TEAM.title()} — Feature Importance (mean |SHAP|)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, gb_preds, alpha=0.5, s=30)
    lims = [min(y_test.min(), gb_preds.min()) * 0.9, max(y_test.max(), gb_preds.max()) * 1.05]
    ax.plot(lims, lims, "--r", lw=1.5, label="Perfect")
    ax.set_xlabel("Actual Fill Rate"); ax.set_ylabel("Predicted Fill Rate")
    ax.set_title(f"{TEAM.title()} — Actual vs Predicted\nMAE={metrics['mae']:.3f} R²={metrics['r2']:.3f}")
    ax.legend(); plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    bundle = {
        "model": gb, "explainer": explainer, "features": FEATURES,
        "metrics": metrics,
        "train_years": sorted(train_df["season"].unique().tolist()),
        "test_fraction": TEST_FRACTION, "baseline_mae": baseline["mae"],
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nModel saved → {MODEL_FILE}")
    print(f"GB MAE {metrics['mae']:.4f} vs Ridge MAE {baseline['mae']:.4f}")


if __name__ == "__main__":
    main()
