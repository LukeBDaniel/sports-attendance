"""
train_model.py
--------------
Loads data/nwsl_matches.csv, trains a Gradient Boosting attendance model,
evaluates it, and saves model.pkl for use in the Streamlit app.

Train/test split: random 80/20 across all available seasons (2016–2025,
excluding 2020–2021 COVID years).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "nwsl_matches.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURES = [
    "season",           # captures NWSL secular attendance growth over time
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

TARGET = "attendance_pct"  # fill rate: attendance / venue_capacity (0–1)
TEST_FRACTION = 0.20
RANDOM_SEED = 42


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    # Drop rows with any NaN in features or target
    df = df.dropna(subset=FEATURES + [TARGET])
    # Drop games where reported attendance > 150% of capacity (stale capacity data)
    before = len(df)
    df = df[df[TARGET] <= 1.5].copy()
    print(f"Dropped {before - len(df)} games with fill rate > 1.5 (bad capacity data)")
    # Cap remaining over-reports (standing room etc.) at 1.0
    df[TARGET] = df[TARGET].clip(upper=1.0)
    print(f"Loaded {len(df)} games across {df['season'].nunique()} seasons")
    print(f"Seasons: {sorted(df['season'].unique())}")
    return df


def random_train_test_split(df, test_fraction, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_fraction, random_state=seed)


def evaluate(name: str, y_true, y_pred, venue_capacity_series=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  MAE (fill rate):  {mae:.4f}  ({mae*100:.1f}% of capacity)")
    print(f"  RMSE (fill rate): {rmse:.4f}")
    print(f"  R²:               {r2:.3f}")
    if venue_capacity_series is not None:
        mae_fans = mean_absolute_error(
            y_true * venue_capacity_series,
            y_pred * venue_capacity_series,
        )
        print(f"  MAE (fans):       {mae_fans:,.0f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    # ── Load data ──
    df = load_and_validate(DATA_FILE)

    train_df, test_df = random_train_test_split(df, TEST_FRACTION, RANDOM_SEED)
    print(f"\nTrain: {len(train_df)} games | Test: {len(test_df)} games")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # ── Baseline: Ridge regression ──
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    ridge.fit(X_train, y_train)
    baseline_metrics = evaluate("Baseline (Ridge Regression)",
                                y_test, ridge.predict(X_test),
                                test_df["venue_capacity"])

    # ── Primary: Gradient Boosting ──
    gb_model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=30,
        verbose=1,
    )
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_metrics = evaluate("Gradient Boosting", y_test, gb_preds,
                          test_df["venue_capacity"])

    # Feature importance plot
    importances = pd.Series(gb_model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.barh(ax=ax, color="#e94560")
    ax.set_title("Feature Importance", fontsize=13)
    ax.set_xlabel("Importance")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "data", "feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved feature_importance.png")

    # Actual vs predicted scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, gb_preds, alpha=0.5, s=30, color="#1a1a2e")
    lims = [min(y_test.min(), gb_preds.min()) * 0.9,
            max(y_test.max(), gb_preds.max()) * 1.05]
    ax.plot(lims, lims, "--", color="#e94560", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Fill Rate")
    ax.set_ylabel("Predicted Fill Rate")
    ax.set_title(f"Gradient Boosting: Actual vs Predicted Fill Rate\nMAE = {gb_metrics['mae']:.3f} | R² = {gb_metrics['r2']:.3f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "data", "actual_vs_predicted.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved actual_vs_predicted.png")

    # ── Save model bundle ──
    bundle = {
        "model": gb_model,
        "features": FEATURES,
        "train_means": X_train.mean().to_dict(),
        "metrics": {
            "mae": gb_metrics["mae"],
            "rmse": gb_metrics["rmse"],
            "r2": gb_metrics["r2"],
        },
        "train_years": sorted(train_df["season"].unique().tolist()),
        "test_fraction": TEST_FRACTION,
        "baseline_mae": baseline_metrics["mae"],
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Gradient Boosting MAE {gb_metrics['mae']:.4f} fill rate vs Ridge MAE {baseline_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
