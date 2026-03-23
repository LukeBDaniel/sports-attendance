"""
app.py — Gotham FC Attendance Predictor
----------------------------------------
Streamlit app that predicts home attendance for an upcoming Gotham FC game.
Weather is auto-fetched from Open-Meteo when the selected date is within
the 16-day forecast window; otherwise the user can enter it manually.
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "data", "nwsl_matches.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
RED_BULL_ARENA_CAPACITY = 25_189
GOTHAM_LAT = 40.7369
GOTHAM_LON = -74.0742

NWSL_TEAMS = [
    "Angel City FC",
    "Bay FC",
    "Boston Breakers",
    "Chicago Red Stars",
    "Houston Dash",
    "Kansas City Current",
    "North Carolina Courage",
    "Orlando Pride",
    "Portland Thorns FC",
    "Racing Louisville FC",
    "San Diego Wave FC",
    "Seattle Reign FC",
    "Utah Royals FC",
    "Washington Spirit",
]

RIVALRIES = {"Portland Thorns FC", "North Carolina Courage", "Chicago Red Stars"}

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    bundle = joblib.load(MODEL_FILE)
    # Recreate SHAP explainer from the model at runtime — avoids numba pickle
    # incompatibility across Python/library versions on different environments.
    if bundle.get("explainer") is None:
        bundle["explainer"] = shap.TreeExplainer(bundle["model"])
    return bundle


@st.cache_data
def load_history():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    return df[df["home_team"] == "Gotham FC"].copy()


# ── Weather helpers ────────────────────────────────────────────────────────────
def fetch_forecast_weather(date: datetime.date) -> dict | None:
    """
    Fetch Open-Meteo forecast for Red Bull Arena on the given date.
    Returns dict with temperature_f and precipitation_mm, or None if unavailable.
    """
    today = datetime.date.today()
    days_ahead = (date - today).days
    if days_ahead < 0 or days_ahead > 15:
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": GOTHAM_LAT,
        "longitude": GOTHAM_LON,
        "daily": "temperature_2m_max,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": 16,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])
        precips = daily.get("precipitation_sum", [])
        target = date.strftime("%Y-%m-%d")
        for d, t, p in zip(dates, temps, precips):
            if d == target:
                return {
                    "temperature_f": round(t, 1) if t is not None else None,
                    "precipitation_mm": round(p, 1) if p is not None else 0.0,
                }
    except Exception:
        pass
    return None


# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(
    date: datetime.date,
    opponent: str,
    kickoff_evening: bool,
    temperature_f: float,
    precipitation_mm: float,
    home_win_rate: float,
    opp_win_rate: float,
    season_progress: float,
    features: list[str],
) -> pd.DataFrame:
    is_rivalry = int(opponent in RIVALRIES)
    row = {
        "season": date.year,
        "day_of_week": date.weekday(),
        "month": date.month,
        "is_weekend": int(date.weekday() >= 5),
        "is_evening": int(kickoff_evening),
        "temperature_f": temperature_f,
        "precipitation_mm": precipitation_mm,
        "home_win_rate_last5": home_win_rate,
        "opp_win_rate_season": opp_win_rate,
        "is_rivalry": is_rivalry,
        "venue_capacity": RED_BULL_ARENA_CAPACITY,
        "season_progress": season_progress,
    }
    return pd.DataFrame([row])[features]


# ── SHAP waterfall chart ───────────────────────────────────────────────────────
def shap_waterfall_fig(explainer, X_row: pd.DataFrame) -> plt.Figure:
    shap_vals = explainer(X_row)
    fig, ax = plt.subplots(figsize=(7, 4))

    # Build a waterfall manually for clean display
    vals = shap_vals.values[0]
    base = float(shap_vals.base_values[0])
    feat_names = X_row.columns.tolist()

    # Sort by absolute value
    order = np.argsort(np.abs(vals))[::-1]
    vals_sorted = vals[order]
    names_sorted = [feat_names[i] for i in order]

    # Friendly label map
    label_map = {
        "season": "Season (League Growth)",
        "day_of_week": "Day of Week",
        "month": "Month",
        "is_weekend": "Weekend Game",
        "is_evening": "Evening Kickoff",
        "temperature_f": "Temperature (°F)",
        "precipitation_mm": "Precipitation (mm)",
        "home_win_rate_last5": "Gotham Recent Form",
        "opp_win_rate_season": "Opponent Strength",
        "is_rivalry": "Rivalry Match",
        "venue_capacity": "Venue Capacity",
        "season_progress": "Season Progress",
    }
    names_sorted = [label_map.get(n, n) for n in names_sorted]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals_sorted]
    ax.barh(names_sorted[::-1], vals_sorted[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on predicted fill rate)")
    ax.set_title("What's driving this prediction?", fontsize=12, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Gotham FC Attendance Predictor",
        page_icon="⚽",
        layout="wide",
    )

    # Header
    st.title("⚽ Gotham FC Attendance Predictor")
    st.markdown(
        "Predict home attendance for upcoming Gotham FC games at Red Bull Arena. "
        "Trained on NWSL data from 2013–2024 *(2020–2021 excluded — COVID-era capacity restrictions)*."
    )

    bundle = load_model()
    history = load_history()

    if bundle is None:
        st.error(
            "Model not found. Run `python train_model.py` first, "
            "then restart the app."
        )
        return

    model = bundle["model"]
    explainer = bundle["explainer"]
    features = bundle["features"]
    metrics = bundle["metrics"]

    # ── Sidebar inputs ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Upcoming Game")

        opponent = st.selectbox("Opponent", NWSL_TEAMS, index=0)

        game_date = st.date_input(
            "Game Date",
            value=datetime.date.today() + datetime.timedelta(days=14),
            min_value=datetime.date(2025, 1, 1),
        )

        kickoff_evening = st.checkbox("Evening kickoff (after 5 PM)?", value=True)

        # Auto-fetch weather if within forecast window
        st.subheader("Weather")
        wx = fetch_forecast_weather(game_date)
        if wx:
            st.success(f"Weather auto-fetched for {game_date}")
            default_temp = wx["temperature_f"] or 65.0
            default_precip = wx["precipitation_mm"] or 0.0
        else:
            st.info("Enter weather manually (date outside 16-day forecast window)")
            default_temp = 65.0
            default_precip = 0.0

        temperature_f = st.slider(
            "High Temperature (°F)", min_value=20, max_value=105,
            value=int(default_temp)
        )
        precipitation_mm = st.slider(
            "Precipitation (mm)", min_value=0.0, max_value=50.0,
            value=float(default_precip), step=0.5,
        )

        # Team form inputs
        st.subheader("Team Context")
        home_win_rate = st.slider(
            "Gotham win rate (last 5 home games)", 0.0, 1.0, 0.5, 0.1,
            help="0 = lost all 5, 1 = won all 5",
        )
        opp_win_rate = st.slider(
            "Opponent season win rate", 0.0, 1.0, 0.5, 0.1,
            help="How strong is the opponent this season?",
        )
        season_progress = st.slider(
            "Season progress (game # / total home games)", 0.0, 1.0, 0.5, 0.05,
            help="0.0 = opener, 1.0 = final home game",
        )

    # ── Prediction ──────────────────────────────────────────────────────────────
    X = build_features(
        date=game_date,
        opponent=opponent,
        kickoff_evening=kickoff_evening,
        temperature_f=float(temperature_f),
        precipitation_mm=float(precipitation_mm),
        home_win_rate=home_win_rate,
        opp_win_rate=opp_win_rate,
        season_progress=season_progress,
        features=features,
    )

    # Model predicts fill rate (0–1); convert to absolute attendance for display
    fill_rate = float(model.predict(X)[0])
    fill_rate = max(0.0, min(fill_rate, 1.0))
    prediction = fill_rate * RED_BULL_ARENA_CAPACITY
    pct_capacity = fill_rate * 100
    mae_fans = int(metrics["mae"] * RED_BULL_ARENA_CAPACITY)

    # ── Layout: 3 columns for the prediction card ───────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.metric(
            label="Predicted Attendance",
            value=f"{prediction:,.0f}",
            delta=f"± {mae_fans:,} (model MAE)",
        )
        st.progress(fill_rate)
        st.caption(f"{pct_capacity:.1f}% of Red Bull Arena capacity ({RED_BULL_ARENA_CAPACITY:,})")

    with col2:
        rivalry_text = "**Yes — rivalry boost expected**" if opponent in RIVALRIES else "No"
        weekend_text = "Yes" if game_date.weekday() >= 5 else "No"
        st.markdown("**Game Summary**")
        st.markdown(f"- Opponent: **{opponent}**")
        st.markdown(f"- Date: **{game_date.strftime('%A, %B %-d, %Y')}**")
        st.markdown(f"- Weekend game: {weekend_text}")
        st.markdown(f"- Rivalry match: {rivalry_text}")

    with col3:
        st.markdown("**Model Performance (random 20% holdout)**")
        st.markdown(f"- MAE: **{mae_fans:,} fans** ({metrics['mae']*100:.1f}% of capacity)")
        st.markdown(f"- R²: **{metrics['r2']:.3f}**")

    st.divider()

    # ── SHAP chart + historical context (side by side) ──────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("What's Driving This Prediction?")
        st.markdown(
            "Each bar shows how much a feature pushes the predicted fill rate "
            "above (green) or below (red) the league average. "
            "Multiply by 25,189 to convert to fans."
        )
        shap_fig = shap_waterfall_fig(explainer, X)
        st.pyplot(shap_fig, use_container_width=True)
        plt.close(shap_fig)

    with right:
        st.subheader("Gotham FC — Attendance History")
        if not history.empty:
            seasonal = (
                history.groupby("season")["attendance"]
                .agg(["mean", "max", "count"])
                .reset_index()
            )
            seasonal.columns = ["Season", "Avg Attendance", "Peak Attendance", "Games"]

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(seasonal["Season"], seasonal["Avg Attendance"],
                    color="#1a1a2e", alpha=0.85, label="Season Average")
            ax2.plot(seasonal["Season"], seasonal["Peak Attendance"],
                     "o--", color="#e94560", linewidth=1.5, label="Season Peak")
            ax2.axhline(prediction, color="#2ecc71", linewidth=2,
                        linestyle="--", label=f"This prediction: {prediction:,.0f}")
            ax2.set_xlabel("Season")
            ax2.set_ylabel("Attendance")
            ax2.set_title("Gotham FC Home Attendance by Season", fontsize=11)
            ax2.legend(fontsize=8)
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.info("Historical Gotham FC data not loaded (run collect_data.py first).")

    # ── Methodology note ────────────────────────────────────────────────────────
    with st.expander("Methodology"):
        st.markdown("""
**Model:** XGBoost regression trained on NWSL match data (2013–2019, 2022–2024).
2020 and 2021 are excluded because the 2020 season was played in a fan-free bubble
and 2021 operated under COVID-era capacity restrictions — including those years would
bias the model toward lower predictions.

**Features used:**
- Scheduling: day of week, month, evening kickoff, season progress
- Weather: max temperature (°F), precipitation (mm) for Red Bull Arena (Harrison, NJ)
- Team performance: Gotham's win rate over last 5 home games, opponent's season win rate
- Opponent context: rivalry flag (Portland Thorns, NC Courage, Chicago Red Stars)
- Venue: Red Bull Arena capacity (25,189)

**Train/test split:** trained on 2013–2022, tested on 2023–2024 to prevent data leakage.

**Weather source:** [Open-Meteo](https://open-meteo.com/) — free, no API key required.
Historical weather is matched to each team's home venue coordinates.
        """)


if __name__ == "__main__":
    main()
