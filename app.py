"""
app.py — Sports Attendance Predictor
--------------------------------------
Unified Streamlit app for predicting home attendance across teams.
Currently supports: Gotham FC (NWSL), New York Yankees (MLB),
NYCFC (MLS), Knicks (NBA), Rangers (NHL), Islanders (NHL),
Devils (NHL), Giants (NFL).
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(__file__)

# ── Team configurations ────────────────────────────────────────────────────────
TEAMS = {
    "⚽ Gotham FC": {
        "model_path": os.path.join(BASE_DIR, "gotham", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "gotham", "data", "nwsl_matches.csv"),
        "home_team_col": "home_team",
        "home_team_name": "Gotham FC",
        "stadium": "Red Bull Arena",
        "capacity": 25_189,
        "lat": 40.7369,
        "lon": -74.0742,
        "primary_color": "#1a1a2e",
        "accent_color":  "#e94560",
        "sport": "soccer",
        "opponents": [
            "Angel City FC", "Bay FC", "Boston Breakers", "Chicago Red Stars",
            "Houston Dash", "Kansas City Current", "North Carolina Courage",
            "Orlando Pride", "Portland Thorns FC", "Racing Louisville FC",
            "San Diego Wave FC", "Seattle Reign FC", "Utah Royals FC",
            "Washington Spirit",
        ],
        "rivalries": {"Portland Thorns FC", "North Carolina Courage", "Chicago Red Stars"},
        "evening_label": "Evening kickoff (after 5 PM)?",
        "win_rate_label": "Gotham win rate (last 5 home games)",
        "data_years": "2016–2025 (2020–2021 excluded)",
        "capacity_note": "Red Bull Arena capacity (25,189)",
        "shap_labels": {
            "home_win_rate_last5": "Gotham Recent Form",
            "is_rivalry": "Rivalry Match",
            "is_evening": "Evening Kickoff",
        },
        "history_title": "Gotham FC Home Attendance by Season",
    },
    "⚾ Yankees": {
        "model_path": os.path.join(BASE_DIR, "yankees", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "yankees", "data", "yankees_games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "Yankee Stadium",
        "capacity": 46_537,
        "lat": 40.8296,
        "lon": -73.9262,
        "primary_color": "#132448",
        "accent_color":  "#C4011D",
        "sport": "baseball",
        "opponents": [
            "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles",
            "Boston Red Sox", "Chicago Cubs", "Chicago White Sox",
            "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies",
            "Detroit Tigers", "Houston Astros", "Kansas City Royals",
            "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
            "Milwaukee Brewers", "Minnesota Twins", "New York Mets",
            "Oakland Athletics", "Philadelphia Phillies", "Pittsburgh Pirates",
            "San Diego Padres", "Seattle Mariners", "San Francisco Giants",
            "St. Louis Cardinals", "Tampa Bay Rays", "Texas Rangers",
            "Toronto Blue Jays", "Washington Nationals",
        ],
        "rivalries": {"Boston Red Sox", "New York Mets"},
        "evening_label": "Night game?",
        "win_rate_label": "Yankees win rate (last 5 home games)",
        "data_years": "2000–2025 (2020–2021 excluded)",
        "capacity_note": "Yankee Stadium capacity (46,537)",
        "shap_labels": {
            "home_win_rate_last5": "Yankees Recent Form",
            "is_rivalry": "Rivalry Game",
            "is_evening": "Night Game",
        },
        "history_title": "Yankees Home Attendance by Season",
    },
    "⚽ NYCFC": {
        "model_path": os.path.join(BASE_DIR, "nycfc", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "nycfc", "data", "games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "Yankee Stadium (soccer config)",
        "capacity": 28_743,
        "lat": 40.8296,
        "lon": -73.9262,
        "primary_color": "#9FD2FF",
        "accent_color":  "#000229",
        "sport": "soccer",
        "opponents": [
            "Atlanta United FC", "Austin FC", "Charlotte FC", "Chicago Fire FC",
            "Colorado Rapids", "Columbus Crew", "D.C. United", "FC Cincinnati",
            "FC Dallas", "Houston Dynamo FC", "Inter Miami CF", "LA Galaxy",
            "LAFC", "Minnesota United FC", "CF Montréal", "Nashville SC",
            "New England Revolution", "New York Red Bulls", "Orlando City SC",
            "Philadelphia Union", "Portland Timbers", "Real Salt Lake",
            "San Jose Earthquakes", "Seattle Sounders FC", "Sporting Kansas City",
            "St. Louis City SC", "Toronto FC", "Vancouver Whitecaps FC",
        ],
        "rivalries": {"New York Red Bulls", "Philadelphia Union", "New England Revolution"},
        "evening_label": "Evening kickoff (after 5 PM)?",
        "win_rate_label": "NYCFC win rate (last 5 home games)",
        "data_years": "2015–2025 (2020–2021 excluded)",
        "capacity_note": "Yankee Stadium soccer capacity (28,743)",
        "shap_labels": {
            "home_win_rate_last5": "NYCFC Recent Form",
            "is_rivalry": "Rivalry Match",
            "is_evening": "Evening Kickoff",
        },
        "history_title": "NYCFC Home Attendance by Season",
    },
    "🏀 Knicks": {
        "model_path": os.path.join(BASE_DIR, "knicks", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "knicks", "data", "games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "Madison Square Garden",
        "capacity": 19_812,
        "lat": 40.7505,
        "lon": -73.9934,
        "primary_color": "#006BB6",
        "accent_color":  "#F58426",
        "sport": "basketball",
        "opponents": [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
            "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers",
            "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
            "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards",
        ],
        "rivalries": {"Boston Celtics", "Miami Heat", "Chicago Bulls", "Brooklyn Nets", "Indiana Pacers"},
        "evening_label": "Evening tip-off (after 5 PM)?",
        "win_rate_label": "Knicks win rate (last 5 home games)",
        "data_years": "2001–2025 (2020–2021 excluded)",
        "capacity_note": "Madison Square Garden capacity (19,812)",
        "shap_labels": {
            "home_win_rate_last5": "Knicks Recent Form",
            "is_rivalry": "Rivalry Game",
            "is_evening": "Evening Tip-off",
        },
        "history_title": "Knicks Home Attendance by Season",
    },
    "🏒 Rangers": {
        "model_path": os.path.join(BASE_DIR, "rangers", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "rangers", "data", "games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "Madison Square Garden",
        "capacity": 18_006,
        "lat": 40.7505,
        "lon": -73.9934,
        "primary_color": "#0038A8",
        "accent_color":  "#CE1126",
        "sport": "hockey",
        "opponents": [
            "Anaheim Ducks", "Arizona Coyotes", "Boston Bruins", "Buffalo Sabres",
            "Calgary Flames", "Carolina Hurricanes", "Chicago Blackhawks",
            "Colorado Avalanche", "Columbus Blue Jackets", "Dallas Stars",
            "Detroit Red Wings", "Edmonton Oilers", "Florida Panthers",
            "Los Angeles Kings", "Minnesota Wild", "Montréal Canadiens",
            "Nashville Predators", "New Jersey Devils", "New York Islanders",
            "Ottawa Senators", "Philadelphia Flyers", "Pittsburgh Penguins",
            "Seattle Kraken", "San Jose Sharks", "St. Louis Blues",
            "Tampa Bay Lightning", "Toronto Maple Leafs", "Utah Hockey Club",
            "Vancouver Canucks", "Vegas Golden Knights", "Washington Capitals",
            "Winnipeg Jets",
        ],
        "rivalries": {"New Jersey Devils", "New York Islanders", "Pittsburgh Penguins", "Philadelphia Flyers", "Washington Capitals"},
        "evening_label": "Evening puck drop (after 5 PM)?",
        "win_rate_label": "Rangers win rate (last 5 home games)",
        "data_years": "2001–2025 (2020–2021 excluded)",
        "capacity_note": "Madison Square Garden capacity (18,006)",
        "shap_labels": {
            "home_win_rate_last5": "Rangers Recent Form",
            "is_rivalry": "Rivalry Game",
            "is_evening": "Evening Puck Drop",
        },
        "history_title": "Rangers Home Attendance by Season",
    },
    "🏒 Islanders": {
        "model_path": os.path.join(BASE_DIR, "islanders", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "islanders", "data", "games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "UBS Arena",
        "capacity": 17_255,
        "lat": 40.7221,
        "lon": -73.7202,
        "primary_color": "#00539B",
        "accent_color":  "#FC4C02",
        "sport": "hockey",
        "opponents": [
            "Anaheim Ducks", "Arizona Coyotes", "Boston Bruins", "Buffalo Sabres",
            "Calgary Flames", "Carolina Hurricanes", "Chicago Blackhawks",
            "Colorado Avalanche", "Columbus Blue Jackets", "Dallas Stars",
            "Detroit Red Wings", "Edmonton Oilers", "Florida Panthers",
            "Los Angeles Kings", "Minnesota Wild", "Montréal Canadiens",
            "Nashville Predators", "New Jersey Devils", "New York Rangers",
            "Ottawa Senators", "Philadelphia Flyers", "Pittsburgh Penguins",
            "Seattle Kraken", "San Jose Sharks", "St. Louis Blues",
            "Tampa Bay Lightning", "Toronto Maple Leafs", "Utah Hockey Club",
            "Vancouver Canucks", "Vegas Golden Knights", "Washington Capitals",
            "Winnipeg Jets",
        ],
        "rivalries": {"New York Rangers", "New Jersey Devils", "Pittsburgh Penguins", "Philadelphia Flyers"},
        "evening_label": "Evening puck drop (after 5 PM)?",
        "win_rate_label": "Islanders win rate (last 5 home games)",
        "data_years": "2001–2025 (2020–2021 excluded)",
        "capacity_note": "UBS Arena capacity (17,255)",
        "shap_labels": {
            "home_win_rate_last5": "Islanders Recent Form",
            "is_rivalry": "Rivalry Game",
            "is_evening": "Evening Puck Drop",
        },
        "history_title": "Islanders Home Attendance by Season",
    },
    "🏒 Devils": {
        "model_path": os.path.join(BASE_DIR, "devils", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "devils", "data", "games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "Prudential Center",
        "capacity": 16_514,
        "lat": 40.7334,
        "lon": -74.1713,
        "primary_color": "#CE1126",
        "accent_color":  "#000000",
        "sport": "hockey",
        "opponents": [
            "Anaheim Ducks", "Arizona Coyotes", "Boston Bruins", "Buffalo Sabres",
            "Calgary Flames", "Carolina Hurricanes", "Chicago Blackhawks",
            "Colorado Avalanche", "Columbus Blue Jackets", "Dallas Stars",
            "Detroit Red Wings", "Edmonton Oilers", "Florida Panthers",
            "Los Angeles Kings", "Minnesota Wild", "Montréal Canadiens",
            "Nashville Predators", "New York Islanders", "New York Rangers",
            "Ottawa Senators", "Philadelphia Flyers", "Pittsburgh Penguins",
            "Seattle Kraken", "San Jose Sharks", "St. Louis Blues",
            "Tampa Bay Lightning", "Toronto Maple Leafs", "Utah Hockey Club",
            "Vancouver Canucks", "Vegas Golden Knights", "Washington Capitals",
            "Winnipeg Jets",
        ],
        "rivalries": {"New York Rangers", "New York Islanders", "Philadelphia Flyers", "Pittsburgh Penguins"},
        "evening_label": "Evening puck drop (after 5 PM)?",
        "win_rate_label": "Devils win rate (last 5 home games)",
        "data_years": "2001–2025 (2020–2021 excluded)",
        "capacity_note": "Prudential Center capacity (16,514)",
        "shap_labels": {
            "home_win_rate_last5": "Devils Recent Form",
            "is_rivalry": "Rivalry Game",
            "is_evening": "Evening Puck Drop",
        },
        "history_title": "Devils Home Attendance by Season",
    },
    "🏈 Giants": {
        "model_path": os.path.join(BASE_DIR, "giants", "model.pkl"),
        "data_path":  os.path.join(BASE_DIR, "giants", "data", "games.csv"),
        "home_team_col": None,
        "home_team_name": None,
        "stadium": "MetLife Stadium",
        "capacity": 82_500,
        "lat": 40.8135,
        "lon": -74.0745,
        "primary_color": "#0B2265",
        "accent_color":  "#A71930",
        "sport": "football",
        "opponents": [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens",
            "Buffalo Bills", "Carolina Panthers", "Chicago Bears",
            "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys",
            "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
            "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
            "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings",
            "New England Patriots", "New Orleans Saints", "New York Jets",
            "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans",
            "Washington Commanders",
        ],
        "rivalries": {"Philadelphia Eagles", "Dallas Cowboys", "Washington Commanders", "New York Jets"},
        "evening_label": "Primetime game (after 5 PM kickoff)?",
        "win_rate_label": "Giants win rate (last 5 home games)",
        "data_years": "2000–2025 (2020 excluded)",
        "capacity_note": "MetLife Stadium capacity (82,500)",
        "shap_labels": {
            "home_win_rate_last5": "Giants Recent Form",
            "is_rivalry": "Rivalry Game",
            "is_evening": "Primetime Game",
        },
        "history_title": "Giants Home Attendance by Season",
    },
}

SHAP_BASE_LABELS = {
    "season":             "Season (Long-term Trend)",
    "day_of_week":        "Day of Week",
    "month":              "Month",
    "is_weekend":         "Weekend Game",
    "temperature_f":      "Temperature (°F)",
    "precipitation_mm":   "Precipitation (mm)",
    "opp_win_rate_season":"Opponent Strength",
    "venue_capacity":     "Venue Capacity",
    "season_progress":    "Season Progress",
}

YANKEES_DISPLAY_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}
YANKEES_RIVALRY_ABBRS = {"BOS", "NYM"}


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics(path: str) -> dict | None:
    """Lightweight loader — extracts metrics only, no SHAP. Used by the About page."""
    if not os.path.exists(path):
        return None
    return joblib.load(path).get("metrics")


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_history(path: str, home_team_col: str | None, home_team_name: str | None):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    if home_team_col and home_team_name:
        df = df[df[home_team_col] == home_team_name]
    return df.copy()


# ── Weather ────────────────────────────────────────────────────────────────────
def fetch_forecast_weather(date: datetime.date, lat: float, lon: float) -> dict | None:
    today = datetime.date.today()
    days_ahead = (date - today).days
    if days_ahead < 0 or days_ahead > 15:
        return None
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": 16,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        target = date.strftime("%Y-%m-%d")
        for d, t, p in zip(daily.get("time", []),
                            daily.get("temperature_2m_max", []),
                            daily.get("precipitation_sum", [])):
            if d == target:
                return {
                    "temperature_f": round(t, 1) if t is not None else None,
                    "precipitation_mm": round(p, 1) if p is not None else 0.0,
                }
    except Exception:
        pass
    return None


# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(cfg, date, opponent, is_evening, temperature_f,
                   precipitation_mm, home_win_rate, opp_win_rate,
                   season_progress, features) -> pd.DataFrame:
    # Rivalry check and capacity vary by sport
    if cfg["sport"] == "baseball":
        abbr = YANKEES_DISPLAY_TO_ABBR.get(opponent, opponent)
        is_rivalry = int(abbr in YANKEES_RIVALRY_ABBRS)
        capacity = 46_537 if date.year >= 2009 else 57_545
    else:
        is_rivalry = int(opponent in cfg["rivalries"])
        capacity = cfg["capacity"]

    row = {
        "season": date.year,
        "day_of_week": date.weekday(),
        "month": date.month,
        "is_weekend": int(date.weekday() >= 5),
        "is_evening": int(is_evening),
        "temperature_f": temperature_f,
        "precipitation_mm": precipitation_mm,
        "home_win_rate_last5": home_win_rate,
        "opp_win_rate_season": opp_win_rate,
        "is_rivalry": is_rivalry,
        "venue_capacity": capacity,
        "season_progress": season_progress,
    }
    return pd.DataFrame([row])[features]


# ── Contribution chart ─────────────────────────────────────────────────────────
def contribution_fig(model, X_row, train_means, cfg) -> plt.Figure:
    """
    Leave-one-out feature contributions: for each feature, replace its value
    with the training mean and measure the drop in predicted fill rate.
    This gives an interpretable per-prediction breakdown without any extra library.
    """
    features = X_row.columns.tolist()
    current_pred = float(model.predict(X_row)[0])

    mean_row = pd.DataFrame([{f: train_means.get(f, X_row[f].iloc[0]) for f in features}])
    base_pred = float(model.predict(mean_row)[0])

    contributions = {}
    for feat in features:
        perturbed = X_row.copy()
        perturbed[feat] = train_means.get(feat, X_row[feat].iloc[0])
        contributions[feat] = current_pred - float(model.predict(perturbed)[0])

    label_map = {**SHAP_BASE_LABELS, **cfg["shap_labels"]}
    contrib = pd.Series(contributions).rename(label_map)
    contrib = contrib.reindex(contrib.abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in contrib]
    ax.barh(contrib.index, contrib.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on predicted fill rate")
    ax.set_title("What's driving this prediction?", fontsize=12, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ── History chart ──────────────────────────────────────────────────────────────
def history_fig(history, prediction, cfg) -> plt.Figure | None:
    if history.empty:
        return None
    seasonal = (
        history.groupby("season")["attendance"]
        .agg(["mean", "max"])
        .reset_index()
    )
    seasonal.columns = ["Season", "Avg", "Peak"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(seasonal["Season"], seasonal["Avg"],
           color=cfg["primary_color"], alpha=0.85, label="Season Average")
    ax.plot(seasonal["Season"], seasonal["Peak"],
            "o--", color=cfg["accent_color"], linewidth=1.5, label="Season Peak")
    ax.axhline(prediction, color="#2ecc71", linewidth=2, linestyle="--",
               label=f"This prediction: {prediction:,.0f}")
    ax.set_xlabel("Season")
    ax.set_ylabel("Attendance")
    ax.set_title(cfg["history_title"], fontsize=11)
    ax.legend(fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Attendance Predictor",
        page_icon="🏟️",
        layout="wide",
    )

    TEAM_OPTIONS = ["ℹ️ About"] + list(TEAMS.keys())

    with st.sidebar:
        st.title("🏟️ Attendance Predictor")
        team_key = st.selectbox("Select team", TEAM_OPTIONS, index=0)

    # ── Info / landing screen ────────────────────────────────────────────────────
    if team_key == "ℹ️ About":
        st.title("🏟️ NY Area Sports Attendance Predictor")
        st.markdown(
            "This tool predicts **home game attendance** for 8 NY-area professional "
            "sports teams using machine learning. Select a team from the sidebar to get started."
        )
        st.divider()

        team_rows = []
        for key, c in TEAMS.items():
            metrics = load_metrics(c["model_path"])
            r2       = f"{metrics['r2']:.3f}" if metrics else "—"
            mae_fans = f"{int(metrics['mae'] * c['capacity']):,}" if metrics else "—"
            team_rows.append({
                "Team": key,
                "Stadium": c["stadium"],
                "Capacity": f"{c['capacity']:,}",
                "Training Data": c["data_years"],
                "MAE (fans)": mae_fans,
                "R²": r2,
            })
        st.subheader("Teams & Model Performance")
        st.dataframe(
            pd.DataFrame(team_rows).set_index("Team"),
            use_container_width=True,
        )

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("How it works")
            st.markdown("""
**Data collection**
- Game results and attendance pulled from the ESPN API and Baseball Reference (Yankees) / American Soccer Analysis (Gotham FC)
- Historical weather matched to each venue via [Open-Meteo](https://open-meteo.com/) (free, no key required)

**Model**
- Gradient Boosting Regressor (scikit-learn) trained on normalized fill rate (attendance ÷ capacity)
- Compared against a Ridge regression baseline
- COVID-era seasons excluded (no fans / restricted capacity)

**Explainability**
- Leave-one-out feature contributions show which factors drive each prediction above or below the historical average
            """)
        with col_b:
            st.subheader("Features used")
            st.markdown("""
| Feature | Description |
|---------|-------------|
| Season | Long-term trend in attendance |
| Day of week | Mon–Sun |
| Month | Seasonal patterns |
| Weekend game | Sat/Sun boost |
| Evening game | Night vs afternoon |
| Temperature (°F) | High temp at venue |
| Precipitation (mm) | Rain/snow impact |
| Recent form | Win rate, last 5 home games |
| Opponent strength | Opponent season win rate |
| Rivalry | Historically high-draw matchup |
| Venue capacity | Accounts for stadium changes |
| Season progress | Opener vs late-season game |
            """)

        st.divider()
        st.caption(
            "Data sources: ESPN API · Baseball Reference (pybaseball) · "
            "American Soccer Analysis (itscalledsoccer) · Open-Meteo"
        )
        return

    cfg = TEAMS[team_key]

    st.title(f"{team_key} Attendance Predictor")
    st.markdown(
        f"Predict home attendance at **{cfg['stadium']}**. "
        f"Trained on data from {cfg['data_years']}."
    )

    bundle = load_model(cfg["model_path"])
    history = load_history(cfg["data_path"], cfg["home_team_col"], cfg["home_team_name"])

    if bundle is None:
        st.error(
            f"Model not found at `{cfg['model_path']}`. "
            "Run `collect_data.py` then `train_model.py` in the team folder first."
        )
        return

    model       = bundle["model"]
    train_means = bundle.get("train_means", {})
    features    = bundle["features"]
    metrics     = bundle["metrics"]

    # ── Sidebar inputs ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Upcoming Game")

        opponent = st.selectbox("Opponent", cfg["opponents"])

        game_date = st.date_input(
            "Game Date",
            value=datetime.date.today() + datetime.timedelta(days=14),
            min_value=datetime.date(2025, 1, 1),
        )

        is_evening = st.checkbox(cfg["evening_label"], value=True)

        st.subheader("Weather")
        wx = fetch_forecast_weather(game_date, cfg["lat"], cfg["lon"])
        if wx:
            st.success(f"Weather auto-fetched for {game_date}")
            default_temp   = wx["temperature_f"] or 68.0
            default_precip = wx["precipitation_mm"] or 0.0
        else:
            st.info("Enter weather manually (date outside 16-day forecast window)")
            default_temp, default_precip = 68.0, 0.0

        temperature_f    = st.slider("High Temperature (°F)", 20, 105, int(default_temp))
        precipitation_mm = st.slider("Precipitation (mm)", 0.0, 50.0,
                                     float(default_precip), step=0.5)

        st.subheader("Team Context")
        home_win_rate = st.slider(cfg["win_rate_label"], 0.0, 1.0, 0.5, 0.1,
                                  help="0 = lost all 5, 1 = won all 5")
        opp_win_rate  = st.slider("Opponent season win rate", 0.0, 1.0, 0.5, 0.1)
        season_progress = st.slider("Season progress (game # / total home games)",
                                    0.0, 1.0, 0.5, 0.05,
                                    help="0.0 = opener, 1.0 = final home game")

    # ── Prediction ──────────────────────────────────────────────────────────────
    X = build_features(cfg, game_date, opponent, is_evening,
                       float(temperature_f), float(precipitation_mm),
                       home_win_rate, opp_win_rate, season_progress, features)

    fill_rate  = float(np.clip(model.predict(X)[0], 0.0, 1.0))
    capacity   = int(X["venue_capacity"].iloc[0])
    prediction = fill_rate * capacity
    mae_fans   = int(metrics["mae"] * capacity)

    # ── Prediction card ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.metric("Predicted Attendance", f"{prediction:,.0f}",
                  delta=f"± {mae_fans:,} (model MAE)")
        st.progress(fill_rate)
        st.caption(f"{fill_rate*100:.1f}% of {cfg['capacity_note']}")

    with col2:
        rivalry_text = ("**Yes — rivalry boost expected**"
                        if opponent in cfg["rivalries"] else "No")
        weekend_text = "Yes" if game_date.weekday() >= 5 else "No"
        st.markdown("**Game Summary**")
        st.markdown(f"- Opponent: **{opponent}**")
        st.markdown(f"- Date: **{game_date.strftime('%A, %B %-d, %Y')}**")
        st.markdown(f"- Weekend game: {weekend_text}")
        st.markdown(f"- Rivalry: {rivalry_text}")

    with col3:
        st.markdown("**Model Performance (random 20% holdout)**")
        st.markdown(f"- MAE: **{mae_fans:,} fans** ({metrics['mae']*100:.1f}% of capacity)")
        st.markdown(f"- R²: **{metrics['r2']:.3f}**")

    st.divider()

    # ── SHAP + history ──────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("What's Driving This Prediction?")
        st.markdown(
            "Each bar shows how much a feature pushes the predicted fill rate "
            "above (green) or below (red) the average."
        )
        fig_contrib = contribution_fig(model, X, train_means, cfg)
        st.pyplot(fig_contrib, use_container_width=True)
        plt.close(fig_contrib)

    with right:
        st.subheader("Attendance History")
        fig_hist = history_fig(history, prediction, cfg)
        if fig_hist:
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
        else:
            st.info("Historical data not loaded (run collect_data.py first).")


if __name__ == "__main__":
    main()
