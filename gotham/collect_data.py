"""
collect_data.py
---------------
Pulls NWSL match + attendance data from the American Soccer Analysis API
(via itscalledsoccer) and joins historical weather from Open-Meteo.
Outputs data/nwsl_matches.csv.

COVID note: 2020 and 2021 are intentionally excluded.
  - 2020: bubble season with zero fans
  - 2021: restricted/inconsistent capacity throughout the season
Both years would corrupt the attendance model.

Run with:  python3.10 collect_data.py
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from itscalledsoccer.client import AmericanSoccerAnalysis

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2016, 2020)) + list(range(2022, 2026))  # excludes 2020-2021
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "nwsl_matches.csv")

# Fallback coords if a stadium has no lat/lon in the API
FALLBACK_COORDS = {
    "NJ/NY Gotham FC": (40.7367, -74.1503),  # Red Bull Arena, Harrison NJ
    "default": (39.8283, -98.5795),           # geographic center of US
}


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_nwsl_data() -> pd.DataFrame:
    """Pull games, teams, and stadia from the ASA API and return a joined DataFrame."""
    print("Connecting to American Soccer Analysis API...")
    asa = AmericanSoccerAnalysis()

    print("  Fetching games...")
    games = asa.get_games(leagues="nwsl")

    print("  Fetching teams...")
    teams = asa.get_teams(leagues="nwsl")

    print("  Fetching stadia...")
    stadia = asa.get_stadia(leagues="nwsl")

    print(f"  Raw games: {len(games)}")

    # ── Parse date and kickoff time ──
    games["date_time_utc"] = pd.to_datetime(games["date_time_utc"], utc=True)
    # Convert to Eastern Time (NWSL is primarily US-based)
    games["date_et"] = games["date_time_utc"].dt.tz_convert("America/New_York")
    games["date"] = games["date_et"].dt.date
    games["kickoff_hour"] = games["date_et"].dt.hour
    games["is_evening"] = (games["kickoff_hour"] >= 17).astype(int)

    # ── Filter to training years only ──
    games["season"] = games["season_name"]
    games = games[games["season"].isin(TRAIN_YEARS)].copy()
    print(f"  After year filter ({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}, excl. 2020-21): {len(games)}")

    # ── Drop zero/missing attendance ──
    games = games[games["attendance"] > 0].copy()
    print(f"  After dropping zero attendance: {len(games)}")

    # ── Join team names ──
    team_map = teams.set_index("team_id")["team_name"].to_dict()
    games["home_team"] = games["home_team_id"].map(team_map).fillna("Unknown")
    games["away_team"] = games["away_team_id"].map(team_map).fillna("Unknown")

    # Normalize Gotham name (Sky Blue FC became NJ/NY Gotham FC — same franchise)
    games["home_team"] = games["home_team"].replace(
        {"NJ/NY Gotham FC": "Gotham FC", "Sky Blue FC": "Gotham FC"}
    )
    games["away_team"] = games["away_team"].replace(
        {"NJ/NY Gotham FC": "Gotham FC", "Sky Blue FC": "Gotham FC"}
    )

    # ── Join stadium data ──
    stadia_slim = stadia[
        ["stadium_id", "stadium_name", "latitude", "longitude", "capacity"]
    ].drop_duplicates("stadium_id")
    games = games.merge(stadia_slim, on="stadium_id", how="left")

    # Fill missing stadium coords with team fallback
    def fill_coords(row):
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            return row["latitude"], row["longitude"]
        coords = FALLBACK_COORDS.get(row["home_team"], FALLBACK_COORDS["default"])
        return coords

    coords = games.apply(fill_coords, axis=1)
    games["lat"] = [c[0] for c in coords]
    games["lon"] = [c[1] for c in coords]

    # Fill missing capacity with median per team
    median_cap = games.groupby("home_team")["capacity"].median()
    games["venue_capacity"] = games.apply(
        lambda r: r["capacity"] if pd.notna(r["capacity"])
        else median_cap.get(r["home_team"], 10000),
        axis=1,
    ).fillna(10000).astype(int)

    cols = [
        "season", "date", "home_team", "away_team",
        "home_score", "away_score", "attendance",
        "is_evening", "lat", "lon", "venue_capacity",
        "stadium_name",
    ]
    return games[cols].reset_index(drop=True)


# ── Weather Fetching ────────────────────────────────────────────────────────────

def fetch_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature_f and precipitation_mm columns using Open-Meteo Historical API.
    Groups by (lat, lon) to minimize API calls.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (round(row["lat"], 2), round(row["lon"], 2))
        groups[key].append(str(row["date"]))

    weather_records = []
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    total = len(groups)

    for i, ((lat, lon), dates) in enumerate(groups.items(), 1):
        print(f"  Weather fetch {i}/{total}: ({lat:.2f}, {lon:.2f}) — {len(dates)} dates")
        dates_sorted = sorted(set(dates))
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": dates_sorted[0],
            "end_date": dates_sorted[-1],
            "daily": "temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("daily", {})
            date_set = set(dates_sorted)
            for d, t, p in zip(
                data.get("time", []),
                data.get("temperature_2m_max", []),
                data.get("precipitation_sum", []),
            ):
                if d in date_set:
                    weather_records.append({
                        "date": d,
                        "lat": lat,
                        "lon": lon,
                        "temperature_f": t,
                        "precipitation_mm": p if p is not None else 0.0,
                    })
        except Exception as e:
            print(f"    WARNING: weather fetch failed: {e}")

        time.sleep(0.4)

    wx = pd.DataFrame(weather_records)
    if wx.empty:
        df["temperature_f"] = 65.0
        df["precipitation_mm"] = 0.0
        return df

    wx["date"] = wx["date"].astype(str)
    wx["lat"] = wx["lat"].round(2)
    wx["lon"] = wx["lon"].round(2)

    df["date_str"] = df["date"].astype(str)
    df["lat_r"] = df["lat"].round(2)
    df["lon_r"] = df["lon"].round(2)
    df = df.merge(wx, left_on=["date_str", "lat_r", "lon_r"],
                  right_on=["date", "lat", "lon"], how="left", suffixes=("", "_wx"))
    df = df.drop(columns=["date_str", "lat_r", "lon_r",
                           "date_wx", "lat_wx", "lon_wx"], errors="ignore")

    # Fill any remaining NaN with medians
    df["temperature_f"] = df["temperature_f"].fillna(df["temperature_f"].median())
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Season progress (normalized position in home schedule)
    df = df.sort_values(["season", "home_team", "date"])
    df["home_game_num"] = df.groupby(["season", "home_team"]).cumcount() + 1
    df["home_games_total"] = df.groupby(["season", "home_team"])["home_game_num"].transform("max")
    df["season_progress"] = df["home_game_num"] / df["home_games_total"]

    # Rivalry flag (historically high-draw opponents for Gotham)
    RIVALRIES = {"Portland Thorns FC", "North Carolina Courage", "Chicago Red Stars",
                 "Chicago Stars FC"}
    df["is_rivalry"] = df["away_team"].isin(RIVALRIES).astype(int)

    # Rolling home win rate (last 5 home games, lag-1 to avoid leakage)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["home_win_rate_last5"] = (
        df.groupby("home_team")["home_win"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.5)
    )

    # Opponent season win rate (all games, prior to this one, to avoid leakage)
    # Build long-form results table
    home_r = df[["season", "date", "home_team", "home_win"]].rename(
        columns={"home_team": "team", "home_win": "win"}
    )
    away_r = df[["season", "date", "away_team", "home_win"]].copy()
    away_r["win"] = 1 - away_r["home_win"]
    away_r = away_r.rename(columns={"away_team": "team"}).drop(columns="home_win")
    all_r = pd.concat([home_r, away_r], ignore_index=True).sort_values("date")

    all_r["opp_win_rate"] = (
        all_r.groupby(["season", "team"])["win"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    opp_map = (
        all_r.drop_duplicates(subset=["date", "team"], keep="last")
        [["date", "team", "opp_win_rate"]]
        .rename(columns={"team": "away_team"})
    )
    df = df.merge(opp_map, on=["date", "away_team"], how="left")
    df["opp_win_rate_season"] = df["opp_win_rate"].fillna(0.5)
    df = df.drop(columns=["opp_win_rate"], errors="ignore")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Fetch match data
    print("\n=== Step 1: Fetching NWSL match data ===")
    df = fetch_nwsl_data()
    print(f"\nGames by season:\n{df.groupby('season').size()}")

    # 2. Fetch weather
    print("\n=== Step 2: Fetching historical weather ===")
    df = fetch_weather(df)
    print(f"Weather coverage: {df['temperature_f'].notna().sum()} / {len(df)} games")

    # 3. Add engineered features
    print("\n=== Step 3: Engineering features ===")
    df = add_features(df)

    # 4. Add fill rate target and final cleanup
    df["attendance_pct"] = df["attendance"] / df["venue_capacity"]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n=== Done ===")
    print(f"Saved {len(df)} games to {OUTPUT_FILE}")
    print(f"\nSample stats:")
    print(df[["season", "attendance", "temperature_f", "precipitation_mm",
              "home_win_rate_last5", "opp_win_rate_season"]].describe().round(1))


if __name__ == "__main__":
    main()
