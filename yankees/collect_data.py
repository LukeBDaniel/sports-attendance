"""
collect_data.py
---------------
Pulls Yankees home game + attendance data from pybaseball and joins
historical weather from Open-Meteo.
Outputs data/yankees_games.csv.

COVID note: 2020 and 2021 are intentionally excluded.
  - 2020: no fans allowed
  - 2021: limited/inconsistent capacity throughout the season
Both years would corrupt the attendance model.

Run with:  python collect_data.py
"""

import os
import re
import time
import datetime
import requests
import pandas as pd
import numpy as np
import pybaseball

pybaseball.cache.enable()

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2000, 2020)) + list(range(2022, 2026))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "yankees_games.csv")

# Yankee Stadium — new stadium opened 2009
YANKEE_STADIUM_LAT = 40.8296
YANKEE_STADIUM_LON = -73.9262

# Old Yankee Stadium (≤2008) held ~57,545; new stadium holds 46,537
def get_capacity(year: int) -> int:
    return 57_545 if year < 2009 else 46_537

# Rivalry opponents (high-draw games for the Yankees)
RIVALRIES = {"BOS", "NYM"}  # Red Sox + Mets (Subway Series)

# Map pybaseball team abbreviations → display names for the app
TEAM_NAME_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}


# ── Data Fetching ──────────────────────────────────────────────────────────────

def parse_game_date(date_str: str, year: int):
    """Parse pybaseball date strings like 'Monday, Apr 1' or 'Wednesday, Sep 10 (1)'."""
    # Strip doubleheader markers like "(1)" or "(2)"
    clean = re.sub(r'\s*\(\d+\)', '', str(date_str)).strip()
    try:
        return datetime.datetime.strptime(f"{clean} {year}", "%A, %b %d %Y").date()
    except ValueError:
        return None


def fetch_yankees_data() -> pd.DataFrame:
    """Pull Yankees home game records from pybaseball for all training years."""
    all_games = []

    for year in TRAIN_YEARS:
        print(f"  Fetching {year} Yankees schedule...")
        try:
            sched = pybaseball.schedule_and_record(year, "NYY")
        except Exception as e:
            print(f"    WARNING: failed to fetch {year}: {e}")
            continue

        if sched is None or sched.empty:
            print(f"    No data for {year}")
            continue

        # Home games: Home_Away column is "Home" for home games, "@" for away
        home_games = sched[sched["Home_Away"] == "Home"].copy()

        # Parse dates
        home_games["date"] = home_games["Date"].apply(lambda d: parse_game_date(d, year))
        home_games = home_games.dropna(subset=["date"])

        # Keep only rows with real attendance numbers
        # pybaseball returns attendance as floats (e.g. 46172.0) — convert via numeric
        home_games["attendance"] = pd.to_numeric(home_games["Attendance"], errors="coerce")
        home_games = home_games.dropna(subset=["attendance"])
        home_games["attendance"] = home_games["attendance"].astype(int)
        home_games = home_games[home_games["attendance"] > 0]

        # Day/night game
        home_games["is_evening"] = (home_games["D/N"] == "N").astype(int)

        # Win/loss (some rows have 'W' or 'L'; ties/postponements get dropped)
        home_games = home_games[home_games["W/L"].str.match(r"^[WL]", na=False)].copy()
        home_games["home_win"] = home_games["W/L"].str.startswith("W").astype(int)

        # Opponent abbreviation
        home_games["opp_abbr"] = home_games["Opp"].astype(str).str.strip()

        home_games["season"] = year
        home_games["venue_capacity"] = get_capacity(year)
        home_games["lat"] = YANKEE_STADIUM_LAT
        home_games["lon"] = YANKEE_STADIUM_LON

        cols = ["season", "date", "opp_abbr", "home_win", "attendance",
                "is_evening", "venue_capacity", "lat", "lon"]
        all_games.append(home_games[cols])

        time.sleep(0.3)

    if not all_games:
        raise RuntimeError("No Yankees game data retrieved — check pybaseball installation.")

    df = pd.concat(all_games, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Map abbreviation → display name
    df["opponent"] = df["opp_abbr"].map(TEAM_NAME_MAP).fillna(df["opp_abbr"])

    print(f"  Total home games fetched: {len(df)}")
    print(f"  Years: {sorted(df['season'].unique())}")
    return df


# ── Weather Fetching ────────────────────────────────────────────────────────────

def fetch_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature_f and precipitation_mm from Open-Meteo Historical API."""
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
        # Split into chunks of ~1 year to avoid overly large requests
        chunk_size = 365
        for start in range(0, len(dates_sorted), chunk_size):
            chunk = dates_sorted[start:start + chunk_size]
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": chunk[0],
                "end_date": chunk[-1],
                "daily": "temperature_2m_max,precipitation_sum",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
            }
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("daily", {})
                date_set = set(chunk)
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
            time.sleep(0.5)

    wx = pd.DataFrame(weather_records)
    if wx.empty:
        df["temperature_f"] = 72.0
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

    # Season progress (normalized home game position)
    df = df.sort_values(["season", "date"])
    df["home_game_num"] = df.groupby("season").cumcount() + 1
    df["home_games_total"] = df.groupby("season")["home_game_num"].transform("max")
    df["season_progress"] = df["home_game_num"] / df["home_games_total"]

    # Rivalry flag
    df["is_rivalry"] = df["opp_abbr"].isin(RIVALRIES).astype(int)

    # Rolling home win rate (last 5 home games, lag-1 to avoid leakage)
    df["home_win_rate_last5"] = (
        df["home_win"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
        .fillna(0.5)
    )

    # Opponent win rate: approximate using their record in games vs Yankees this season
    # (lag-1 expanding mean — same opponent may appear multiple times per season)
    opp_records = []
    for (season, opp), grp in df.groupby(["season", "opp_abbr"]):
        # From Yankees' perspective: opp won when Yankees lost
        opp_win = 1 - grp["home_win"]
        opp_cum_rate = opp_win.shift(1).expanding().mean().fillna(0.5)
        opp_records.append(pd.Series(opp_cum_rate.values, index=grp.index))
    df["opp_win_rate_season"] = pd.concat(opp_records).sort_index()
    df["opp_win_rate_season"] = df["opp_win_rate_season"].fillna(0.5)

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n=== Step 1: Fetching Yankees game data ===")
    df = fetch_yankees_data()
    print(f"\nGames by season:\n{df.groupby('season').size()}")

    print("\n=== Step 2: Fetching historical weather ===")
    df = fetch_weather(df)
    print(f"Weather coverage: {df['temperature_f'].notna().sum()} / {len(df)} games")

    print("\n=== Step 3: Engineering features ===")
    df = add_features(df)

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
