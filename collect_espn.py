"""
collect_espn.py  (shared utility)
----------------------------------
Generic ESPN API data collection pipeline used by all team collect_data.py wrappers.
Each team's collect_data.py imports and calls main_pipeline(CONFIG).
"""

import os, time, requests, numpy as np, pandas as pd
from datetime import datetime, timezone


# ── ESPN fetch ──────────────────────────────────────────────────────────────────

def fetch_espn_season(sport, league, slug, home_abbrev, year):
    """Fetch one season of home games from ESPN API. Returns list of dicts."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports"
        f"/{sport}/{league}/teams/{slug}/schedule?season={year}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"    WARNING: ESPN fetch failed for {year}: {e}")
        return []

    events = resp.json().get("events", [])
    games = []

    for event in events:
        comp = event.get("competitions", [{}])[0]

        # Find home and away competitors
        home_comp = next(
            (c for c in comp.get("competitors", [])
             if c.get("homeAway") == "home"
             and c["team"].get("abbreviation", "").upper() == home_abbrev.upper()),
            None,
        )
        away_comp = next(
            (c for c in comp.get("competitors", [])
             if c.get("homeAway") == "away"),
            None,
        )
        if not home_comp or not away_comp:
            continue

        # Attendance
        attendance = comp.get("attendance")
        if not attendance or attendance == 0:
            continue

        # Date + time in ET (parse ISO UTC string)
        raw_dt = event.get("date", "")
        try:
            utc_dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            # Convert to ET (rough: UTC-4 EDT / UTC-5 EST)
            # Use UTC-4 as approximation (covers most of sports seasons Apr-Nov)
            et_hour = (utc_dt.hour - 4) % 24
        except Exception:
            et_hour = 19  # default evening
        is_evening = int(et_hour >= 17)
        game_date = raw_dt[:10]  # YYYY-MM-DD

        # Skip unfinished games — use completed flag (works across all ESPN sports)
        status_type = comp.get("status", {}).get("type", {})
        if not status_type.get("completed", False):
            continue
        score_h = home_comp.get("score") or {}
        score_a = away_comp.get("score") or {}
        try:
            home_score = float(score_h.get("value", 0) if isinstance(score_h, dict) else score_h)
            away_score = float(score_a.get("value", 0) if isinstance(score_a, dict) else score_a)
        except (TypeError, ValueError):
            continue

        home_win = int(home_score > away_score)

        games.append({
            "season":     year,
            "date":       game_date,
            "opponent":   away_comp["team"].get("displayName", "Unknown"),
            "opp_abbr":   away_comp["team"].get("abbreviation", "UNK"),
            "home_score": home_score,
            "away_score": away_score,
            "home_win":   home_win,
            "attendance": int(attendance),
            "is_evening": is_evening,
        })

    return games


def fetch_all_seasons(sport, league, slug, home_abbrev, train_years):
    """Loop over all training years and return concatenated DataFrame."""
    all_games = []
    for year in train_years:
        print(f"  Fetching {year}...")
        games = fetch_espn_season(sport, league, slug, home_abbrev, year)
        print(f"    {len(games)} home games")
        all_games.extend(games)
        time.sleep(0.4)

    if not all_games:
        raise RuntimeError("No game data retrieved. Check ESPN API config.")

    df = pd.DataFrame(all_games)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Weather ─────────────────────────────────────────────────────────────────────

def fetch_weather(df, lat, lon):
    """Add temperature_f and precipitation_mm via Open-Meteo Historical API."""
    from collections import defaultdict

    dates_sorted = sorted(df["date"].dt.strftime("%Y-%m-%d").unique())
    weather_records = []
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    chunk_size = 365

    print(f"  Fetching weather for {len(dates_sorted)} dates...")
    for start in range(0, len(dates_sorted), chunk_size):
        chunk = dates_sorted[start : start + chunk_size]
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": chunk[0], "end_date": chunk[-1],
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
                        "temperature_f": t,
                        "precipitation_mm": p if p is not None else 0.0,
                    })
        except Exception as e:
            print(f"    WARNING: weather fetch failed: {e}")
        time.sleep(0.5)

    wx = pd.DataFrame(weather_records)
    if wx.empty:
        df["temperature_f"] = 68.0
        df["precipitation_mm"] = 0.0
        return df

    wx["date"] = pd.to_datetime(wx["date"])
    df = df.merge(wx, on="date", how="left")
    df["temperature_f"] = df["temperature_f"].fillna(df["temperature_f"].median())
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)
    return df


# ── Feature engineering ─────────────────────────────────────────────────────────

def add_features(df, rivalries, capacity_fn):
    """Add all ML features to the DataFrame."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Season progress
    df = df.sort_values(["season", "date"])
    df["home_game_num"]    = df.groupby("season").cumcount() + 1
    df["home_games_total"] = df.groupby("season")["home_game_num"].transform("max")
    df["season_progress"]  = df["home_game_num"] / df["home_games_total"]

    # Rivalry flag
    df["is_rivalry"] = df["opponent"].isin(rivalries).astype(int)

    # Rolling home win rate (last 5, lag-1)
    df["home_win_rate_last5"] = (
        df["home_win"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
        .fillna(0.5)
    )

    # Opponent win rate from games in our dataset (lag-1 expanding mean per season)
    opp_records = []
    for (season, opp), grp in df.groupby(["season", "opp_abbr"]):
        opp_win = 1 - grp["home_win"]
        rate = opp_win.shift(1).expanding().mean().fillna(0.5)
        opp_records.append(pd.Series(rate.values, index=grp.index))
    df["opp_win_rate_season"] = pd.concat(opp_records).sort_index()
    df["opp_win_rate_season"] = df["opp_win_rate_season"].fillna(0.5)

    # Venue capacity (may vary by year for some teams)
    df["venue_capacity"] = df["season"].apply(capacity_fn)

    return df


# ── Main pipeline ───────────────────────────────────────────────────────────────

def main_pipeline(cfg):
    """
    cfg keys: sport, league, slug, home_abbrev, train_years, team_name,
              lat, lon, rivalries, capacity_fn, output_file
    """
    os.makedirs(os.path.dirname(cfg["output_file"]), exist_ok=True)

    print(f"\n=== Step 1: Fetching {cfg['team_name']} game data ===")
    df = fetch_all_seasons(
        cfg["sport"], cfg["league"], cfg["slug"], cfg["home_abbrev"],
        cfg["train_years"],
    )
    print(f"  Total home games: {len(df)}")
    print(f"  Seasons: {sorted(df['season'].unique())}")

    print("\n=== Step 2: Fetching historical weather ===")
    df = fetch_weather(df, cfg["lat"], cfg["lon"])
    print(f"  Weather coverage: {df['temperature_f'].notna().sum()} / {len(df)}")

    print("\n=== Step 3: Engineering features ===")
    df = add_features(df, cfg["rivalries"], cfg["capacity_fn"])

    df["attendance_pct"] = df["attendance"] / df["venue_capacity"]
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cfg["output_file"], index=False)

    print(f"\n=== Done ===")
    print(f"Saved {len(df)} games to {cfg['output_file']}")
    print(df[["season","attendance","temperature_f","home_win_rate_last5","attendance_pct"]].describe().round(2))
