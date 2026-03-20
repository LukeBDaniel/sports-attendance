"""
add_competition_features.py
---------------------------
Adds NYC local competition features to the existing nwsl_matches.csv
without re-fetching all weather data.

Features added:
  nyc_mlb_home_game  — 1 if Yankees or Mets have a home game on the same date
  nyc_nfl_home_game  — 1 if Giants or Jets have a home game on the same date

Data sources:
  MLB: pybaseball.schedule_and_record()
  NFL: nfl_data_py.import_schedules()
"""

import os
import warnings
import pandas as pd
import pybaseball as mlb
import nfl_data_py as nfl

warnings.filterwarnings("ignore")

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "nwsl_matches.csv")

MLB_TEAMS = ["NYY", "NYM"]
NFL_TEAMS = {"NYG", "NYJ"}


# ── MLB home game dates ────────────────────────────────────────────────────────

def get_mlb_home_dates(years: list[int]) -> set[str]:
    """Return set of date strings (YYYY-MM-DD) when Yankees or Mets play at home."""
    home_dates = set()
    for team in MLB_TEAMS:
        for year in years:
            try:
                sched = mlb.schedule_and_record(year, team)
                home = sched[sched["Home_Away"] == "Home"].copy()
                # Date format: "Thursday, Mar 30" — append year and parse
                home["date_parsed"] = pd.to_datetime(
                    home["Date"].astype(str) + f" {year}",
                    format="%A, %b %d %Y",
                    errors="coerce",
                )
                dates = home["date_parsed"].dropna().dt.strftime("%Y-%m-%d").tolist()
                home_dates.update(dates)
                print(f"  {team} {year}: {len(dates)} home games")
            except Exception as e:
                print(f"  WARNING: {team} {year} failed: {e}")
    return home_dates


# ── NFL home game dates ────────────────────────────────────────────────────────

def get_nfl_home_dates(years: list[int]) -> set[str]:
    """Return set of date strings (YYYY-MM-DD) when Giants or Jets play at home."""
    home_dates = set()
    try:
        # nfl_data_py uses season years; each season starts in Sep of that year
        sched = nfl.import_schedules(years)
        home = sched[sched["home_team"].isin(NFL_TEAMS)].copy()
        home["date_parsed"] = pd.to_datetime(home["gameday"], errors="coerce")
        dates = home["date_parsed"].dropna().dt.strftime("%Y-%m-%d").tolist()
        home_dates.update(dates)
        print(f"  Giants/Jets {min(years)}–{max(years)}: {len(dates)} home games")
    except Exception as e:
        print(f"  WARNING: NFL schedule fetch failed: {e}")
    return home_dates


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DATA_FILE)
    years = sorted(df["season"].unique().tolist())
    print(f"Loaded {len(df)} games across seasons: {years}")

    # ── Fetch competition schedules ──
    print("\nFetching MLB home game dates...")
    mlb_dates = get_mlb_home_dates(years)
    print(f"Total Yankees/Mets home dates: {len(mlb_dates)}")

    print("\nFetching NFL home game dates...")
    # NFL season years in the data (season starts Sep of that year)
    nfl_dates = get_nfl_home_dates(years)
    print(f"Total Giants/Jets home dates: {len(nfl_dates)}")

    # ── Add features ──
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["nyc_mlb_home_game"] = df["date_str"].isin(mlb_dates).astype(int)
    df["nyc_nfl_home_game"] = df["date_str"].isin(nfl_dates).astype(int)
    df = df.drop(columns=["date_str"])

    # ── Spot-check ──
    print("\nSpot-check — games with NFL competition:")
    cols = ["date", "season", "home_team", "away_team", "nyc_nfl_home_game"]
    sample = df[df["nyc_nfl_home_game"] == 1][cols].head(5)
    print(sample.to_string(index=False))

    print("\nSpot-check — games with MLB competition:")
    cols = ["date", "season", "home_team", "away_team", "nyc_mlb_home_game"]
    sample = df[df["nyc_mlb_home_game"] == 1][cols].head(5)
    print(sample.to_string(index=False))

    print(f"\nGames with MLB competition:  {df['nyc_mlb_home_game'].sum()} / {len(df)}")
    print(f"Games with NFL competition:  {df['nyc_nfl_home_game'].sum()} / {len(df)}")

    df.to_csv(DATA_FILE, index=False)
    print(f"\nSaved updated CSV to {DATA_FILE}")


if __name__ == "__main__":
    main()
