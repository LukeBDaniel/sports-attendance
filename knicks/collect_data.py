"""
collect_data.py — Knicks Attendance Data
------------------------------------------
Fetches Knicks NBA home game data from ESPN API.
Run with: python collect_data.py

COVID note: 2020 (2019-20 bubble) and 2021 (2020-21 no/limited fans) excluded.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collect_espn import main_pipeline

# Madison Square Garden
VENUE_LAT = 40.7505
VENUE_LON = -73.9934

def capacity(year):
    # MSG renovated in stages; post-2013 renovation ~19,812
    return 19_812 if year >= 2014 else 19_763

CONFIG = {
    "sport":       "basketball",
    "league":      "nba",
    "slug":        "ny",
    "home_abbrev": "NY",
    "train_years": list(range(2001, 2020)) + list(range(2022, 2026)),
    "team_name":   "New York Knicks",
    "lat":         VENUE_LAT,
    "lon":         VENUE_LON,
    "rivalries":   {
        "Boston Celtics",
        "Miami Heat",
        "Chicago Bulls",
        "Brooklyn Nets",
        "Indiana Pacers",
    },
    "capacity_fn": capacity,
    "output_file": os.path.join(os.path.dirname(__file__), "data", "games.csv"),
}

if __name__ == "__main__":
    main_pipeline(CONFIG)
