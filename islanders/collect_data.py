"""
collect_data.py — Islanders Attendance Data
---------------------------------------------
Fetches Islanders NHL home game data from ESPN API.
Run with: python collect_data.py

COVID note: 2020 and 2021 excluded.
Stadium history:
  2001-2015: Nassau Veterans Memorial Coliseum (16,234)
  2015-2020: Barclays Center, hockey config (15,813)
  2022+:     UBS Arena (17,255)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collect_espn import main_pipeline

# UBS Arena (Elmont, NY) — current home
VENUE_LAT = 40.7221
VENUE_LON = -73.7202

def capacity(year):
    if year <= 2015:
        return 16_234   # Nassau Coliseum
    elif year <= 2020:
        return 15_813   # Barclays Center (hockey config)
    else:
        return 17_255   # UBS Arena

# Use UBS Arena coords for weather (most recent)
CONFIG = {
    "sport":       "hockey",
    "league":      "nhl",
    "slug":        "nyi",
    "home_abbrev": "NYI",
    "train_years": list(range(2001, 2020)) + list(range(2022, 2026)),
    "team_name":   "New York Islanders",
    "lat":         VENUE_LAT,
    "lon":         VENUE_LON,
    "rivalries":   {
        "New York Rangers",
        "New Jersey Devils",
        "Pittsburgh Penguins",
        "Philadelphia Flyers",
    },
    "capacity_fn": capacity,
    "output_file": os.path.join(os.path.dirname(__file__), "data", "games.csv"),
}

if __name__ == "__main__":
    main_pipeline(CONFIG)
