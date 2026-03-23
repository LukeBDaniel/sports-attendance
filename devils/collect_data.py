"""
collect_data.py — Devils Attendance Data
------------------------------------------
Fetches Devils NHL home game data from ESPN API.
Run with: python collect_data.py

COVID note: 2020 and 2021 excluded.
Stadium history:
  2001-2007: Continental Airlines Arena / Meadowlands (~19,040)
  2007+:     Prudential Center, Newark NJ (16,514)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collect_espn import main_pipeline

# Prudential Center, Newark NJ — current home
VENUE_LAT = 40.7334
VENUE_LON = -74.1713

def capacity(year):
    return 19_040 if year <= 2007 else 16_514

CONFIG = {
    "sport":       "hockey",
    "league":      "nhl",
    "slug":        "njd",
    "home_abbrev": "NJ",          # ESPN uses "NJ" not "NJD"
    "train_years": list(range(2001, 2020)) + list(range(2022, 2026)),
    "team_name":   "New Jersey Devils",
    "lat":         VENUE_LAT,
    "lon":         VENUE_LON,
    "rivalries":   {
        "New York Rangers",
        "New York Islanders",
        "Philadelphia Flyers",
        "Pittsburgh Penguins",
        "New York Rangers",
    },
    "capacity_fn": capacity,
    "output_file": os.path.join(os.path.dirname(__file__), "data", "games.csv"),
}

if __name__ == "__main__":
    main_pipeline(CONFIG)
