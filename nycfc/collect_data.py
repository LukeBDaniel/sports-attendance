"""
collect_data.py — NYCFC Attendance Data
-----------------------------------------
Fetches NYCFC MLS home game data from ESPN API.
Run with: python collect_data.py

COVID note: 2020 and 2021 excluded (no fans / restricted capacity).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collect_espn import main_pipeline

# Yankee Stadium soccer configuration
VENUE_LAT = 40.8296
VENUE_LON = -73.9262
CAPACITY  = 28_743  # soccer configuration, relatively stable

CONFIG = {
    "sport":       "soccer",
    "league":      "usa.1",
    "slug":        "17606",       # NYCFC's ESPN team ID
    "home_abbrev": "NYC",
    "train_years": list(range(2015, 2020)) + list(range(2022, 2026)),
    "team_name":   "New York City FC",
    "lat":         VENUE_LAT,
    "lon":         VENUE_LON,
    "rivalries":   {
        "New York Red Bulls",
        "Philadelphia Union",
        "New England Revolution",
    },
    "capacity_fn": lambda year: CAPACITY,
    "output_file": os.path.join(os.path.dirname(__file__), "data", "games.csv"),
}

if __name__ == "__main__":
    main_pipeline(CONFIG)
