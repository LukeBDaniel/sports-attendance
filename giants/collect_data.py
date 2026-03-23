"""
collect_data.py — Giants Attendance Data
------------------------------------------
Fetches Giants NFL home game data from ESPN API.
Run with: python collect_data.py

COVID note: 2020 excluded (MetLife Stadium fully closed to fans).
Stadium history:
  2000-2009: Giants Stadium, East Rutherford NJ (80,242)
  2010+:     MetLife Stadium, East Rutherford NJ (82,500)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collect_espn import main_pipeline

# MetLife Stadium, East Rutherford NJ
VENUE_LAT = 40.8135
VENUE_LON = -74.0745

def capacity(year):
    return 80_242 if year < 2010 else 82_500

CONFIG = {
    "sport":       "football",
    "league":      "nfl",
    "slug":        "nyg",
    "home_abbrev": "NYG",
    "train_years": list(range(2000, 2020)) + list(range(2021, 2026)),
    "team_name":   "New York Giants",
    "lat":         VENUE_LAT,
    "lon":         VENUE_LON,
    "rivalries":   {
        "Philadelphia Eagles",
        "Dallas Cowboys",
        "Washington Commanders",
        "New York Jets",
    },
    "capacity_fn": capacity,
    "output_file": os.path.join(os.path.dirname(__file__), "data", "games.csv"),
}

if __name__ == "__main__":
    main_pipeline(CONFIG)
