"""
collect_data.py — Rangers Attendance Data
-------------------------------------------
Fetches Rangers NHL home game data from ESPN API.
Run with: python collect_data.py

COVID note: 2020 (2019-20 bubble) and 2021 (2020-21 no fans) excluded.
NHL season year = end year (2024 = 2023-24 season).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collect_espn import main_pipeline

# Madison Square Garden (hockey configuration — smaller than basketball)
VENUE_LAT = 40.7505
VENUE_LON = -73.9934
CAPACITY  = 18_006

CONFIG = {
    "sport":       "hockey",
    "league":      "nhl",
    "slug":        "nyr",
    "home_abbrev": "NYR",
    "train_years": list(range(2001, 2020)) + list(range(2022, 2026)),
    "team_name":   "New York Rangers",
    "lat":         VENUE_LAT,
    "lon":         VENUE_LON,
    "rivalries":   {
        "New Jersey Devils",
        "New York Islanders",
        "Pittsburgh Penguins",
        "Philadelphia Flyers",
        "Washington Capitals",
    },
    "capacity_fn": lambda year: CAPACITY,
    "output_file": os.path.join(os.path.dirname(__file__), "data", "games.csv"),
}

if __name__ == "__main__":
    main_pipeline(CONFIG)
