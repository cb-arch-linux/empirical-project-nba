"""
Collects all raw player statistics from the NBA API and saves them as CSV files
to data/raw/. No cleaning or transformation is performed here.

Two datasets are collected:
  - Historical (2015-16 to 2021-22): used to train the ridge regression model
    that learns MVP vote share weights from past seasons.
  - Current (2022-23): the season being scored and analysed.

Six stat types are pulled per season via the nba_api package:
  Base       - traditional per-game counting stats (pts, ast, reb, stl, blk, tov)
  Advanced   - box score metrics (TS%, USG%, net rating, PIE, etc.)
  Bio        - supplementary rates (AST%, OREB%, DREB%)
  Scoring    - breakdown of points by shot type
  Drives     - drives per game and drive efficiency
  Defense    - opponent FG% and defensive +/-
"""

import os
import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayerbiostats,
    leaguedashptstats,
)


# Configuration

CURRENT_SEASON = "2022-23"
SEASON_TYPE = "Regular Season"
RAW_DATA_DIR = os.path.join("data", "raw")

# Historical seasons used for regression training only.
HISTORICAL_SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
]

NBA_API_RETRY_LIMIT = 3
NBA_API_RETRY_DELAY = 5  # seconds between retries on failure
NBA_API_REQUEST_DELAY = 2  # seconds between successful requests to avoid rate limiting


# Helper Functions

def save_dataframe_to_csv(dataframe, filename):
    """Save a DataFrame to data/raw/<filename> without the row index."""
    output_path = os.path.join(RAW_DATA_DIR, filename)
    dataframe.to_csv(output_path, index=False)
    print(f"  Saved {len(dataframe)} rows to {output_path}")


# NBA API Fetch Functions

def _fetch_with_retry(api_call, label):
    """Calls api_call(), retrying up to NBA_API_RETRY_LIMIT times on failure."""
    for attempt in range(1, NBA_API_RETRY_LIMIT + 1):
        try:
            return api_call()
        except Exception as error:
            print(f"    Attempt {attempt} failed: {error}")
            if attempt < NBA_API_RETRY_LIMIT:
                time.sleep(NBA_API_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"NBA API {label} failed after {NBA_API_RETRY_LIMIT} attempts."
                ) from error  # "from error" preserves the original exception in the traceback for debugging.


def fetch_league_dash_stats(season, measure_type):
    """
    Fetch per-game player stats from the LeagueDashPlayerStats endpoint.

    measure_type controls which stat group is returned:
      "Base"     - traditional counting stats
      "Advanced" - advanced box score metrics
      "Scoring"  - breakdown of points by shot type
    """
    def call():
        response = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense=measure_type
        )
        df = response.get_data_frames()[0]  # The API returns a list of DataFrames; index 0 is always the primary results table.
        df["season"] = season  # Tag each row with the season so rows can be identified after multi-season concat.
        return df
    return _fetch_with_retry(call, f"{measure_type} {season}")


def fetch_bio_stats(season):
    """
    Fetch per-game bio stats from the LeagueDashPlayerBioStats endpoint.

    Returns supplementary rate stats (AST%, OREB%, DREB%) not available from
    the standard LeagueDashPlayerStats endpoint.
    """
    def call():
        response = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=season,
            season_type_all_star=SEASON_TYPE,
            per_mode_simple="PerGame"
        )
        df = response.get_data_frames()[0]  # The API returns a list of DataFrames; index 0 is always the primary results table.
        df["season"] = season  # Tag each row with the season so rows can be identified after multi-season concat.
        return df
    return _fetch_with_retry(call, f"bio {season}")


def fetch_pt_stats(season, pt_measure_type):
    """
    Fetch player tracking stats from the LeagueDashPtStats endpoint.

    Tracking stats come from the NBA's optical camera system and capture
    movement-based metrics not available in the box score.

    pt_measure_type controls which tracking group is returned:
      "Drives"  - drives per game, drive points, drive assists (offensive creation)
      "Defense" - opponent FG% at the rim, defensive +/- (on-ball defence)
    """
    def call():
        response = leaguedashptstats.LeagueDashPtStats(
            season=season,
            season_type_all_star=SEASON_TYPE,
            per_mode_simple="PerGame",
            player_or_team="Player",
            pt_measure_type=pt_measure_type
        )
        df = response.get_data_frames()[0]  # The API returns a list of DataFrames; index 0 is always the primary results table.
        df["season"] = season  # Tag each row with the season so rows can be identified after multi-season concat.
        return df
    return _fetch_with_retry(call, f"{pt_measure_type} tracking {season}")


# Multi-Season Collection

def collect_seasons(seasons, label):
    """
    Collect all six stat types across a list of seasons and return one combined
    DataFrame per stat type.

    Iterates over each season, fetches each stat group, and concatenates the
    results. A short delay is inserted between requests to stay within the NBA
    API's rate limit. A 'season' column is added to every row so individual
    player-seasons can be distinguished after combining.

    Returns a tuple of six DataFrames: (per_game, advanced, bio, scoring, drives, defense).
    """
    per_game_frames = []
    advanced_frames = []
    bio_frames = []
    scoring_frames = []
    drives_frames = []
    defense_frames = []

    total = len(seasons)
    for i, season in enumerate(seasons, 1):
        print(f"  [{i}/{total}] {season}")

        per_game_frames.append(fetch_league_dash_stats(season, "Base"))
        time.sleep(NBA_API_REQUEST_DELAY)

        advanced_frames.append(fetch_league_dash_stats(season, "Advanced"))
        time.sleep(NBA_API_REQUEST_DELAY)

        bio_frames.append(fetch_bio_stats(season))
        time.sleep(NBA_API_REQUEST_DELAY)

        scoring_frames.append(fetch_league_dash_stats(season, "Scoring"))
        time.sleep(NBA_API_REQUEST_DELAY)

        drives_frames.append(fetch_pt_stats(season, "Drives"))
        time.sleep(NBA_API_REQUEST_DELAY)

        defense_frames.append(fetch_pt_stats(season, "Defense"))
        time.sleep(NBA_API_REQUEST_DELAY)

    per_game_df = pd.concat(per_game_frames, ignore_index=True)  # ignore_index=True resets row indices after combining seasons to avoid duplicates.
    advanced_df = pd.concat(advanced_frames, ignore_index=True)  # ignore_index=True resets row indices after combining seasons to avoid duplicates.
    bio_df = pd.concat(bio_frames, ignore_index=True)  # ignore_index=True resets row indices after combining seasons to avoid duplicates.
    scoring_df = pd.concat(scoring_frames, ignore_index=True)  # ignore_index=True resets row indices after combining seasons to avoid duplicates.
    drives_df = pd.concat(drives_frames, ignore_index=True)  # ignore_index=True resets row indices after combining seasons to avoid duplicates.
    defense_df = pd.concat(defense_frames, ignore_index=True)  # ignore_index=True resets row indices after combining seasons to avoid duplicates.

    print(f"  {label} collection complete: {len(per_game_df)} total player-season rows.")
    return per_game_df, advanced_df, bio_df, scoring_df, drives_df, defense_df


# Main Entry Point

def main():
    """Collect historical and current season data and save all raw CSV files to data/raw/."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"\n[1/2] Historical data ({len(HISTORICAL_SEASONS)} seasons)")
    per_game, advanced, bio, scoring, drives, defense = collect_seasons(HISTORICAL_SEASONS, "Historical")
    save_dataframe_to_csv(per_game, "historical_per_game_raw.csv")
    save_dataframe_to_csv(advanced, "historical_advanced_raw.csv")
    save_dataframe_to_csv(bio, "historical_bio_raw.csv")
    save_dataframe_to_csv(scoring, "historical_scoring_raw.csv")
    save_dataframe_to_csv(drives, "historical_drives_raw.csv")
    save_dataframe_to_csv(defense, "historical_defense_raw.csv")

    print(f"\n[2/2] Current season ({CURRENT_SEASON})")
    per_game, advanced, bio, scoring, drives, defense = collect_seasons([CURRENT_SEASON], "Current")
    save_dataframe_to_csv(per_game, "current_per_game_raw.csv")
    save_dataframe_to_csv(advanced, "current_advanced_raw.csv")
    save_dataframe_to_csv(bio, "current_bio_raw.csv")
    save_dataframe_to_csv(scoring, "current_scoring_raw.csv")
    save_dataframe_to_csv(drives, "current_drives_raw.csv")
    save_dataframe_to_csv(defense, "current_defense_raw.csv")

    print("\nData collection complete.")


if __name__ == "__main__":
    main()
