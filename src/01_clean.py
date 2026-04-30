"""
Cleans and merges raw NBA API CSVs into two datasets.

Outputs historical_clean.csv (2015-16 to 2021-22), used to train the ridge
regression model highlighting MVP calibre players, and current_clean.csv (2022-23),
used for scoring and ranking.
"""

import os

import numpy as np
import pandas as pd


# Configuration

RAW_DATA_DIR = os.path.join("data", "raw")
CLEAN_DATA_DIR = os.path.join("data", "clean")

HISTORICAL_CLEAN_OUT = os.path.join(CLEAN_DATA_DIR, "historical_clean.csv")
CURRENT_CLEAN_OUT = os.path.join(CLEAN_DATA_DIR, "current_clean.csv")

MIN_GAMES_PLAYED = 41

# Columns to retain from the per game endpoint with their output names.
PER_GAME_COLUMNS = {
    "player_id": "player_id",
    "player_name": "player_name",
    "team_abbreviation": "team",
    "gp": "games_played",
    "w_pct": "team_win_pct",
    "min": "minutes_per_game",
    "pts": "points_per_game",
    "reb": "rebounds_per_game",
    "ast": "assists_per_game",
    "stl": "steals_per_game",
    "blk": "blocks_per_game",
    "tov": "turnovers_per_game",
    "fga": "fga_per_game",
    "fg_pct": "fg_pct",
    "fg3_pct": "fg3_pct",
    "ft_pct": "ft_pct",
    "plus_minus": "plus_minus",
    "season": "season",
}

# Columns to retain from the advanced endpoint with their output names.
ADVANCED_COLUMNS = {
    "player_id": "player_id",
    "season": "season",
    "e_off_rating": "off_rating",
    "e_def_rating": "def_rating",
    "e_net_rating": "net_rating",
    "e_pace": "pace",
    "usg_pct": "usage_pct",
    "ts_pct": "true_shooting_pct",
    "pie": "pie",
}

# Columns to retain from the bio stats endpoint with their output names.
BIO_COLUMNS = {
    "player_id": "player_id",
    "season": "season",
    "ast_pct": "ast_pct",
    "oreb_pct": "oreb_pct",
    "dreb_pct": "dreb_pct",
}

# Columns to retain from the scoring endpoint with their output names.
SCORING_COLUMNS = {
    "player_id": "player_id",
    "season": "season",
    "pct_pts_paint": "pct_pts_paint",
    "pct_pts_3pt": "pct_pts_3pt",
    "pct_pts_ft": "pct_pts_ft",
}

# Columns to retain from the drives tracking endpoint.
# DRIVES = number of drive attempts per game.
# DRIVE_FG_PCT = field goal percentage on drive attempts.
# DRIVE_PTS = points scored per game on drives.
DRIVES_COLUMNS = {
    "player_id": "player_id",
    "season": "season",
    "drives": "drives_per_game",
    "drive_fg_pct": "drive_fg_pct",
    "drive_pts": "drive_pts_per_game",
    "drive_passes": "drive_passes_per_game",
}

# Columns to retain from the defense tracking endpoint.
# DEF_RIM_FG_PCT = opponent field goal percentage on rim attempts when the
#                  player is the closest defender.
# DEF_RIM_FGA = number of opponent rim attempts defended per game. 
DEFENSE_COLUMNS = {
    "player_id": "player_id",
    "season": "season",
    "def_rim_fg_pct": "def_rim_fg_pct",
    "def_rim_fga": "def_rim_fga",
}

RATE_COLUMNS = [
    "team_win_pct", "minutes_per_game", "fg_pct", "fg3_pct", "ft_pct",
    "off_rating", "def_rating", "net_rating", "pace", "usage_pct",
    "true_shooting_pct", "pie", "ast_pct", "oreb_pct", "dreb_pct",
    "pct_pts_paint", "pct_pts_3pt", "pct_pts_ft", "plus_minus",
    "drives_per_game", "drive_fg_pct", "drive_pts_per_game",
    "drive_passes_per_game", "def_rim_fg_pct", "def_rim_fga",
]

COUNTING_COLUMNS = [
    "points_per_game", "rebounds_per_game", "assists_per_game",
    "steals_per_game", "blocks_per_game", "turnovers_per_game",
    "fga_per_game",
]

# Historical MVP winners and top 3 finishers by season. Each entry maps a
# season string to the set of player names who finished in the top 3 of MVP
# voting that year. Names are lowercase to match the normalised player_name
# column after cleaning.
HISTORICAL_MVP_LABELS = {
    "2015-16": {"stephen curry", "kawhi leonard", "lebron james"},
    "2016-17": {"russell westbrook", "james harden", "kawhi leonard"},
    "2017-18": {"james harden", "lebron james", "anthony davis"},
    "2018-19": {"giannis antetokounmpo", "james harden", "paul george"},
    "2019-20": {"giannis antetokounmpo", "lebron james", "luka doncic"},
    "2020-21": {"nikola jokic", "joel embiid", "stephen curry"},
    "2021-22": {"nikola jokic", "giannis antetokounmpo", "devin booker"},
}


# Helper Functions

def standardise_column_names(dataframe):
    """Convert all column names to lowercase snake_case."""
    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")  
        # Removes underscores that appear when a column name starts or ends with a special character.
    )
    return dataframe


def select_and_rename(dataframe, column_map):
    """
    Retain only the columns in column_map and rename them.

    Columns present in column_map but missing from the DataFrame are silently
    skipped.
    """
    available = {k: v for k, v in column_map.items() if k in dataframe.columns}
    return dataframe[list(available.keys())].rename(columns=available)


def remove_traded_player_duplicates(dataframe):
    """
    Remove per-team rows for players traded mid-season, keeping only totals.

    The NBA API returns one row per team for traded players plus a "TOT" row
    representing full-season totals; this function drops the per-team rows for
    any player who has a TOT entry.
    """
    if "team" not in dataframe.columns:
        return dataframe

    # transform broadcasts the TOT check to every row in the group, not just one per group.
    has_tot = dataframe.groupby(["player_id", "season"])["team"].transform(
        lambda x: "TOT" in x.values
    )
    # Keep rows that are either not in a traded group, or are the TOT season-total row.
    mask = ~has_tot | (dataframe["team"] == "TOT") 
    result = dataframe[mask].copy()
    return result


def filter_minimum_games(dataframe):
    """Remove players who appeared in fewer than MIN_GAMES_PLAYED games."""
    before = len(dataframe)
    df = dataframe[dataframe["games_played"] >= MIN_GAMES_PLAYED].copy()
    print(f"  Games filter: {before} -> {len(df)} rows (>= {MIN_GAMES_PLAYED} games).")
    return df


def impute_missing_values(dataframe):
    """
    Fill missing values using per-column strategies appropriate for each type.

    Rate columns are filled with the per-season median. Counting columns
    receive zeros.
    """
    if "season" in dataframe.columns:
        for col in RATE_COLUMNS:
            if col in dataframe.columns and dataframe[col].isna().any():
                dataframe[col] = dataframe.groupby("season")[col].transform(
                    lambda x: x.fillna(x.median())
                )
    else:
        for col in RATE_COLUMNS:
            if col in dataframe.columns and dataframe[col].isna().any():
                dataframe[col] = dataframe[col].fillna(dataframe[col].median())

    for col in COUNTING_COLUMNS:
        if col in dataframe.columns and dataframe[col].isna().any():
            dataframe[col] = dataframe[col].fillna(0)

    return dataframe


def add_mvp_labels(dataframe):
    """Add an is_mvp_calibre column to the historical dataset."""
    def label_row(row):
        # Returns 1 if the player is in the MVP calibre set for their season, 0 otherwise.
        season_mvps = HISTORICAL_MVP_LABELS.get(row["season"], set())
        return int(row["player_name"].lower().strip() in season_mvps)

    dataframe = dataframe.copy()
    dataframe["is_mvp_calibre"] = dataframe.apply(label_row, axis=1)
    total_labels = dataframe["is_mvp_calibre"].sum()
    print(f"  Labelled {total_labels} MVP calibre player-season rows across "
          f"{len(HISTORICAL_MVP_LABELS)} seasons.")
    return dataframe


def _load_tracking(path, column_map):
    """Load a tracking stats CSV, standardise column names, and NaN missing values."""
    df = pd.read_csv(path)
    df = standardise_column_names(df)
    df = select_and_rename(df, column_map)
    numeric_cols = [c for c in df.columns if c not in ["player_id", "season"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # errors="coerce" converts unparseable strings to NaN rather than raising.
    return df


# Merge Pipeline

def merge_all_sources(per_game_df, advanced_df, bio_df, scoring_df,
                      drives_df, defense_df):
    """Merge all six cleaned DataFrames into a single player-level dataset.

    All merges use player_id and season as the composite join key. Left joins
    are used so that all players in the per-game source are retained even if
    they are missing from other endpoints.
    """
    join_keys = ["player_id", "season"]

    df = per_game_df.merge(advanced_df, on=join_keys, how="left")
    # Empty string keeps the original column name on the left side; avoids duplicate column names from the join.
    df = df.merge(bio_df, on=join_keys, how="left", suffixes=("", "_bio"))  
    df = df.merge(scoring_df, on=join_keys, how="left", suffixes=("", "_sc"))
    df = df.merge(drives_df, on=join_keys, how="left", suffixes=("", "_drv"))
    df = df.merge(defense_df, on=join_keys, how="left", suffixes=("", "_def"))

    # Suffix columns (_bio, _sc, etc.) are duplicate join keys added by the merges, drop them.
    dup_cols = [c for c in df.columns if
                c.endswith("_bio") or c.endswith("_sc") or
                c.endswith("_drv") or c.endswith("_def")]
    df = df.drop(columns=dup_cols, errors="ignore")  # errors="ignore" silently skips any suffix columns that don't exist.

    print(f"  Merged: {len(df)} rows, {len(df.columns)} columns.")
    return df


def clean_dataset(per_game_path, advanced_path, bio_path, scoring_path,
                  drives_path, defense_path, output_path, add_labels=False):
    """
    Run the full cleaning pipeline for one dataset and save the result.

    Loads and merges all six raw sources, removes traded-player duplicates,
    filters by minimum games, fills missing values, and adds MVP
    player labels for the historical training dataset.
    """
    df = pd.read_csv(per_game_path)
    df = standardise_column_names(df)
    df = select_and_rename(df, PER_GAME_COLUMNS)

    advanced_df = pd.read_csv(advanced_path)
    advanced_df = standardise_column_names(advanced_df)
    advanced_df = select_and_rename(advanced_df, ADVANCED_COLUMNS)

    bio_df = pd.read_csv(bio_path)
    bio_df = standardise_column_names(bio_df)
    bio_df = select_and_rename(bio_df, BIO_COLUMNS)

    scoring_df = pd.read_csv(scoring_path)
    scoring_df = standardise_column_names(scoring_df)
    scoring_df = select_and_rename(scoring_df, SCORING_COLUMNS)

    drives_df = _load_tracking(drives_path, DRIVES_COLUMNS)
    defense_df = _load_tracking(defense_path, DEFENSE_COLUMNS)

    df = merge_all_sources(df, advanced_df, bio_df, scoring_df,
                           drives_df, defense_df)
    df = remove_traded_player_duplicates(df)
    df = filter_minimum_games(df)
    df = impute_missing_values(df)

    if add_labels:
        df = add_mvp_labels(df)

    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows to {output_path}")


# Main Entry Point

def main():
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

    print("\n[1/2] Historical dataset (2015-16 to 2021-22)")
    clean_dataset(
        per_game_path=os.path.join(RAW_DATA_DIR, "historical_per_game_raw.csv"),
        advanced_path=os.path.join(RAW_DATA_DIR, "historical_advanced_raw.csv"),
        bio_path=os.path.join(RAW_DATA_DIR, "historical_bio_raw.csv"),
        scoring_path=os.path.join(RAW_DATA_DIR, "historical_scoring_raw.csv"),
        drives_path=os.path.join(RAW_DATA_DIR, "historical_drives_raw.csv"),
        defense_path=os.path.join(RAW_DATA_DIR, "historical_defense_raw.csv"),
        output_path=HISTORICAL_CLEAN_OUT,
        add_labels=True
    )

    print("\n[2/2] Current season dataset (2022-23)")
    clean_dataset(
        per_game_path=os.path.join(RAW_DATA_DIR, "current_per_game_raw.csv"),
        advanced_path=os.path.join(RAW_DATA_DIR, "current_advanced_raw.csv"),
        bio_path=os.path.join(RAW_DATA_DIR, "current_bio_raw.csv"),
        scoring_path=os.path.join(RAW_DATA_DIR, "current_scoring_raw.csv"),
        drives_path=os.path.join(RAW_DATA_DIR, "current_drives_raw.csv"),
        defense_path=os.path.join(RAW_DATA_DIR, "current_defense_raw.csv"),
        output_path=CURRENT_CLEAN_OUT,
        add_labels=False
    )

    print("\nCleaning complete.")


if __name__ == "__main__":
    main()
