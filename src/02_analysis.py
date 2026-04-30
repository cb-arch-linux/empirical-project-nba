"""
Loads cleaned data from 01_clean.py and runs three analytical stages:

  Stage 1: Build custom BPM, VORP, and WS/48 approximations from NBA API stats.
  Stage 2: Fit ridge regression on historical seasons to derive composite score
           weights; fit supplementary OLS for p-values and confidence intervals.
  Stage 3: Double ML Causal Forest estimating the causal effect of usage rate
           on true shooting percentage per player.
"""

import os
import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=UserWarning)


# Configuration

CLEAN_DATA_DIR = os.path.join("data", "clean")
TABLES_OUTPUT_DIR = os.path.join("output", "tables")

CURRENT_CLEAN = os.path.join(CLEAN_DATA_DIR, "current_clean.csv")
HISTORICAL_CLEAN = os.path.join(CLEAN_DATA_DIR, "historical_clean.csv")
MVP_SCORES_OUT = os.path.join(TABLES_OUTPUT_DIR, "mvp_scores.csv")
MODEL_COEFFICIENTS_OUT = os.path.join(TABLES_OUTPUT_DIR, "model_coefficients.csv")
OLS_RESULTS_OUT = os.path.join(TABLES_OUTPUT_DIR, "ols_results.csv")
CAUSAL_EFFECTS_OUT = os.path.join(TABLES_OUTPUT_DIR, "causal_effects.csv")

TOP_N_PLAYERS = 15
REPLACEMENT_LEVEL = -2.0
RIDGE_ALPHA = 1.0
RANDOM_STATE = 42

# MVP calibre labels are now read directly from the is_mvp_calibre column
# in historical_clean.csv, which was populated by 01_clean.py using the
# HISTORICAL_MVP_LABELS lookup covering 2015-16 through 2021-22.
# No hardcoded list is needed here.

# Composite score weights are derived from ridge regression coefficients
# in derive_regression_weights() and applied in calculate_composite_score().

REGRESSION_FEATURES = [
    "true_shooting_pct",
    "usage_pct",
    "ast_pct",
    "oreb_pct",
    "dreb_pct",
    "pie",
    "team_win_pct",
    "custom_bpm",
    "custom_ws48",
]
# The following variables were tested in the regression but removed because
# they were statistically insignificant (p > 0.05) and added no predictive
# value beyond the variables already included:
# drives_per_game 
# drive_fg_pct 
# def_rim_fg_pct
#
# net_rating is excluded because custom_ws48 is algebraically derived from
# off_rating minus def_rating, which equals net_rating. Including both creates
# multicollinearity.
#
# custom_vorp is excluded to avoid triple-counting — it is custom_bpm scaled
# by minutes and carries no independent information.

CAUSAL_FOREST_CONTROLS = [
    "true_shooting_pct",
    "ast_pct",
    "oreb_pct",
    "dreb_pct",
    "net_rating",
    "drives_per_game",
    "def_rim_fg_pct",
    "custom_bpm",
]

# Hard qualifying thresholds applied before composite scoring. Players below
# these thresholds are excluded from the MVP ranking entirely,
# These reflect how MVP voting works in practice.
#
# Team win percentage threshold: 50 wins out of 82 games = 0.610. Since
# 1975-76, only three players have won MVP on a team below this mark.
# I use 0.50 as the threshold rather than 0.61 to avoid being overly strict.
#
# Net rating threshold: a player on a team with a negative net rating is
# never in serious MVP contention.
MIN_TEAM_WIN_PCT = 0.50
MIN_NET_RATING = -2.0

# Usage rate threshold expressed as a decimal (0.26 = 26%). All MVP winners
# from 2015-16 through 2022-23 had usage rates above this level. 
# The OLS results confirm usage_pct is a statistically
# significant positive predictor of MVP calibre seasons.
MIN_USAGE_PCT = 0.26


# Helper Functions

def _to_pct(series):
    """Convert a Series to percentage scale if the NBA API stored it as a decimal (max <= 1)."""
    return series * 100 if series.max() <= 1 else series


def normalise_to_zero_one(series):
    """Scale a numeric Series to [0, 1] using min-max normalisation.

    Returns a Series of zeros if all values are identical to avoid division
    by zero.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


# Stage 1: Custom Impact Metrics

def calculate_custom_bpm(df):
    """
    Custom Box Plus Minus approximation adapted from Myers (2014).

    Uses rate stats rather than counting stats to control for pace and playing time.
    """
    stl_pct = df["steals_per_game"] / df["minutes_per_game"] * 100
    blk_pct = df["blocks_per_game"] / df["minutes_per_game"] * 100
    ast_pct = _to_pct(df["ast_pct"])
    oreb_pct = _to_pct(df["oreb_pct"])
    dreb_pct = _to_pct(df["dreb_pct"])
    usg_pct = _to_pct(df["usage_pct"])
    ts_pct = _to_pct(df["true_shooting_pct"])
    # Efficiency is measured relative to league average — not absolute shooting percentage.
    league_avg_ts = ts_pct.mean()

    raw_bpm = (
        (stl_pct * 0.9)
        + (blk_pct * 0.75)
        + (dreb_pct * 0.35)
        + (oreb_pct * 0.6)
        + (ast_pct * 0.6)
        + (usg_pct * (2.0 * (ts_pct - league_avg_ts) / 100))
        - (usg_pct * df["turnovers_per_game"] / (df["points_per_game"] + 0.001) * 0.5)  # + 0.001 avoids division by zero for players with zero points.
    )

    # Subtract team average net rating so players on good teams aren't rewarded just for their teammates.
    team_avg_net = df.groupby("team")["net_rating"].transform("mean")
    individual_net_above_team = df["net_rating"] - team_avg_net
    return raw_bpm + individual_net_above_team * 0.3


def calculate_custom_vorp(df):
    """
    Custom Value Over Replacement Player approximation.

    Scales BPM by minutes played; 4800 is Basketball Reference's normalising constant (48 min x 100 games).
    """
    minutes_played = df["minutes_per_game"] * df["games_played"]
    return (df["custom_bpm"] - REPLACEMENT_LEVEL) * (minutes_played / 4800)


def calculate_custom_ws48(df):
    """
    Custom Win Shares Per 48 Minutes approximation, adapted from Oliver (2004).

    Combines off/def rating differentials relative to league average, pace-adjusted.
    """
    league_avg_off = df["off_rating"].mean()
    league_avg_def = df["def_rating"].mean()
    # pace_factor is a ratio to league average, e.g. 1.05 = 5% faster than average.
    pace_factor = df["pace"] / df["pace"].mean()
    off_contribution = (df["off_rating"] - league_avg_off) * pace_factor
    def_contribution = (league_avg_def - df["def_rating"]) * pace_factor
    return (off_contribution + def_contribution) / 100


def add_custom_metrics(df):
    print("Calculating custom impact metrics...")
    df = df.copy()
    df["custom_bpm"] = calculate_custom_bpm(df)
    df["custom_vorp"] = calculate_custom_vorp(df)
    df["custom_ws48"] = calculate_custom_ws48(df)
    for metric in ["custom_bpm", "custom_vorp", "custom_ws48"]:
        print(f"  {metric}: {df[metric].min():.3f} to {df[metric].max():.3f}")
    return df


# Stage 2: Ridge Regression and Composite Scores

def fit_ols_regression(historical_df):
    """
    Supplementary OLS Linear Probability Model on historical data.

    Complements ridge regression by providing p-values and confidence intervals for each metric.
    """
    print("Fitting supplementary OLS regression...")

    available_features = [f for f in REGRESSION_FEATURES if f in historical_df.columns]

    feature_matrix = historical_df[available_features].copy()
    target = historical_df["is_mvp_calibre"]

    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    # add_constant appends an intercept column, which statsmodels OLS requires.
    feature_matrix_scaled = sm.add_constant(feature_matrix_scaled)

    col_names = ["const"] + available_features
    feature_df = pd.DataFrame(feature_matrix_scaled, columns=col_names)

    model = sm.OLS(target, feature_df).fit()

    # params[1:] and related slices skip index 0, which is the intercept term.
    results_df = pd.DataFrame({
        "metric": available_features,
        "coefficient": model.params[1:].round(4),
        "std_error": model.bse[1:].round(4),
        "t_statistic": model.tvalues[1:].round(3),
        "p_value": model.pvalues[1:].round(4),
    })

    results_df["significant_5pct"] = results_df["p_value"] < 0.05
    results_df = results_df.sort_values("t_statistic", key=abs, ascending=False)

    sig_count = results_df["significant_5pct"].sum()
    print(f"  OLS complete. {sig_count} features significant at 5% level.")
    print(f"  R-squared: {model.rsquared:.4f}")

    return results_df


def fit_ridge_regression(historical_df):
    """
    Fit ridge regression on historical seasons to derive composite score weights.

    Trained on 2015-16 to 2021-22 only so the model never sees 2022-23 data.
    """
    print("Fitting ridge regression on historical data (2015-16 to 2021-22)...")

    available_features = [f for f in REGRESSION_FEATURES if f in historical_df.columns]

    mvp_count = historical_df["is_mvp_calibre"].sum()
    total = len(historical_df)
    print(f"  Training set: {total} player-season rows, {mvp_count} labelled MVP calibre.")

    feature_matrix = historical_df[available_features].copy()
    target = historical_df["is_mvp_calibre"]

    # Standardising puts all features on the same scale so coefficients are comparable.
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(feature_matrix_scaled, target)

    coeff_df = pd.DataFrame({
        "metric": available_features,
        "coefficient": model.coef_.round(4)
    })
    coeff_df = coeff_df.sort_values("coefficient", key=abs, ascending=False)

    print(f"  Top predictive metric: {coeff_df.iloc[0]['metric']}")
    return coeff_df


def derive_regression_weights(coeff_df):
    """Normalise positive ridge coefficients into composite score weights."""
    positive_df = coeff_df[coeff_df["coefficient"] > 0].copy()

    if positive_df.empty:
        print("  Warning: no positive coefficients found. Returning equal weights.")
        equal_weight = 1.0 / len(coeff_df)
        return {metric: equal_weight for metric in coeff_df["metric"].tolist()}

    total = positive_df["coefficient"].sum()
    regression_weights = dict(
        zip(positive_df["metric"], (positive_df["coefficient"] / total).round(4))
    )

    print("  Data-driven weights derived from positive regression coefficients:")
    for metric, weight in sorted(regression_weights.items(), key=lambda x: -x[1]):
        print(f"    {metric}: {weight:.4f}")

    return regression_weights


def apply_qualifying_thresholds(df):
    """Filter out players below the hard MVP qualifying thresholds before scoring.

    Thresholds (MIN_TEAM_WIN_PCT, MIN_NET_RATING, MIN_USAGE_PCT) and their
    justifications are documented in the configuration constants above.
    """
    before = len(df)

    win_pct_mask = df["team_win_pct"] >= MIN_TEAM_WIN_PCT
    net_rating_mask = df["net_rating"] >= MIN_NET_RATING
    usage_mask = df["usage_pct"] >= MIN_USAGE_PCT
    combined_mask = win_pct_mask & net_rating_mask & usage_mask

    excluded = df[~combined_mask][["player_name", "team", "team_win_pct", "net_rating"]]
    if len(excluded) > 0:
        print(f"  Excluded {len(excluded)} players below qualifying thresholds:")
        notable = excluded.nlargest(5, "net_rating")
        for _, row in notable.iterrows():
            print(f"    {row['player_name']} ({row['team']}) "
                  f"win%={row['team_win_pct']:.3f} net_rtg={row['net_rating']:.1f}")

    filtered_df = df[combined_mask].copy()
    print(f"  Qualifying players: {before} -> {len(filtered_df)} "
          f"(removed {before - len(filtered_df)}).")
    return filtered_df

def calculate_composite_score(df, weights, score_column_name):
    """
    Calculate a weighted composite score from normalised metrics.

    Metrics not present in the DataFrame are skipped and the total weight
    is rescaled.
    """
    score = pd.Series(np.zeros(len(df)), index=df.index)
    total_weight = 0.0

    for metric, weight in weights.items():
        if metric not in df.columns:
            continue
        score += normalise_to_zero_one(df[metric]) * weight
        total_weight += weight

    # If any metrics were missing, rescale so the score still spans the full range.
    if 0 < total_weight < 1.0:
        score = score / total_weight

    df = df.copy()
    df[score_column_name] = score
    return df


def build_mvp_scores_table(df):
    """Return the top N players ranked by composite MVP score."""
    display_columns = [
        "player_name", "team", "games_played",
        "true_shooting_pct", "usage_pct", "ast_pct",
        "oreb_pct", "dreb_pct", "pie", "net_rating",
        "team_win_pct", "custom_bpm", "custom_vorp", "custom_ws48",
        "mvp_score_datadriven",
    ]
    available = [c for c in display_columns if c in df.columns]

    top_df = df.sort_values("mvp_score_datadriven", ascending=False).head(TOP_N_PLAYERS)[available].copy()
    top_df.insert(0, "rank", range(1, len(top_df) + 1))

    numeric_cols = top_df.select_dtypes(include=[np.number]).columns
    top_df[numeric_cols] = top_df[numeric_cols].round(3)
    return top_df


# Stage 3: Causal Forest

def run_causal_forest(df):
    """
    Double ML Causal Forest estimating the causal effect of usage_pct on TS%.

    TS% is used rather than PIE as the outcome because PIE is mechanically linked to usage.
    """
    print("Running causal forest (this may take a minute)...")

    available_controls = [c for c in CAUSAL_FOREST_CONTROLS if c in df.columns]
    missing_controls = [c for c in CAUSAL_FOREST_CONTROLS if c not in df.columns]
    if missing_controls:
        print(f"  Note: {missing_controls} not found in data. Running without these controls.")

    required_cols = ["usage_pct", "true_shooting_pct"] + available_controls
    missing_essential = [c for c in ["usage_pct", "true_shooting_pct"] if c not in df.columns]
    if missing_essential:
        raise ValueError(f"Causal forest requires essential columns not found: {missing_essential}")

    df_cf = df.dropna(subset=required_cols).copy()

    outcome = df_cf["true_shooting_pct"].values
    treatment = df_cf["usage_pct"].values
    controls = df_cf[available_controls].values

    # Two random forests: one residualises the outcome, one residualises the treatment.
    # The causal forest then fits on those residuals, removing confounding.
    model_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE)
    model_t = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE)

    causal_forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=500,
        min_samples_leaf=5,
        max_depth=5,
        random_state=RANDOM_STATE,
        verbose=0
    )

    causal_forest.fit(outcome, treatment, X=controls)

    # effect() returns the estimated causal effect per player; effect_interval() gives the 95% CI.
    effects = causal_forest.effect(controls)
    lower_bounds, upper_bounds = causal_forest.effect_interval(controls, alpha=0.05)

    results_df = pd.DataFrame({
        "player_name": df_cf["player_name"].values,
        "team": df_cf["team"].values,
        "usage_pct": treatment.round(3),
        "true_shooting_pct": outcome.round(3),
        "causal_effect": effects.round(4),
        "ci_lower": lower_bounds.round(4),
        "ci_upper": upper_bounds.round(4),
    })

    results_df = results_df.sort_values("causal_effect", ascending=False).reset_index(drop=True)
    results_df.insert(0, "rank", results_df.index + 1)

    print(f"  Average causal effect of usage on TS%: {effects.mean():.4f}")
    print(f"  Players with positive effect: {(effects > 0).sum()}")
    print(f"  Players with negative effect: {(effects < 0).sum()}")

    return results_df


# Main Entry Point

def main():
    os.makedirs(TABLES_OUTPUT_DIR, exist_ok=True)

    print("\nLoading cleaned data...")
    current_df = pd.read_csv(CURRENT_CLEAN)
    historical_df = pd.read_csv(HISTORICAL_CLEAN)
    print(f"  Loaded {len(current_df)} current season players (2022-23).")
    print(f"  Loaded {len(historical_df)} historical player-season rows (2015-16 to 2021-22).")

    print("\n[Stage 1] Custom Impact Metrics")
    current_df = add_custom_metrics(current_df)
    historical_df = add_custom_metrics(historical_df)

    # Stage 2: train on historical data only, then apply the weights to 2022-23.
    print("\n[Stage 2] Ridge Regression, Qualifying Thresholds, and Composite Scores")
    coeff_df = fit_ridge_regression(historical_df)
    coeff_df.to_csv(MODEL_COEFFICIENTS_OUT, index=False)
    print(f"  Saved coefficients to {MODEL_COEFFICIENTS_OUT}")

    # OLS is run separately to get p-values and confidence intervals for the blog post.
    ols_df = fit_ols_regression(historical_df)
    ols_df.to_csv(OLS_RESULTS_OUT, index=False)
    print(f"  Saved OLS results to {OLS_RESULTS_OUT}")

    # Weights are derived from positive ridge coefficients only, then normalised to sum to 1.
    regression_weights = derive_regression_weights(coeff_df)

    print("  Applying qualifying thresholds...")
    df = apply_qualifying_thresholds(current_df)

    df = calculate_composite_score(df, regression_weights, "mvp_score_datadriven")

    mvp_table = build_mvp_scores_table(df)
    mvp_table.to_csv(MVP_SCORES_OUT, index=False)
    print(f"  Saved MVP scores to {MVP_SCORES_OUT}")

    # Stage 3: causal forest runs on the qualifying 2022-23 players only.
    print("\n[Stage 3] Causal Forest")
    causal_df = run_causal_forest(df)
    causal_df.to_csv(CAUSAL_EFFECTS_OUT, index=False)
    print(f"  Saved causal effects to {CAUSAL_EFFECTS_OUT}")


if __name__ == "__main__":
    main()
