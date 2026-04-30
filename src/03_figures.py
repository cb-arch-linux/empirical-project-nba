"""
Loads the cleaned player data and analysis outputs and generates all figures
and tables for the blog post.

Fig 1: Bar chart of top 20 scorers with usage percentage as colour scale.
Fig 2: Scatter of points per game against PIE.
Fig 3: Scatter of custom VORP against custom BPM, sized by team win percentage.
Fig 4: Horizontal bar chart of composite MVP scores for the top 15 players.
Fig 5: Scatter of custom BPM against true shooting percentage with top candidates highlighted.
Fig 6: Dot plot of causal forest treatment effects for top MVP candidates.
Fig 7: Horizontal bar chart of OLS regression coefficients with significance markers.

Outputs: output/figures/fig1_ppg_usage.png, fig2_ppg_vs_pie.png,
fig3_vorp_vs_bpm.png, fig4_mvp_scores.png, fig5_ws48_vs_net_rating.png,
fig6_causal_effects.png, fig7_ols_coefficients.png, output/tables/top10_display.csv
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configuration

CLEAN_DATA_DIR = os.path.join("data", "clean")
TABLES_OUTPUT_DIR = os.path.join("output", "tables")
FIGURES_OUTPUT_DIR = os.path.join("output", "figures")

PLAYERS_CLEAN = os.path.join(CLEAN_DATA_DIR, "current_clean.csv")
MVP_SCORES = os.path.join(TABLES_OUTPUT_DIR, "mvp_scores.csv")
OLS_RESULTS = os.path.join(TABLES_OUTPUT_DIR, "ols_results.csv")

TOP_CANDIDATES = 5
FIGURE_DPI = 150

BLUE = "#1d428a"
RED = "#c8102e"
GREY = "#aaaaaa"
CANDIDATE_COLOURS = ["#1d428a", "#c8102e", "#007a33", "#f58426", "#5c2d91"]


# Custom Metric Calculations (from 02_analysis.py)
# These functions are duplicated here so that 03_figures.py is self contained
# and does not depend on importing from 02_analysis.py. Any changes to the
# metric formulas in 02_analysis.py should be reflected here too.

REPLACEMENT_LEVEL = -2.0


def calculate_custom_bpm(df):
    """Custom Box Plus Minus approximation. See 02_analysis.py for full methodology."""
    stl_pct = df["steals_per_game"] / df["minutes_per_game"] * 100
    blk_pct = df["blocks_per_game"] / df["minutes_per_game"] * 100
    ast_pct = df["ast_pct"] * 100 if df["ast_pct"].max() <= 1 else df["ast_pct"]
    oreb_pct = df["oreb_pct"] * 100 if df["oreb_pct"].max() <= 1 else df["oreb_pct"]
    dreb_pct = df["dreb_pct"] * 100 if df["dreb_pct"].max() <= 1 else df["dreb_pct"]
    usg_pct = df["usage_pct"] * 100 if df["usage_pct"].max() <= 1 else df["usage_pct"]
    ts_pct = df["true_shooting_pct"] * 100 if df["true_shooting_pct"].max() <= 1 else df["true_shooting_pct"]
    league_avg_ts = ts_pct.mean()

    raw_bpm = (
        (stl_pct * 0.9)
        + (blk_pct * 0.75)
        + (dreb_pct * 0.35)
        + (oreb_pct * 0.6)
        + (ast_pct * 0.6)
        + (usg_pct * (2.0 * (ts_pct - league_avg_ts) / 100))
        - (usg_pct * df["turnovers_per_game"] / (df["points_per_game"] + 0.001) * 0.5)
    )
    team_avg_net = df.groupby("team")["net_rating"].transform("mean")
    individual_net_above_team = df["net_rating"] - team_avg_net
    return raw_bpm + individual_net_above_team * 0.3


def calculate_custom_vorp(df):
    """Custom Value Over Replacement Player approximation. See 02_analysis.py for full methodology."""
    minutes_played = df["minutes_per_game"] * df["games_played"]
    return (df["custom_bpm"] - REPLACEMENT_LEVEL) * (minutes_played / 4800)


def calculate_custom_ws48(df):
    """Custom Win Shares Per 48 Minutes approximation. See 02_analysis.py for full methodology."""
    league_avg_off = df["off_rating"].mean()
    league_avg_def = df["def_rating"].mean()
    pace_factor = df["pace"] / df["pace"].mean()
    off_contribution = (df["off_rating"] - league_avg_off) * pace_factor
    def_contribution = (league_avg_def - df["def_rating"]) * pace_factor
    return (off_contribution + def_contribution) / 100


def add_custom_metrics(df):
    df = df.copy()
    df["custom_bpm"] = calculate_custom_bpm(df)
    df["custom_vorp"] = calculate_custom_vorp(df)
    df["custom_ws48"] = calculate_custom_ws48(df)
    return df


# Helper Functions

def apply_figure_style():
    # whitegrid adds light horizontal grid lines; all other settings override matplotlib defaults.
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        # Remove top and right borders for a cleaner, modern chart appearance.
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(figure, filename):
    """Save a matplotlib figure to the figures directory as a PNG file."""
    output_path = os.path.join(FIGURES_OUTPUT_DIR, filename)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")  # bbox_inches="tight" crops to content, removing blank whitespace.
    plt.close(figure)  # Frees memory - without this, matplotlib holds all figures open.
    print(f"  Saved: {output_path}")


def add_source_note(fig, source_text):
    # fig.text uses figure coordinates (0–1); (0.5, -0.02) places text centred just below the axes.
    fig.text(0.5, -0.02, source_text, ha="center", fontsize=8, color=GREY)


# Figure 1: PPG with Usage Overlay

def plot_ppg_usage(players_df):
    """Top 20 scorers bar chart coloured by usage rate."""
    print("Generating Figure 1: PPG with usage overlay...")

    top_scorers = (
        players_df
        .nlargest(20, "points_per_game")
        .sort_values("points_per_game", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    # Normalize maps usage values to [0, 1] so they can be passed to a colormap.
    norm = plt.Normalize(top_scorers["usage_pct"].min(), top_scorers["usage_pct"].max())
    # YlOrRd converts each normalised value to a yellow-orange-red colour.
    colours = plt.cm.YlOrRd(norm(top_scorers["usage_pct"]))

    bars = ax.barh(
        top_scorers["player_name"], top_scorers["points_per_game"],
        color=colours, edgecolor="white", linewidth=0.5
    )

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])  # Required by matplotlib to attach the colour scale to the colorbar.
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)  # shrink=0.8 makes the colorbar 80% of the axis height.
    cbar.set_label(
        "Usage Rate (proportion of team possessions,\ne.g. 0.30 = 30%)",
        fontsize=9
    )

    for bar, ppg in zip(bars, top_scorers["points_per_game"]):
        # get_y() + get_height()/2 calculates the vertical centre of each bar for label placement.
        ax.text(ppg + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{ppg:.1f}", va="center", fontsize=9)

    ax.set_xlabel("Points Per Game")
    ax.set_title(
        "Top 20 Scorers by Points Per Game\n"
        "Coloured by Usage Rate (share of team possessions used)",
        fontweight="bold", pad=12
    )
    ax.set_xlim(0, top_scorers["points_per_game"].max() * 1.12)  # 1.12 adds padding so value labels don't clip at the axis edge.
    add_source_note(fig, "Source: NBA API (2022-23 Regular Season)")
    plt.tight_layout()
    save_figure(fig, "fig1_ppg_usage.png")


# Figure 2: PPG vs PIE

def plot_ppg_vs_pie(players_df, top_candidate_names):
    """Scatter of points per game against PIE, with top candidates highlighted."""
    print("Generating Figure 2: PPG vs PIE scatter...")

    if "pie" not in players_df.columns:
        print("  Skipping: PIE column not found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    is_candidate = players_df["player_name"].isin(top_candidate_names)

    # zorder controls layering - candidates (zorder=4) are drawn on top of background points (zorder=2).
    ax.scatter(
        players_df.loc[~is_candidate, "points_per_game"],
        players_df.loc[~is_candidate, "pie"],
        color=GREY, alpha=0.4, s=40, label="All other players", zorder=2
    )

    for i, name in enumerate(top_candidate_names):
        row = players_df[players_df["player_name"] == name]
        if row.empty:
            continue
        colour = CANDIDATE_COLOURS[i % len(CANDIDATE_COLOURS)]
        ax.scatter(row["points_per_game"], row["pie"],
                   color=colour, s=120, zorder=4,
                   edgecolors="white", linewidths=1.2)
        # xytext=(6, 4) offsets the label 6px right and 4px up from the data point.
        ax.annotate(name.split(" ")[-1],
                    xy=(row["points_per_game"].values[0], row["pie"].values[0]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=colour)

    # axvline/axhline draw league-average reference lines so readers can see above/below average.
    ax.axvline(players_df["points_per_game"].mean(),
               color=GREY, linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(players_df["pie"].mean(),
               color=GREY, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Points Per Game")
    ax.set_ylabel("Player Impact Estimate (PIE)")
    ax.set_title("Scoring Volume vs Overall Impact\nPoints Per Game against PIE",
                 fontweight="bold", pad=12)

    # mpatches.Patch creates a coloured rectangle for the legend - used because scatter
    # points were plotted separately per candidate rather than as a single labelled series.
    patches = [mpatches.Patch(color=CANDIDATE_COLOURS[i % len(CANDIDATE_COLOURS)], label=n)
               for i, n in enumerate(top_candidate_names)
               if n in players_df["player_name"].values]
    patches.append(mpatches.Patch(color=GREY, alpha=0.5, label="All other players"))
    ax.legend(handles=patches, loc="upper left", framealpha=0.9)  # framealpha=0.9 gives a slightly transparent legend background.

    add_source_note(fig, "Source: NBA API (2022-23 Regular Season)")
    plt.tight_layout()
    save_figure(fig, "fig2_ppg_vs_pie.png")


# Figure 3: Custom VORP vs Custom BPM

def plot_vorp_vs_bpm(players_df, top_candidate_names):
    """Scatter of custom VORP vs custom BPM, bubble size = team win percentage."""
    print("Generating Figure 3: Custom VORP vs Custom BPM...")

    required = ["custom_vorp", "custom_bpm", "team_win_pct"]
    if not all(c in players_df.columns for c in required):
        print("  Skipping: required columns not found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    is_candidate = players_df["player_name"].isin(top_candidate_names)
    others = players_df[~is_candidate]

    # zorder controls layering - candidates (zorder=4) are drawn on top of background points (zorder=2).
    ax.scatter(
        others["custom_bpm"], others["custom_vorp"],
        s=(others["team_win_pct"] * 200).clip(20, 200),  # clip prevents extreme bubble sizes from outlier win percentages.
        color=GREY, alpha=0.35, zorder=2
    )

    for i, name in enumerate(top_candidate_names):
        row = players_df[players_df["player_name"] == name]
        if row.empty:
            continue
        colour = CANDIDATE_COLOURS[i % len(CANDIDATE_COLOURS)]
        ax.scatter(row["custom_bpm"], row["custom_vorp"],
                   s=row["team_win_pct"].values[0] * 400,
                   color=colour, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(name.split(" ")[-1],
                    xy=(row["custom_bpm"].values[0], row["custom_vorp"].values[0]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=colour)

    ax.axvline(0, color=GREY, linewidth=1, linestyle="-", alpha=0.4)
    ax.axhline(0, color=GREY, linewidth=1, linestyle="-", alpha=0.4)
    ax.set_xlabel("Custom BPM (net impact per 100 possessions)")
    ax.set_ylabel("Custom VORP (cumulative value over replacement)")
    ax.set_title("Custom Impact Metrics with Team Context\n"
                 "Bubble size represents team win percentage",
                 fontweight="bold", pad=12)

    # Empty scatter calls ([], []) create legend entries showing bubble size without adding data.
    size_handles = [
        plt.scatter([], [], s=v * 400, color=GREY, alpha=0.5, label=l)
        for v, l in [(0.4, "40% wins"), (0.6, "60% wins"), (0.8, "80% wins")]
    ]
    ax.legend(handles=size_handles, loc="upper left", framealpha=0.9)

    add_source_note(fig, "Source: NBA API (2022-23). Custom metrics derived from box score stats.")
    plt.tight_layout()
    save_figure(fig, "fig3_vorp_vs_bpm.png")


# Figure 4: MVP Scores

def plot_mvp_scores(mvp_scores_df):
    """Horizontal bar chart of composite MVP scores for the top 15 players."""
    print("Generating Figure 4: MVP scores bar chart...")

    score_col = "mvp_score_datadriven"
    if score_col not in mvp_scores_df.columns:
        print("  Skipping: mvp_score_datadriven column not found.")
        return

    plot_df = (
        mvp_scores_df
        .head(15)
        .sort_values(score_col, ascending=True)
        .copy()
    )

    # Last bar in the ascending-sorted list is the top-ranked player - highlight it.
    colours = [
        BLUE if i == len(plot_df) - 1 else GREY
        for i in range(len(plot_df))
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        plot_df["player_name"], plot_df[score_col],
        color=colours, edgecolor="white", linewidth=0.5
    )

    for bar, score in zip(bars, plot_df[score_col]):
        # +0.003 nudges the label past the bar end; get_height()/2 centres it vertically.
        ax.text(
            score + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=9
        )

    ax.set_xlabel("Composite MVP Score (0 to 1)")
    ax.set_title(
        "2022-23 NBA MVP Rankings\n"
        "Composite score weighted by historical ridge regression",
        fontweight="bold", pad=12
    )
    ax.set_xlim(0, plot_df[score_col].max() * 1.20)

    add_source_note(
        fig,
        "Source: NBA API (2022-23). Weights derived from ridge regression "
        "trained on 2015-16 to 2021-22 seasons."
    )
    plt.tight_layout()
    save_figure(fig, "fig4_mvp_scores.png")


# Figure 5: BPM vs True Shooting

def plot_ws48_vs_net_rating(players_df, top_candidate_names):
    """Scatter of custom BPM against true shooting percentage, candidates highlighted."""
    print("Generating Figure 5: Custom BPM vs True Shooting...")

    if "custom_bpm" not in players_df.columns or "true_shooting_pct" not in players_df.columns:
        print("  Skipping: required columns not found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    is_candidate = players_df["player_name"].isin(top_candidate_names)

    # zorder controls layering - candidates (zorder=4) are drawn on top of background points (zorder=2).
    ax.scatter(
        players_df.loc[~is_candidate, "custom_bpm"],
        players_df.loc[~is_candidate, "true_shooting_pct"],
        color=GREY, alpha=0.4, s=40, zorder=2,
        label="All other players"
    )

    for i, name in enumerate(top_candidate_names):
        row = players_df[players_df["player_name"] == name]
        if row.empty:
            continue
        colour = CANDIDATE_COLOURS[i % len(CANDIDATE_COLOURS)]
        ax.scatter(
            row["custom_bpm"], row["true_shooting_pct"],
            color=colour, s=120, zorder=4,
            edgecolors="white", linewidths=1.2
        )
        ax.annotate(
            name.split(" ")[-1],
            xy=(row["custom_bpm"].values[0], row["true_shooting_pct"].values[0]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, fontweight="bold", color=colour
        )

    # axvline/axhline draw league-average reference lines so readers can see above/below average.
    ax.axvline(players_df["custom_bpm"].mean(),
               color=GREY, linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(players_df["true_shooting_pct"].mean(),
               color=GREY, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Custom BPM (net impact per 100 possessions)")
    ax.set_ylabel("True Shooting Percentage")
    ax.set_title(
        "Impact vs Efficiency - Where Do MVP Candidates Stand?\n"
        "Custom BPM against True Shooting Percentage",
        fontweight="bold", pad=12
    )

    patches = [mpatches.Patch(color=CANDIDATE_COLOURS[i % len(CANDIDATE_COLOURS)], label=n)
               for i, n in enumerate(top_candidate_names)
               if n in players_df["player_name"].values]
    patches.append(mpatches.Patch(color=GREY, alpha=0.5, label="All other players"))
    ax.legend(handles=patches, loc="lower right", framealpha=0.9)

    add_source_note(
        fig,
        "Source: NBA API (2022-23). Custom BPM derived from rate-based box score stats."
    )
    plt.tight_layout()
    save_figure(fig, "fig5_ws48_vs_net_rating.png")


# Figure 6: Causal Effects

def plot_causal_effects(causal_effects_df, top_candidate_names):
    """Dot plot of causal forest treatment effects with 95% CIs, top candidates only."""
    print("Generating Figure 6: Causal effects dot plot (MVP candidates only)...")

    required = ["player_name", "causal_effect", "ci_lower", "ci_upper"]
    if not all(c in causal_effects_df.columns for c in required):
        print("  Skipping: required columns not found.")
        return

    plot_df = (
        causal_effects_df[causal_effects_df["player_name"].isin(top_candidate_names)]
        .sort_values("causal_effect", ascending=True)
        .copy()
    )

    if plot_df.empty:
        print("  Skipping: none of the top candidates found in causal effects table.")
        return

    # Height scales with number of players; minimum of 4 inches for a single player.
    fig, ax = plt.subplots(figsize=(11, max(4, len(plot_df) * 1.5)))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        name = row["player_name"]
        effect = row["causal_effect"]
        ci_low = row["ci_lower"]
        ci_high = row["ci_upper"]

        # Look up by name to keep colours consistent with other figures regardless of sort order.
        candidate_index = next(
            (j for j, n in enumerate(top_candidate_names) if n == name), 0
        )
        colour = CANDIDATE_COLOURS[candidate_index % len(CANDIDATE_COLOURS)]

        ax.errorbar(
            effect, i,
            xerr=[[effect - ci_low], [ci_high - effect]],  # Asymmetric CI: distance from effect to lower bound, then to upper bound.
            fmt="o", color=colour, markersize=10,
            elinewidth=2, capsize=5, zorder=4,
            label=name
        )

    ax.axvline(0, color=GREY, linewidth=1.5, linestyle="--", alpha=0.8)

    # Use player names as y-axis tick labels so there is no annotation overlap.
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["player_name"].tolist(), fontsize=11)

    # Set symmetric x limits so zero sits in a meaningful position.
    max_abs = max(
        abs(plot_df["ci_lower"].min()),
        abs(plot_df["ci_upper"].max())
    ) * 1.1
    ax.set_xlim(-max_abs, max_abs)

    # Add direction labels inside the plot area at the top rather than
    # below the axis where they clash with x-axis labels.
    ylim = ax.get_ylim()
    label_y = ylim[1] - (ylim[1] - ylim[0]) * 0.05
    ax.text(
        -max_abs * 0.05, label_y,
        "← More usage, lower efficiency",
        fontsize=9, color=GREY, va="top", ha="right"
    )
    ax.text(
        max_abs * 0.05, label_y,
        "More usage, higher efficiency →",
        fontsize=9, color=GREY, va="top", ha="left"
    )

    ax.set_xlabel(
        "Estimated Causal Effect of Usage Rate on True Shooting %\n"
        "(change in TS% per one unit increase in usage rate)"
    )
    ax.set_title(
        "Does More Usage Hurt Efficiency? - MVP Candidates\n"
        "Causal Forest: Effect of Usage Rate on True Shooting %",
        fontweight="bold", pad=12
    )

    add_source_note(
        fig,
        "Source: NBA API (2022-23). Causal effects estimated via Double ML "
        "Causal Forest (econml) trained on all qualifying players. Wide "
        "confidence intervals reflect uncertainty in individual "
        "treatment effects."
    )
    plt.tight_layout()
    save_figure(fig, "fig6_causal_effects.png")


# Figure 7: OLS Coefficients

def plot_ols_coefficients(ols_results_df):
    """Bar chart of OLS coefficients with 95% CIs, coloured by significance at 5%."""
    print("Generating Figure 7: OLS coefficients chart...")

    required = ["metric", "coefficient", "std_error", "significant_5pct"]
    if not all(c in ols_results_df.columns for c in required):
        print("  Skipping: required columns not found.")
        return

    plot_df = ols_results_df.sort_values("coefficient", ascending=True).copy()
    colours = [
        BLUE if sig else GREY
        for sig in plot_df["significant_5pct"]
    ]

    ci_95 = plot_df["std_error"] * 1.96  # 1.96 is the z-score for a 95% confidence interval.

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        plot_df["metric"], plot_df["coefficient"],
        color=colours, edgecolor="white", linewidth=0.5, alpha=0.85
    )
    ax.errorbar(
        plot_df["coefficient"],
        range(len(plot_df)),
        xerr=ci_95,
        fmt="none",  # fmt="none" draws error bars only - the bars are already plotted above.
        color="#333333",
        elinewidth=1.2,
        capsize=4,
        zorder=4
    )

    ax.axvline(0, color=GREY, linewidth=1.2, linestyle="--", alpha=0.8)

    sig_patch = mpatches.Patch(color=BLUE, label="Significant at 5% level")
    insig_patch = mpatches.Patch(color=GREY, alpha=0.85, label="Not significant at 5% level")
    ax.legend(handles=[sig_patch, insig_patch], loc="lower right", framealpha=0.9)

    ax.set_xlabel(
        "Standardised OLS Coefficient\n"
        "(change in probability of MVP calibre per 1 SD increase in metric)"
    )
    ax.set_title(
        "Which Metrics Historically Predict MVP Quality?\n"
        "OLS Coefficients with 95% Confidence Intervals (2015-16 to 2021-22)",
        fontweight="bold", pad=12
    )

    add_source_note(
        fig,
        "Source: NBA API (2015-16 to 2021-22). OLS Linear Probability Model. "
        "Ridge regression used for composite score weights."
    )
    plt.tight_layout()
    save_figure(fig, "fig7_ols_coefficients.png")

# Display Table

def build_display_table(mvp_scores_df):
    print("Building top 10 display table...")

    rename_map = {
        "rank": "Rank", "player_name": "Player", "team": "Team",
        "true_shooting_pct": "TS%", "usage_pct": "USG%",
        "ast_pct": "AST%", "pie": "PIE", "net_rating": "Net Rtg",
        "team_win_pct": "Win%", "custom_bpm": "Custom BPM",
        "custom_vorp": "Custom VORP", "custom_ws48": "Custom WS/48",
        "mvp_score_datadriven": "MVP Score",
    }
    available = [c for c in rename_map.keys() if c in mvp_scores_df.columns]
    display_df = mvp_scores_df.head(10)[available].rename(columns=rename_map)
    output_path = os.path.join(TABLES_OUTPUT_DIR, "top10_display.csv")
    display_df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


# Main Entry Point

def main():
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_OUTPUT_DIR, exist_ok=True)
    apply_figure_style()

    print("\nLoading data...")
    players_df = pd.read_csv(PLAYERS_CLEAN)
    mvp_scores_df = pd.read_csv(MVP_SCORES)
    causal_effects_path = os.path.join(TABLES_OUTPUT_DIR, "causal_effects.csv")
    causal_effects_df = (
        pd.read_csv(causal_effects_path)
        if os.path.exists(causal_effects_path) else None
    )
    ols_results_df = (
        pd.read_csv(OLS_RESULTS)
        if os.path.exists(OLS_RESULTS) else None
    )
    print(f"  Loaded {len(players_df)} players.")

    players_df = add_custom_metrics(players_df)

    top_candidate_names = mvp_scores_df.head(TOP_CANDIDATES)["player_name"].tolist()
    print(f"  Top candidates: {', '.join(top_candidate_names)}")

    print("\nGenerating figures...")
    plot_ppg_usage(players_df)
    plot_ppg_vs_pie(players_df, top_candidate_names)
    plot_vorp_vs_bpm(players_df, top_candidate_names)
    plot_mvp_scores(mvp_scores_df)
    plot_ws48_vs_net_rating(players_df, top_candidate_names)

    if causal_effects_df is not None:
        plot_causal_effects(causal_effects_df, top_candidate_names)
    else:
        print("  Skipping Figure 6: causal_effects.csv not found.")

    if ols_results_df is not None:
        plot_ols_coefficients(ols_results_df)
    else:
        print("  Skipping Figure 7: ols_results.csv not found.")

    build_display_table(mvp_scores_df)


if __name__ == "__main__":
    main()
