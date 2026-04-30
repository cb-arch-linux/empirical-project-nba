---
title: The Most Controversial MVP of the Decade
tags: NBA, basketball analytics, MVP, data science
description: Embiid won. The data says otherwise.
---

# The Most Controversial MVP of the Decade
### A Statistical Analysis of the 2022-23 NBA Most Valuable Player

**University of Exeter | BEE2041 Empirical Project**

---

Only three players in NBA history have ever won three consecutive MVP awards: Bill Russell, Wilt Chamberlain, and Larry Bird. Going into the 2022-23 season, Nikola Jokić had a chance to become the fourth and only the second player to do it in the modern media-vote era. Despite a historic season he was beaten to it by the scoring prowess of Joel Embiid.

But why? Even casual fans questioned it - Jokić received only 15 first-place votes despite a dominant season. Embiid scored 33.1 points per game, the highest average since Michael Jordan in 1987. The narrative was simple: Embiid finally beat his injury demons and put together a great season on a decent team.

But scoring 33 points a game doesn't make you the most valuable player. It makes you the highest usage player who is also very good at basketball. Those aren't the same thing. This analysis builds a clear analytical picture - using seven seasons of historical MVP voting data, three custom impact metrics, and a causal forest model - to find out who actually deserved the award.

The answer is not Embiid.

---

## The Problem with Points Per Game

The most cited number: points per game. It dominates MVP debates because it's easy to understand. 33 points sounds better than 24.5. But points per game is mainly a product of how many shots you take. And not all shots are good shots.

The chart below shows the top 20 scorers in 2022-23 coloured by usage rate - the share of team possessions a player used while on the court. Embiid led with 33.1 points, but his bar is almost black: a usage rate of 0.37, meaning over a third of every Philadelphia possession ran through him. Giannis Antetokounmpo, fifth on the scoring list at 31.1 points, has an equally dark bar. Both score a lot because both players have an enormous share of their team's offence.

![Top 20 scorers in 2022-23 coloured by usage rate. The darkest bars - the highest usage players - are almost always at the top of the scoring charts.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig1_ppg_usage.png)

Further down the list, players such as Kevin Durant and Stephen Curry score nearly as much on significantly less usage. A player who scores 29 points on 30% usage is doing something fundamentally different from one who scores 33 on 37%.

This becomes clearer when you replace points with Player Impact Estimate (PIE) - the NBA's own metric capturing each player's share of positive contributions across the full box score.

![Scoring volume against Player Impact Estimate. Many high scorers sit bottom right - heavy volume, below-average overall impact.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig2_ppg_vs_pie.png)

Jokić scores 24.5 points - not even registering in the top 20 - yet his PIE of 0.211 is the highest in the entire league. He produces the most positive impact per possession of anyone in basketball, while barely appearing in the scoring conversation.

---

## Before We Score: Who Qualifies?

Not every player should be in MVP contention regardless of their individual numbers. Three thresholds are applied before scoring, each grounded in voting reality.

First, a **team win percentage of at least 50%**. In the entire history of the award, only three players have won MVP on a team below a 61% win rate - Moses Malone twice in the late 1970s and early 1980s, Russell Westbrook in 2016-17, and Jokić in 2021-22.

Second, a **net rating of at least -2.0**. Players whose teams perform well below average with them on the court are not in contention.

Third, a **usage rate of at least 26%**. Every MVP winner from 2015-16 through 2022-23 exceeded this threshold. Role players whose metrics are inflated by elite teammates are excluded here.

---

## Building Better Metrics

Points, rebounds, and assists tell you what happened. They don't tell you why, or how efficiently. To answer those questions properly, this analysis constructs three custom metrics, adapted from the Box Plus Minus methodology developed by Daniel Myers at Basketball Reference (Myers, 2014).

The key difference from raw box score stats is that these metrics use **rate statistics** - percentages of possessions, rebounds, and assists - rather than per-game totals. Rate stats account for usage. They also adjust for pace, so players on faster teams aren't rewarded for playing in more possessions.

**Custom BPM** estimates each player's net impact per 100 possessions. The formula combines steal rate, block rate, assist rate, rebound rates, and true shooting percentage relative to the league average. The net rating component measures each player's contribution *above their own team's baseline*. This matters because players on elite teams - like Aaron Gordon playing alongside Jokić in Denver - would receive inflated scores simply from having a superstar teammate. Gordon's net rating of 11.6 looks elite; once adjusted for Denver's team baseline, it becomes much more ordinary.

**Custom VORP** (Value Over Replacement Player) converts Custom BPM into a cumulative season value by comparing each player to replacement level - the level a team could achieve by signing anyone freely available from the waiver wire - set at -2.0 points per 100 possessions, following Basketball Reference's convention (Myers, 2014). A player with a VORP of 26 contributed 26 points per 100 possessions more than replacement over the full season, scaled by minutes played.

**Custom WS/48** (Win Shares Per 48 Minutes) estimates wins contributed per 48 minutes by combining offensive and defensive rating differentials relative to league averages, pace adjusted. A value above 0.100 is above average; elite seasons sit around 0.200.

The chart below plots Custom VORP against Custom BPM for all qualifying players, with bubble size representing team win percentage. BPM tells you how impactful a player was per possession; VORP tells you how much of that impact was sustained across the full season. A player in the top right is both elite per possession and delivered it consistently. One player redefines the scale of the chart.

![Custom VORP against Custom BPM. Bubble size represents team win percentage. Jokić sits in a class of his own - top right, on a winning team, by a significant margin.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig3_vorp_vs_bpm.png)

Jokić's Custom BPM of 52.3 is the highest in the league by a wide margin. Embiid sits at 37.5. On a per-possession basis (BPM) and sustained across the whole season (VORP), accounting for role and efficiency, Jokić is not just better - he is in a class of his own.

---

## What History Says Actually Predicts MVP Quality

Custom metrics are only as credible as the methodology behind their weighting. Rather than assign weights based on intuition, this analysis trains a **ridge regression model** on seven seasons of historical MVP voting data - 2015-16 through 2021-22 - where each player-season is labelled as MVP calibre based on actual top-3 finishers that year. The model learns which metrics statistically separate those seasons from the rest in the modern era.

The model is trained only on data *before* the 2022-23 season. It never sees the season it is being asked to score. This prevents data leakage - where information from the future contaminates the model used to evaluate the subject.

A supplementary OLS regression tests statistical significance, providing p-values for each coefficient. The results are shown below.

![OLS regression coefficients from historical MVP data (2015-16 to 2021-22). Blue bars are significant at the 5% level. Custom BPM is the strongest positive predictor - by far.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig7_ols_coefficients.png)

Custom BPM is the strongest positive predictor of MVP quality (coefficient 0.046, p < 0.001). Team win percentage and usage rate are also significantly positive - the historical reality is that MVPs always come from winning teams and efficiently carry a heavy offensive load.

The most striking finding is what is *not* significant: **PIE**, the NBA's own composite impact metric, fails to reach the 5% threshold (p = 0.077). The NBA's official impact metric doesn't reliably predict MVP calibre seasons.

One variable missing from the regression is net rating. It was tested but removed because custom WS/48 is algebraically derived from the same offensive and defensive rating differentials - including both in the same model creates multicollinearity. Custom WS/48 is retained as the cleaner formulation since it is pace-adjusted and expressed as wins per 48 minutes. The fact that custom WS/48 itself carries a negative coefficient, despite measuring team impact, is worth noting. It reflects the same teammate contamination problem that BPM solves, where average players on winning teams benefit from their environment.

These regression-derived weights are then used to build a composite MVP score for every qualifying 2022-23 player. Each metric is normalised on a scale between 0 and 1. The weights derived from the regression are applied to only the scaled metrics which had positive coefficients. These are then summed to produce a score between 0 and 1. The result is a single number that captures, in the proportions that history says matter, who had the most MVP-calibre season.

---

## The Rankings

Applying the regression-derived weights to all qualifying 2022-23 players produces the following composite scores.

![2022-23 MVP composite rankings. Weights derived from ridge regression trained on 2015-16 to 2021-22. Jokić leads by a meaningful margin.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig4_mvp_scores.png)

| Rank | Player | Team | TS% | USG% | AST% | Custom BPM | Custom VORP | Custom WS/48 | MVP Score |
|------|--------|------|-----|------|------|------------|-------------|--------------|-----------|
| 1 | Nikola Jokić | DEN | 0.701 | 0.263 | 0.412 | 52.3 | 26.3 | 0.121 | **0.825** |
| 2 | Giannis Antetokounmpo | MIL | 0.605 | 0.373 | 0.314 | 36.8 | 16.3 | 0.073 | **0.767** |
| 3 | Joel Embiid | PHI | 0.655 | 0.370 | 0.233 | 37.5 | 18.8 | 0.097 | **0.733** |
| 4 | Luka Dončić | DAL | 0.609 | 0.368 | 0.408 | 38.9 | 20.4 | 0.024 | **0.653** |
| 5 | Ja Morant | MEM | 0.557 | 0.338 | 0.412 | 31.8 | 13.7 | 0.066 | **0.564** |

Jokić leads with a score of 0.825. His 70.1% true shooting percentage is one of the most efficient seasons ever recorded by an offensive leader. His 41.2% assist percentage is extraordinary - typically a point guard number - produced by a centre who also leads the league in Custom BPM. His Denver team won 69.6% of their games.

No other player in the top five comes close on all three metrics at the same time. Embiid's individual case is strong - Custom BPM of 37.5, true shooting of 65.5% - but his team won fewer games than Denver, Milwaukee, or Phoenix, and his usage of 37% means a lot more of those numbers come from volume than Jokić's 26.3%.

Giannis ranks second because Milwaukee had the best team record of any top candidate (74.6% wins). The regression says team success matters, and precedent backs that up.

---

## Impact vs Efficiency

The chart below places every qualifying player on two of the strongest predictors of MVP quality simultaneously - Custom BPM (net impact per 100 possessions) on the x-axis and true shooting percentage on the y-axis. The top-right quadrant is where genuine MVP candidates live: high impact and high efficiency.

![Custom BPM against true shooting percentage. Top right is where MVP candidates should be - high impact and high efficiency. Jokić sits alone.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig5_ws48_vs_net_rating.png)

Jokić occupies the top right entirely alone. Embiid is clearly excellent but sits noticeably below Jokić on both axes. Giannis and Dončić cluster together at high BPM but lower efficiency. Morant is high BPM but below-average true shooting. No one combines elite impact and elite efficiency the way Jokić does.

---

## Does Piling On Possessions Actually Help?

The final question this analysis asks is the most direct test of the Embiid argument: does giving a player more of the ball cause them to be more efficient, or do their counting stats just reflect the volume of possessions?

If Embiid genuinely improves the more the offence runs through him, then his 33-point average reflects real dominance. If usage doesn't cause efficiency gains, or even hurts them, then those numbers are partly a product of volume.

To answer this question, a **Double Machine Learning Causal Forest** is fitted using the `econml` package (a Python library for causal inference). The treatment variable is usage rate; the outcome is true shooting percentage. True shooting percentage is used rather than a composite metric because composite metrics like PIE have circular relationships with usage. True shooting percentage is independent: it measures how efficiently a player converts their shots regardless of how many they take.

The DML approach removes the influence of confounding player characteristics - their passing rates, rebounding rates, defensive metrics and custom BPM - then estimates how much usage itself drives shooting efficiency. This is the key difference from a simple correlation: it asks not whether high-usage players happen to be efficient, but whether *increasing a specific player's usage would cause their efficiency to change*.

![Causal forest estimates of the effect of usage rate on true shooting percentage for top MVP candidates. Wide confidence intervals reflect uncertainty - the data cannot isolate individual causal effects from a single season with confidence.](https://raw.githubusercontent.com/cb-arch-linux/empirical-project-nba/main/output/figures/fig6_causal_effects.png)

The confidence intervals are wide - wide enough that the model cannot confidently say whether more usage helps or hurts any individual candidate. This is an honest finding rather than a failure of the method. Isolating causal effects from observational data in a single NBA season is difficult; too many confounding factors move together for the model to separate them. What the wide intervals tell us is that the causal effect of usage on efficiency is uncertain, which is itself informative: the MVP case cannot be built on any single causal relationship, but is instead a combination of dimensions that together paint a picture of *value*.

All point estimates are positive. But given the uncertainty, this should be treated as directional rather than a definitive finding.

---

## The Verdict

The data says Nikola Jokić should have won the 2022-23 MVP.

His MVP score of 0.825 leads the field. His Custom BPM of 52.3 is the highest in the league by a distance. His team went 16-4 in the playoffs and Jokić averaged 30.2 points, 14.0 rebounds and 7.2 assists in the Finals on his way to his first ring and Finals MVP. The regular season voters, it turned out, had picked the wrong man.

Embiid's 33.1 points per game is one of the most impressive scoring seasons in modern NBA history. But the scoring average that made his case so compelling in public debate is exactly what this analysis argues is misleading. The metrics trained on historical MVP data reward something harder to see but more important: how much better your team gets when you are on the court. Whether this turns out to be his only MVP, the statistical case against the 2022-23 vote only grows stronger with time.

The reluctance to give Jokić a third consecutive award - despite the numbers - is perhaps the clearest example of human decision making that is so hard to capture in statistics.

The data doesn't know about voter fatigue. It just knows who was the most valuable.

---

## Data and Methodology

All data was sourced from the NBA API via the `nba_api` Python package in accordance with [NBA.com Terms of Use](https://www.nba.com/termsofuse). Historical training data covers 2015-16 through 2021-22. Scoring data covers 2022-23. Custom impact metrics are adapted from Myers (2014) and Oliver (2004). Ridge regression and causal forest models were implemented using `scikit-learn` and `econml`. All code and replication instructions are available at [github.com/cb-arch-linux/empirical-project-nba](https://github.com/cb-arch-linux/empirical-project-nba).

---

## References

Myers, D. (2014). *Introducing box plus/minus (BPM)*. Sports Reference. https://www.sports-reference.com/blog/2014/10/introducing-box-plusminus-bpm-2/

Oliver, D. (2004). *Basketball on paper: Rules and tools for performance analysis*. Potomac Books.

Sports Reference. (2020, February 25). *Introducing BPM 2.0*. https://www.sports-reference.com/blog/2020/02/introducing-bpm-2-0/