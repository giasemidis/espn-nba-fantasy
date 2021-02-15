# espan-nba-fantasy
ESPN NBA fantasy League analysis and predictions. Projections of upcoming match-ups and analysis of league results

# Configuration

To use the ESPN Fantasy API, the `config/config_temp.json` must be first configured. This is a template file, fill in the relevant fields as described above, then rename the file to `config/config.json`. For privacy and security do *not* share the filled in configuration file.

## Cookies
For private leagues, the user must configure the cookies fields, `swid` and `espn_s2`, which can be found in this [blob-post](https://stmorse.github.io/journal/espn-fantasy-3-python.html). You can find these cookies in Safari by opening the Storage tab of Developer tools (you can turn on developer tools in preferences), and looking under espn.com in the Cookies folder. In Chrome, you can go to `Preferences -> Advanced -> Content Settings -> Cookies -> See all cookies and site data`, and looking for ESPN.

## League
Next, the league details should be filled in. These are:

*   `league_id`. The id of the league.
*   `season`. The end year of the season, e.g. for season 2020-2021, use the year 2021.
*   `n_active_players`. The number of active players (non-bench) in a match-up.

## URL

URL links do not require changes.

# Head to Head
Analysis and projections of upcoming head-to-head match-ups.

Commence the jupyter notebook `head2head_weekly_analysis.ipynb` and change the week, start date and team abbreviations accordingly.

Prior to a head-to-head matchup, a fantasy player needs to know the following details:

*   How many games each team is playing.
*   Are there any players in the bench that do not fit in the roster on a any given day?

The number of games is an important piece of information, as many categories are aggregate totals of raw statistics, such as rebounds, assists, points, etc, hence the more the games, the merrier. In addition, players who do not fit in the roster and are forced to stay in the bench in a given day are a waste of resources, the user should consider making changes to optimise the schedule of his/her players during the matchup.

Furthermore, we project the matchups outcome using two methods:
1.   A naive projection method, which is based on each fantasy team's accumulate statistics throughout the season. However, this method has the following drawbacks. First, the current teams might have changed considerably compared to the past teams that achieved these statistics, due to trades or weekly roster changes. Second, NBA game schedule plays a crucial role in the outcome of a head-to-head matchup. This naive method does not consider the fantasy teams' current roster or schedule
2.   A simulation. We run a simulation which replicates 100,000 realisations of the matchup by sampling from the players' mean statistics and following their schedule throught the week.

# Post-round analysis
Analysis of league results of past round.

Commence the notebook `post_round_analysis.ipynb`. Change the week/round and scoring period accordingly.
