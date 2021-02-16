# espn-nba-fantasy
ESPN NBA fantasy League analysis and predictions. Projections of upcoming match-ups and analysis of past league results for insights.

# Configuration

To use the ESPN Fantasy API, the `config/config_temp.json` must be first configured. This is a template file, fill in the relevant fields as described below, then rename the file to `config/config.json`. For privacy and security do *not* share the completed configuration file.

## Cookies
For private leagues, the user must configure the cookies fields, `swid` and `espn_s2`, instructions can be found in this [blog-post](https://stmorse.github.io/journal/espn-fantasy-3-python.html). In Safari these cookies can be found by opening the Storage tab of Developer tools (developer tools can be turned on in preferences), and looking under espn.com in the Cookies folder. In Chrome, they can be found in `Settings -> Cookies and other site data -> See all cookies and site data`, and search for ESPN.

## League
Next, the league details should be filled in. These are:

*   `league_id`. The id of the league. To find the league id, go to either `My Team` or `League` tabs and check the URL. The league id is the numeric id after the `leaugId` keyword in the url, e.g. `https://fantasy.espn.com/basketball/league?leagueId=1111111`.
*   `season`. The end year of the season, e.g. for season 2020-2021, use the year 2021.
*   `n_active_players`. The number of active players (non-bench) in a match-up.

## URL

URL links do not require changes.

# Head to Head
Analysis and projections of upcoming head-to-head match-ups.

Commence the Jupyter notebook `head2head_weekly_analysis.ipynb` and change the week, start date and team abbreviations accordingly.

Prior to a head-to-head matchup, a fantasy player needs to know the following details:

*   How many games each fantasy team is playing.
*   Are there any players in the bench that do not fit in the roster on any given day?

The number of games is an important piece of information, as many categories are aggregate totals of raw statistics, such as rebounds, assists, points, etc., hence the more the games, the merrier. In addition, players who do not fit in the roster and are forced to stay in the bench in a given day are a waste of resources, the user should consider making changes to optimise the schedule of his/her players during the matchup.

Furthermore, we project the matchups outcome using two methods.

## Naive Projection
A naive projection method, which is based on each fantasy team's cumulative statistics throughout the season. However, this method has the following drawbacks. First, the current teams might have changed considerably compared to the past teams that achieved these statistics, due to trades or weekly roster changes. Second, the NBA schedule plays a crucial role in the outcome of a head-to-head matchup. More games implies more chances to get rebounds, assists, points, etc. This naive method does not consider the fantasy teams' current roster or schedule.

## Simulation
We run a simulation which replicates 100,000 realisations (repetitions) of the matchup by sampling from the players' mean statistics and following their schedule throughout the week.

The raw statistical categories, such as rebounds, assists, points, etc. are sampled from a Poisson distribution with mean the player's seasonal  mean of each statistical category. For the FG% and FT% we sample from a gaussian distribution with mean the mean FG% and FT% respectively and standard deviation 0.2 times the mean.

We sample for each player in the roster and each game in the schedule during the period of the matchup (usually a week). Injured players are ignored throughout the matchup and day-to-day players are considered as normal. We simulate for both teams. Next, we aggregate the results at the end of the period (taking the sum for the raw statistical categories and the mean for the % ones). We repeat the sampling 100,000 times to ensure that the results are statistically significant.

Finally, we estimate the chance of a team to win each statistical category by taking the ratio of the winning repetitions over the total number of repetitions. For example, if the home team wins the Rebounds category 60,000 times out of 100,000, its probability for winning this category is 0.6 (or chance 60%) (hence the probability of the away team to win is 0.4, or chance 40%).

# Post-round analysis
Analysis of league results of past round.

Commence the Jupyter notebook `post_round_analysis.ipynb`. Change the week/round and scoring period accordingly.

Very often, After a fantasy round, players would like to know how they did compared to the entire league. A player might have lost a matchup, but this could have happened for a number of reasons:
*   Good team, but opponent marginal better
*   Good team, but schedule was very sparse due to injuries, etc.
*   Bad team.
*   How would a fantasy team perform compared to all other teams that week.

For this reason, in the notebook, we perform the following pieces of analyses.

1.  Aggregate statistical categories for the week for each fantasy team, including total minutes and games played. The minutes and games played are important to gauge a team's performance in a category, say rebounds, is due to the many or too few games and minutes played by the players in the roster in that round (on average, the more games played during the round, the more chances to score more points, grab more rebounds, give more assists, etc.)
2.  Ranking of the team for each statistical category. If a team scored the most points during the round, it will rank #1 in the points category.
3.  The score of all possible head to head matchups in the round. This shows which opponent from the league a team could have won and which could have lost from, and the score.
4.  From the head-to-head table, we estimate the percentage of matchups that a team would have won against all other teams in the league, and its average score. If a fantasy team lost a matchup but would have won 90% of all other matchups, it means that the team is still strong, but was unlucky as it played with the strongest opponent in the league. Do not change much. On the hand if a team won the matchup but would have won only 20% of the matchups, it implies that the team got lucky and played with the weakest opponent in the league, so changes should be made. The percentage of wins across the league, puts the win/loss into perspective.
