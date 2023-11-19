CURRENT_SEASON = 2024
DTYPES = {
    "FG%": float,
    "FT%": float,
    "3PM": int,
    "REB": int,
    "AST": int,
    "STL": int,
    "BLK": int,
    "TO": int,
    "PTS": int,
    "Mins": int,
    "Games": int
}

SWID_HELP = """
    If your league is public, leave blank. Otherwise: In Safari these
    cookies can be found by opening the Storage tab of Developer tools
    (developer tools can be turned on in preferences), and looking
    under espn.com in the Cookies folder. In Chrome, they can be found
    in Settings -> Cookies and other site data -> See all cookies and
    site data, and search for ESPN.
"""

ESPN_S2_HELP = """
    If your league is public, leave blank. Otherwise, follow above
    instructions.
"""

LEAGUE_ID_HELP = """
    Go to your ESPN league page. The URL should contain something like
    `leagueId=12345678`. Copy and paste the number next to the `leagueid=`
    parameter.
"""

SEASON_HELP = f"""
    Leave this to current season, i.e. {CURRENT_SEASON},
    unless you want to run analysis of a round of a previous season.
"""

ROUND_HELP = """
    A positive integer. It must be a current or past round, as this app
    assesses the performance of the fantasy teams in completed rounds
"""

SCORING_PERIOD_HELP = """
    If provided, data extraction is faster. It is the day since the start of
    the season. To find the scoring period of the round under consideration, go
    to "Scoreboard" on ESPN, select the matchup round of interest and read the
    number next to `mSPID=` in the url.
"""

SCENARIO_HELP = """
    To add or remove a player, use the following format:
    ```
    "<player_name1>": ["<date1>", "<date2>"],
    "<player_name2>": ["<date1>"]
    ```
    where replace the player name and the dates in `<...>`.
    For example:
    ```
    "Nikola Jokic": ["2023-10-25", "2023-10-26"]
    ```
    Leave blank if no player is added or removed
"""
