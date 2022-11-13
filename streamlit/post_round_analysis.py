import sys
import streamlit as st

sys.path.append('.')

from src.EspnFantasyRoundAnalysis import EspnFantasyRoundAnalysis  # noqa: E402
from src.utils.app_utils import parameter_checks  # noqa: E402
from src.utils.get_logger import get_logger  # noqa: E402

CURRENT_SEASON = 2023
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

st.title("Post round analysis for ESPN NBA Fantasy leagues")

app_desc_body = """
    This app assesses the teams' performance of a past round. This is based on
    a league with *Head to Head Each Category* scoring type and 9 stat
    categories (FG%, FT%, 3PM, REB, AST, STL, BLK, TO, PTS).

    * It displays the 8 stat categories in addition to total minutes and games
        played by each team in a single table.
    * The ranking of the above stat categories, i.e. how each team is ranked
        for each category.
    * The score differential for all possible match-ups in the round. It gives
        a sense of how a team performed compared to the whole league, not just a
        single match-up.
    * The percentage wins of all possible match-ups. For example, a team might
        have won its match-up, but its percentage win is very low. This
        indicates a weak/lucky team which played against an even weaker
        opponent. Changes are still required. On the other hand, if a team lost
        a match-up, but its percentage win is high, it indicates a strong team
        which happened to play against the strongest opponent. Changes might
        not be required.

    Further details on [this Medium blog post](https://g-giasemidis.medium.com/nba-fantasy-analytics-with-python-on-epsn-f03f10a60187).

    Use this *public* league id `10149515` and year 2022 for trying out app.
    No need for `swid` and `espn_s2` cookies. This league is based on the same
    nine aforementioned stats, but uses a *Head to Head Points* scoring system.
    Here, the league is emulated as if the scoring system was
    "Head to Head Each Category".

    Report bugs and issues
    [here](https://github.com/giasemidis/espn-nba-fantasy/issues).
"""  # noqa: E501

with st.expander("App Description"):
    # st.header("")
    st.markdown(app_desc_body)


st.header("League Parameters")
app_parm_body = """
    Fill in the parameters for the app to run and click on "Submit" button.
    Check the help button for further details and how to identify the cookies
    and the league id.
"""
st.text(app_parm_body)

swid_help = """
    If your league is public, leave blank. Otherwise: In Safari these
    cookies can be found by opening the Storage tab of Developer tools
    (developer tools can be turned on in preferences), and looking
    under espn.com in the Cookies folder. In Chrome, they can be found
    in Settings -> Cookies and other site data -> See all cookies and
    site data, and search for ESPN.
"""
espn_s2_help = """
    If your league is public, leave blank. Otherwise, follow above
    instructions.
"""
league_id_help = """
    Go to your ESPN league page. The URL should contain something like
    `leagueId=12345678`. Copy and paste the number next to the `leagueid=`
    parameter.
"""
season_help = f"""
    Leave this to current season, i.e. {CURRENT_SEASON}, unless you want to run
    analysis of a round of a previous season.
"""
round_help = """
    If provided, data extraction is faster. It is the day since the start of
    the season. To find the scoring period of the round under consideration, go
    to "Scoreboard" on ESPN, select the matchup round of interest and read the
    number next to `mSPID=` in the url.
"""


# parameters
with st.form(key='my_form'):
    swid = st.text_input('swid (cookie)', "", help=swid_help)
    espn_s2 = st.text_input('espn_s2 (cookie)', "", help=espn_s2_help)
    league_id = st.number_input(
        'The ID of the ESPN league', format='%d',
        min_value=0, help=league_id_help)

    season = st.number_input(
        'The season of the league',
        value=CURRENT_SEASON,
        min_value=2019,
        max_value=CURRENT_SEASON,
        step=1,
        help=season_help
    )

    week = st.number_input(
        "The round for analysis",
        value=0,
        min_value=0,
        max_value=100,
        step=1,
        help=round_help
    )

    scoring_period = st.number_input(
        "Scoring period (Optional)",
        value=0,
        min_value=0,
        max_value=500,
        step=1,
        help=round_help
    )
    scoring_period = None if scoring_period == 0 else scoring_period

    submit_button = st.form_submit_button(label='Submit')


# Code
logger = get_logger(__name__)
parameter_checks(logger, swid, espn_s2, league_id)
cookies = {
    "swid": swid,
    "espn_s2": espn_s2
}
league_settings = {
    "league_id": int(league_id),
    "season": season,
}


if submit_button:
    with st.spinner('We are doing the clever stuff'):
        espn = EspnFantasyRoundAnalysis(
            cookies=cookies,
            league_settings=league_settings,
            round=week,
            scoring_period=scoring_period
        )
        adv_stats_df = espn.get_adv_stats_of_round().astype(DTYPES)
        adv_stats_rank_df = espn.compute_stats_ranking_of_round()
        h2h_df = espn.compute_h2h_score_table()
        win_ratio_df = espn.win_ratio_in_round()

    st.text("Navigate across the tabs to access the different analysis tables")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Round Stats", "Stat Ranking", "Head to Head", "% Wins"]
    )

    with tab1:
        st.header("Stats of the round for all fantasy teams")
        st.text(
            """
            9 basic ESPN fantasy stats, in addition to total minutes and games
            for the round under consideration.
            """
        )
        st.table(data=adv_stats_df)

    with tab2:
        st.header("Ranking index for each statistical category")
        st.text(
            """
            Ranking of the teams for the aforementioned statistical categories
            """
        )
        st.table(data=adv_stats_rank_df)

    with tab3:
        st.header("Head to Head scores for all possible match-ups in the round")
        st.text(
            """
            Score differential for the round under consideration of each team in
            the league against each other team.
            """
        )
        st.table(data=h2h_df)

    with tab4:
        st.header("Percentage wins from the H2H matchups")
        st.text(
            """
            Based on the previous head to head scores, what is the percentage of
            winsfor and average differential score for each team?
            """
        )
        st.table(data=win_ratio_df)

    st.success('Done!')